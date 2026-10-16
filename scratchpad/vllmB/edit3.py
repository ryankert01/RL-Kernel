from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_strict.py")
s = p.read_text()
start = s.index("def _patch_sampler(state: StrictBindingState) -> None:")
end = s.index("# ----------------------------------------------------------------------------------------------\n# Install + worker RPC entry points")
new = '''def check_sampling_request(
    *,
    num_logprobs: int,
    max_token_ids: int,
    top_k: Any,
    top_p: Any,
    logprobs_mode: str,
) -> None:
    """Fail closed on sampler requests whose numbers would be native (see module docstring)."""

    problems = []
    if top_k is not None or top_p is not None:
        problems.append("top-k/top-p filtering (native sort-based kernels)")
    if int(num_logprobs) > 0:
        problems.append(f"logprobs={num_logprobs} (top-k columns would be native log_softmax)")
    if int(max_token_ids) > 0:
        problems.append("logprob_token_ids scoring (native log_softmax columns)")
    if str(logprobs_mode) != "raw_logprobs":
        problems.append(f"logprobs_mode={logprobs_mode!r}")
    if problems:
        raise RuntimeError(
            "strict Muse Glimmer sampler: unsupported request (fail closed): " + "; ".join(problems)
        )


def check_prompt_logprobs_request(num_prompt_logprobs: int, logprobs_mode: str) -> None:
    """Strict prompt logprobs: only the selected-token column (``prompt_logprobs=0``)."""

    if int(num_prompt_logprobs) != 0:
        raise RuntimeError(
            f"strict Muse Glimmer: prompt_logprobs={num_prompt_logprobs} requests native top-k "
            "columns; strict mode supports prompt_logprobs=0 only (fail closed)"
        )
    if str(logprobs_mode) != "raw_logprobs":
        raise RuntimeError(f"strict Muse Glimmer: logprobs_mode={logprobs_mode!r} (fail closed)")


def _patch_sampler(state: StrictBindingState) -> None:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if hasattr(Sampler, _MARK):
        return
    original_call = Sampler.__call__

    def call_wrapped(self: Any, logits: torch.Tensor, input_batch: Any) -> Any:
        ctx = state.take_logits_ctx()
        dims = self.get_logprobs_dims(input_batch.idx_mapping_np)
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            input_batch.expanded_idx_mapping, input_batch.idx_mapping_np
        )
        check_sampling_request(
            num_logprobs=dims[0] if dims is not None else 0,
            max_token_ids=dims[1] if dims is not None else 0,
            top_k=top_k,
            top_p=top_p,
            logprobs_mode=str(self.logprobs_mode),
        )
        if ctx is None or ctx[0].shape[0] != logits.shape[0]:
            raise RuntimeError(
                "strict Muse Glimmer sampler: the logits do not come from the strict "
                "compute_logits of this step (fail closed)"
            )
        output = original_call(self, logits, input_batch)
        logprobs_tensors = getattr(output, "logprobs_tensors", None)
        state.call_counts["sampler.strict_logits_rows"] += int(logits.shape[0])
        if logprobs_tensors is None:
            return output
        targets = output.sampled_token_ids.reshape(-1).to(torch.int64)
        if targets.shape[0] != logits.shape[0] or logprobs_tensors.logprobs.shape[1] != 1:
            raise RuntimeError("strict Muse Glimmer logp: unexpected sampled-token layout")
        logprobs_tensors.logprobs[:, 0] = logp_from_z_tiles(state, ctx[1], targets)
        state.call_counts["sampler.strict_logp_calls"] += 1
        state.call_counts["sampler.strict_logp_rows"] += int(logits.shape[0])
        if state.call_counts["sampler.strict_logp_calls"] == 1:
            state.persist()
        return output

    def init_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        getattr(Sampler, _MARK + "init")(self, *args, **kwargs)
        # Fused FlashInfer / XPU samplers take a different (top-k/top-p) code path; strict mode
        # rejects top-k/top-p anyway, this keeps the Gumbel path the only one.
        self.use_flashinfer = False
        self.use_xpu_sampler = False

    setattr(Sampler, _MARK, original_call)
    setattr(Sampler, _MARK + "init", Sampler.__init__)
    Sampler.__call__ = call_wrapped
    Sampler.__init__ = init_wrapped
    state.hooks.append("vllm.v1.worker.gpu.sample.sampler.Sampler.__call__")
    state.hooks.append("vllm.v1.worker.gpu.sample.sampler.Sampler.__init__ (Gumbel path only)")


def _patch_prompt_logprobs(state: StrictBindingState) -> None:
    from vllm.v1.worker.gpu.sample import prompt_logprob as module

    if hasattr(module, _MARK):
        return
    original = module.compute_prompt_logprobs_with_chunking

    def wrapped(
        prompt_token_ids: torch.Tensor,
        prompt_hidden_states: torch.Tensor,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        num_prompt_logprobs: int,
        logprobs_mode: Any = "raw_logprobs",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        check_prompt_logprobs_request(num_prompt_logprobs, str(logprobs_mode))
        state.take_logits_ctx()
        state.stash_logits = False  # the chunked logits_fn calls must not keep their tiles alive
        try:
            # Ranks: vLLM's comparison count over the strict logits (logits_fn is the patched
            # compute_logits); the score column is overwritten with strict O7 below.
            token_ids, scores, ranks = original(
                prompt_token_ids,
                prompt_hidden_states,
                logits_fn,
                num_prompt_logprobs,
                logprobs_mode,
            )
        finally:
            state.stash_logits = True
        targets = prompt_token_ids.to(torch.int64)
        if scores.shape[1] != 1:
            raise RuntimeError("strict Muse Glimmer prompt logp: unexpected column layout")
        scores[:, 0] = strict_selected_logp(
            state, prompt_hidden_states[: targets.shape[0]], targets
        )
        state.call_counts["prompt_logprobs.strict_logp_calls"] += 1
        state.call_counts["prompt_logprobs.strict_logp_rows"] += int(targets.shape[0])
        if state.call_counts["prompt_logprobs.strict_logp_calls"] == 1:
            state.persist()
        return token_ids, scores, ranks

    setattr(module, _MARK, original)
    module.compute_prompt_logprobs_with_chunking = wrapped
    state.hooks.append(
        "vllm.v1.worker.gpu.sample.prompt_logprob.compute_prompt_logprobs_with_chunking"
    )
    worker_cls = module.PromptLogprobsWorker
    original_token_ids = worker_cls.compute_prompt_token_id_logprobs

    def token_id_logprobs(self: Any, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if self.token_id_scores:
            raise RuntimeError(
                "strict Muse Glimmer: prompt_logprob_token_ids scoring would return native "
                "log_softmax values (fail closed)"
            )
        return original_token_ids(self, *args, **kwargs)

    setattr(worker_cls, _MARK, original_token_ids)
    worker_cls.compute_prompt_token_id_logprobs = token_id_logprobs
    state.hooks.append(
        "vllm.v1.worker.gpu.sample.prompt_logprob.PromptLogprobsWorker."
        "compute_prompt_token_id_logprobs (raises on token-id scoring)"
    )


'''
s = s[:start] + new + s[end:]
old = '''    _patch_sampler(state)
    _patch_prompt_logprobs(state)
    atexit.register(state.persist)'''
new = '''    _patch_sampler(state)
    _patch_prompt_logprobs(state)
    _patch_kv_page_writers(state)
    atexit.register(state.persist)'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")

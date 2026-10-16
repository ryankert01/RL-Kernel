from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('''graph (per-layer activation recompute), checks that the recomputed log-probabilities equal the
returned ones bitwise (counted in ``recompute_logp_mismatches``; the strict path is batch- and
repeat-invariant so the count must be 0), back-propagates ``g`` and accumulates the LoRA
gradients in micro-batch order; then AdamW (``foreach=False``, elementwise per parameter) updates
the fp32 LoRA masters. The gradient norm is computed from the TP-gathered full gradients in
canonical key order, so it is identical at every TP size.
''', '''graph (per-layer activation recompute), checks that the recomputed log-probabilities equal the
returned ones bitwise (the strict path is batch- and repeat-invariant, so they must), and
back-propagates ``g``. In strict mode a non-zero mismatch count FAILS CLOSED: the micro-batch is
not back-propagated, every LoRA gradient of the step is discarded and
:class:`StrictRecomputeMismatch` is raised, so no optimizer step uses it (the native mode only
counts mismatches in ``recompute_logp_mismatches``). LoRA weight gradients follow
``megatron_lora.LORA_GRAD_RULE``: per-sequence det_gemm dW partials folded left to right in
sample order, so they do not depend on ``micro_batch_tokens`` (packing) or the TP size. Then AdamW
(``foreach=False``, elementwise per parameter) updates the fp32 LoRA masters. The gradient norm is
computed from the TP-gathered full gradients in canonical key order, so it is identical at every
TP size.
''')

rep('''from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    DEFAULT_LORA_RANK,
    LayerLoRA,''', '''from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    DEFAULT_LORA_RANK,
    LORA_GRAD_RULE,
    LayerLoRA,''')

rep('''from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    SUPPORTED_TP,
    StrictBindingError,
    _dims_from_manifest,
    install_museglimmer_megatron_strict,''', '''from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    SUPPORTED_TP,
    StrictBindingError,
    _dims_from_manifest,
    drop_vision_modules,
    install_museglimmer_megatron_strict,''')

rep('''DEFAULT_MICRO_BATCH_TOKENS = 4096  # TP1 strict training OOMs at 8192 tokens on A100-80GB
''', '''DEFAULT_MICRO_BATCH_TOKENS = 4096  # memory knob only: LoRA gradient bits do not depend on it
PACKING_RULE = (
    "micro_batches(lengths, micro_batch_tokens): greedy, order-preserving packs of <= budget "
    "tokens (a longer sequence is its own pack); the logprob and LoRA-gradient bits do not "
    "depend on the packing (batch-invariant forward, per-sequence gradient fold)"
)


class StrictRecomputeMismatch(RuntimeError):
    """Strict mode: the recomputed forward differs from the returned log-probabilities."""
''')

rep('''        for group in groups:
            seqs = [token_ids[i] for i in group]
            tokens, cu, positions = pack_sequences(seqs, self.device)
            rows: list[int] = []
            targets: list[int] = []
            offset = 0
            for i in group:
                seq, start = token_ids[i], int(starts[i])
                rows.extend(range(offset + start - 1, offset + len(seq) - 1))
                targets.extend(seq[start:])
                offset += len(seq)
            select = torch.tensor(rows, dtype=torch.int64, device=self.device)
            tgt = torch.tensor(targets, dtype=torch.int64, device=self.device)
            with torch.set_grad_enabled(grad):
                outs.append(self.binding.forward_packed(tokens, cu, positions, select, tgt))
        return outs''', '''        for group in groups:
            seqs = [token_ids[i] for i in group]
            tokens, cu, positions = pack_sequences(seqs, self.device)
            rows: list[int] = []
            targets: list[int] = []
            segments: list[tuple[int, int]] = []
            offset = 0
            for i in group:
                seq, start = token_ids[i], int(starts[i])
                rows.extend(range(offset + start - 1, offset + len(seq) - 1))
                targets.extend(seq[start:])
                segments.append((offset, offset + len(seq)))
                offset += len(seq)
            select = torch.tensor(rows, dtype=torch.int64, device=self.device)
            tgt = torch.tensor(targets, dtype=torch.int64, device=self.device)
            with torch.set_grad_enabled(grad):
                outs.append(
                    self.binding.forward_packed(tokens, cu, positions, select, tgt, tuple(segments))
                )
        return outs''')

rep('''        pend, self._pending = self._pending, None
        grad = grad_packed.to(self.device, torch.float32)
        mismatches = 0
        checked = 0
        offset = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for group, stored in zip(pend["groups"], pend["values"], strict=True):
            (logp,) = self._values(pend["token_ids"], pend["starts"], [group], grad=True)
            mismatches += int((logp.detach().view(torch.int32) != stored.view(torch.int32)).sum())
            checked += stored.numel()
            g = grad[offset : offset + stored.numel()]
            offset += stored.numel()
            logp.backward(g)
            del logp
        torch.cuda.synchronize()''', '''        pend, self._pending = self._pending, None
        grad = grad_packed.to(self.device, torch.float32)
        mismatches = 0
        checked = 0
        offset = 0
        self._sync()
        t0 = time.perf_counter()
        for group, stored in zip(pend["groups"], pend["values"], strict=True):
            (logp,) = self._values(pend["token_ids"], pend["starts"], [group], grad=True)
            bad = int((logp.detach().view(torch.int32) != stored.view(torch.int32)).sum())
            mismatches += bad
            checked += stored.numel()
            if bad and self.strict:
                # Fail closed BEFORE back-propagating this micro-batch: drop every gradient of
                # the step (earlier micro-batches included) so no optimizer step can apply it.
                del logp
                self.cmd_zero_grad()
                raise StrictRecomputeMismatch(
                    f"strict recompute differs from the returned log-probabilities in {bad} of "
                    f"{stored.numel()} values (micro-batch {group}); step aborted, LoRA "
                    "gradients discarded"
                )
            g = grad[offset : offset + stored.numel()]
            offset += stored.numel()
            logp.backward(g)
            del logp
        self._sync()''')

rep('''    def _grad_norm(self) -> float:''', '''    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def cmd_set_micro_batch_tokens(self, tokens: int) -> None:
        if int(tokens) < 1:
            raise ValueError("micro_batch_tokens must be positive")
        self.micro_batch_tokens = int(tokens)

    def _grad_norm(self) -> float:''')

rep('''        else:
            for name in ("vision_tower", "vision_adapter", "vision_projection"):
                if hasattr(self.model, name):
                    delattr(self.model, name)
            for p in self.model.parameters():
                p.requires_grad_(False)
            torch.cuda.empty_cache()
            self.comm = sk.TPComm(_tp_group())''', '''        else:
            self.vision_drop = drop_vision_modules(self.model)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.comm = sk.TPComm(_tp_group())''')

rep('''            "lr": self.lr,
            "micro_batch_tokens": self.micro_batch_tokens,
            "torch_flags": self.flags,
        }''', '''            "lr": self.lr,
            "micro_batch_tokens": self.micro_batch_tokens,
            "packing_rule": PACKING_RULE if self.strict else "one sequence per forward (native)",
            "torch_flags": self.flags,
        }''')

rep('''            out["forward"] = "megatron.bridge MuseGlimmerModel.forward (TE kernels), bf16 logits"
        return out''', '''            out["forward"] = "megatron.bridge MuseGlimmerModel.forward (TE kernels), bf16 logits"
            out["vision_drop"] = getattr(self, "vision_drop", None)
        if self.strict:
            out["lora_grad_rule"] = LORA_GRAD_RULE
            out["recompute_check"] = "bitwise; a mismatch raises StrictRecomputeMismatch (fail closed)"
        return out''')

rep('''            cmd, args, kw = payload[0]
            if cmd == "close":
                break
            getattr(state, f"cmd_{cmd}")(*args, **kw)''', '''            cmd, args, kw = payload[0]
            if cmd == "close":
                break
            try:
                getattr(state, f"cmd_{cmd}")(*args, **kw)
            except StrictRecomputeMismatch:
                # Raised identically on every rank at the same point (the compared values are
                # the TP-gathered logprobs); rank 0 reports it, the worker keeps serving.
                traceback.print_exc()''')

rep('''    def set_lora_enabled(self, flag: bool) -> None:
        self._call("set_lora_enabled", bool(flag))
''', '''    def set_lora_enabled(self, flag: bool) -> None:
        self._call("set_lora_enabled", bool(flag))

    def set_micro_batch_tokens(self, tokens: int) -> None:
        """Change the packing budget (memory knob; the bits do not depend on it)."""
        self._call("set_micro_batch_tokens", int(tokens))
''')

rep('''__all__ = ["DEFAULT_MICRO_BATCH_TOKENS", "StrictTrainer", "micro_batches"]''',
    '''__all__ = [
    "DEFAULT_MICRO_BATCH_TOKENS",
    "PACKING_RULE",
    "StrictRecomputeMismatch",
    "StrictTrainer",
    "micro_batches",
]''')
p.write_text(s)
print("ok")

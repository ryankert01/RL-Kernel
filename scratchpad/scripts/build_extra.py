"""Build the --extra JSON (tests, simulation, NOT RUN items, findings) for vllm_collect_results."""
import json
import sys

S = "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad"
unit_outcome, e2e_outcome, head = sys.argv[1], sys.argv[2], sys.argv[3]
tp_sim = json.load(open(f"{S}/vllm_runs/tp_sim.json"))
WT = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-1"
extra = {
    "status": "complete",
    "commit": head,
    "tests": [
        {
            "command": f"~/bin/gpu-lease -n 1 --pool 0-1 -- env PYTHONPATH={WT} ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q",
            "outcome": unit_outcome,
        },
        {
            "command": f"RL_KERNEL_MUSEGLIMMER_E2E=1 ~/bin/gpu-lease -n 1 --pool 0-1 -- env PYTHONPATH={WT} ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q -k engine",
            "outcome": e2e_outcome,
            "covers": "test_strict_engine_protocol_with_real_lora (RolloutEngineLike signatures, transfer_lora via file, wrong-hash rejection, n=2 sampling, decode==prefill under a rank-16 LoRA on 416 slots, second sync replaces the adapter) + test_strict_engine_decode_equals_prefill",
        },
    ],
    "tp_simulation": {
        "what": "binding per-rank arithmetic of one real-shape decoder layer + padded lm_head at simulated TP 2/4/8 (threads + in-process rank-ordered all-gather on ONE GPU), with and without a rank-16 LoRA; compared with torch.equal to TP1",
        "command": f"~/bin/gpu-lease -n 1 --pool 0-1 -- env PYTHONPATH={WT} ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.vllm_tp_sim",
        **tp_sim,
    },
    "tp4_engine": {
        "status": "NOT RUN",
        "reason": "needs 4 GPUs (pool 0-1 has 2); vLLM maps one TP rank per visible device. The binding arithmetic at TP4/TP8 (KV head replicated 2x/4x, q heads 8/4 per rank, 2/1 vocab tiles per rank, LoRA k/v B rows replicated) is shown equal to TP1 by the single-GPU simulation above; the engine-level pieces it cannot cover are vLLM's paged KV with the full 2 KV heads per rank (Attention factory patch), the vocab-parallel embedding gather and NCCL.",
        "command": f"~/bin/gpu-lease -n 4 --pool 0-7 --contiguous -- env PYTHONPATH={WT} ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 4 --strict --prompt-file reports/muse_glimmer_poc/data/xrt_tokens.json --temperatures 0,1 --max-tokens 64 --prefill-check --lora-random-check --xrt-score reports/muse_glimmer_poc/data/xrt_tokens.json --xrt-out <scratch>/xrt_vllm_strict_tp4.json --xrt-lora-out <scratch>/xrt_vllm_strict_lora_tp4.json --out <scratch>/strict_tp4.json",
        "pass_criterion": "<scratch>/xrt_vllm_strict_tp4.json 'sequences' == reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json 'sequences' (and the _lora_ pair); decode_vs_prefill mismatched == 0",
        "fail_closed": "TP sizes that do not divide the canonical 8 (3, 5, 6, 7) raise at model init ('TP t does not divide the canonical TP 8'); TP > 8 is impossible with 32 q heads / 8 canonical chunks.",
    },
    "findings": {
        "o7_positive_logp": {
            "what": "strict O7 (ops.lm_head_softcap_logprob.softcap_tile_stats Triton kernel) can return logp = +1..2 ulp > 0 for a near-certain argmax token: the tile's pass-2 sum s_t for the tile holding the max is < 1 (min 0.9999990), i.e. exp(zhat - m) of the max element is not exactly 1 (pass-2 zhat differs from the pass-1 max in the last bit; likely FMA contraction of softcap*tanh(..) - m or the approximate tl.exp)",
            "probe": "2048 synthetic rows, dominant logit at the target, 8 vocab tiles",
            "positive_rows": 44,
            "rows": 2048,
            "max_logp": 1.9073486328125e-06,
            "max_tile_s_below_1": 0.9999990463256836,
            "observed_in_30B_runs": "48 of 1024 strict generated-token logprobs (greedy+T1, TP1) are > 0 (max 1.9e-6); 0 of 933 teacher-forced xrt tokens; bitwise identical in the reference chain, at TP1 and TP2, decode and prefill (so it is deterministic, only the sign invariant is violated)",
            "probe_script": f"{S}/scripts/o7_positive.py",
        },
    },
    "plot_hints": [
        "Bar: tokens mismatching between two executions (decode vs prefill; TP2 vs TP1) for native vs strict -> xrt.comparisons.*.mismatched / tokens and generate[*].decode_vs_prefill.mismatched",
        "Per-token |delta logp| line for native TP2 vs TP1 and native decode vs prefill (strict is a flat zero line): xrt.comparisons.native_tp2_vs_native_tp1.per_sequence[*].abs_diff_per_token, generate[*].decode_vs_prefill.abs_diff_per_token",
        "Grouped bars strict vs native at TP1/TP2: throughput[*].tokens_per_s, ttft_ms, decode_step_ms_median, memory_per_rank[*].max_allocated_gib / kv_cache_gib",
        "Line: per-step decode ms over the 256 steps (throughput[*].step_series_rank0[*].ms where tokens == reqs)",
        "Stacked bar: top CUDA kernels of one decode / prefill step (profile[*].decode.top_ops where is_kernel or self_cpu_ms == 0)",
        "Heatmap/grid: LoRA unmerged rule check per (layer, slot): lora_random[*].unmerged_rule[*].slots[*].binding_equals_megatron",
    ],
}
extra["report_markdown"] = open(f"{S}/vllm_runs/report_vllm_runtime_adapter.txt").read()
extra["report_markdown_note"] = (
    "content of rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md (the agent "
    "harness blocks subagents from writing REPORT files; the integrator writes it from here)"
)
json.dump(extra, open(f"{S}/vllm_extra.json", "w"), indent=1)
print("ok")

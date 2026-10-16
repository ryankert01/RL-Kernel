# Muse-Glimmer-30B strict train/rollout POC — report

The report is the illustrated blog page `reports/muse_glimmer_poc/blog/index.html`, built from the data in
`reports/muse_glimmer_poc/data/` by `python3 reports/muse_glimmer_poc/blog/build_blog.py` (sections without data say
"未运行"). The paste-ready list of RFC edits is `reports/muse_glimmer_poc/RFC_CHANGES.md` (20 implementation findings +
56 verified audit edits). Per-row details live in `rl_engine/models/muse_glimmer/**/REPORT_*.md`.

## Headline results (8x A100-SXM4-80GB, vLLM 0.31.0, Megatron Bridge 0.7 @ d86a5c09, Triton 3.7.1, checkpoint a4e59da5)

| Claim | Result | Data |
|---|---|---|
| Same 933 tokens scored by every strict implementation | 8 dumps (reference chain x2, vLLM TP1/2/4, Megatron TP1/2/4) byte-equal; with a random LoRA, vLLM TP1/2/4 byte-equal | data/xrt_summary.json |
| Native implementations on the same tokens | 921-933 of 933 tokens differ pairwise | data/xrt_summary.json |
| Training side across TP | logprobs, LoRA gradients (104.8M elements) and the post-AdamW LoRA hash equal at TP1/TP2/TP4; packing-invariant | data/megatron_runtime.json |
| Rollout self-consistency (decode vs prefill) | strict 0 in every tested shape (12 generate configs, batch 256, 2261+256 window crossing, chunked prefill, KV preemption); native 144-4557 tokens | data/vllm_runtime.json |
| Real 30B LoRA GRPO, 8 steps x 8 prompts x 4 samples x 256 tokens | strict/strict 0/61139 tokens drift; strict/native 42417/62097; native/strict 41764/60935; native/native 36041/62129; weight sync 9/9 everywhere | data/e2e_run.json, runs/ |
| Cost | strict vLLM ~50% of native eager, 9-29% of native CUDA graphs; strict train step ~2-3x | data/vllm_runtime.json, data/megatron_runtime.json |
| Where native drift comes from | any change in bf16 GEMM/attention shape or schedule, or a GPU-built RoPE inv_freq: 918-927/933 tokens | data/ablation.json |

How it was built (two sessions, ~40 agents in waves with adversarial review) and what is left for the community are in
the blog's 过程 and 留给社区 sections; RESUME.md records the step-by-step state.

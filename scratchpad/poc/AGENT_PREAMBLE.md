# Muse-Glimmer-30B strict POC — common briefing for all implementation agents

You are one of ~10 agents implementing, in parallel, the proof-of-concept for GitHub RFC
RL-Align/RL-Kernel#501 ("Muse-Glimmer-30B train–rollout consistency"). The goal of the POC is
to make the WHOLE pipeline work end to end (strict kernels → 52-layer chain → multi-GPU parity
→ vLLM rollout + Megatron Bridge trainer → a short LoRA RL run) and to learn where the RFC is
wrong, NOT to deliver community-grade polish per row. Correctness and the strict contract come
first; performance only has to be adequate (no autotuning sweeps, no exotic tricks).

## Read first (in this order)
1. `docs/contracts/muse-glimmer-30b-strict-contract.md` — THE numerical contract. Your op must
   match it exactly (dtypes, cast points, reduction order, tile constants). If you believe the
   contract is wrong or impossible, implement the closest thing, and FLAG THE DEVIATION in your
   report under "Contract deviations". Never silently diverge.
2. `rl_engine/models/muse_glimmer/museglimmer_manifest.json` and `workload.py` — pinned shapes
   and frozen semantics (ground truth for hyperparameters).
3. `rl_engine/models/muse_glimmer/operator_trace.py` — node names/kinds your op covers.
4. The repo digest section of your brief (base classes, det_gemm API, registration, tests).
5. For Hugging Face semantics (tolerance reference only, NOT a bitwise target):
   `/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/hf/modeling_muse_glimmer.py` (transformers 5.17.0; read-only data).

## Environment
- Python: `~/envs/vllm/bin/python` (3.12; torch 2.13.0+cu130, triton 3.7.1, transformers 5.17.0,
  pytest, safetensors). Run everything with `PYTHONPATH=<your worktree root>`.
  Do NOT pip/uv install anything into `~/envs/*`; if you need a package, say so in your report.
- The compiled CUDA extension `rl_engine._C` is NOT available (`_EXT_AVAILABLE` is False) —
  use/create Triton kernels and PyTorch goldens only. CUDA (.cu) kernels are out of scope.
- GPUs: 8× A100-80GB shared by all agents. NEVER set CUDA_VISIBLE_DEVICES yourself and NEVER run
  GPU work without a lease:  `~/bin/gpu-lease -n 1 -- <command>`  (pool is GPUs 0–5; `-n 2
  --contiguous` for 2-GPU tests; GPUs 6–7 are reserved — do not use them unless your brief says).
  Keep leases short (one pytest run, one benchmark); do not hold a lease while editing code.
  If the lease times out, other agents are busy — wait and retry, do not bypass it.
- Checkpoint (60 GB, may still be downloading): `~/models/Muse-Glimmer-30B/`. It is complete
  only when `~/models/Muse-Glimmer-30B/REVISION` exists. Use real weights only if your brief
  says so; otherwise use random tensors of the real shapes.
- TF32 must be disabled in every test (`torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`); see `rl_engine/validation/operators/accelerator.py`.
- Triton compile cache (`~/.triton`) is shared and safe.

## Isolation and ownership — this is how we avoid fighting each other
- You work in your OWN git worktree (already created for you; `git rev-parse --show-toplevel`
  shows it) on your own branch. Commit there. NEVER push. NEVER use `git stash` (the stash is
  shared across worktrees). Never `cd` into the main checkout or another worktree.
- You may create/modify ONLY the files listed under "Files you own" in your brief, plus new
  files under your own directories. Everything else is read-only — in particular these shared
  files are OFF LIMITS: `rl_engine/validation/operators/operator_specs.py`,
  `rl_engine/runtime/registry.py`, `rl_engine/kernels/registry.py`,
  `rl_engine/contracts/profiles/precision/ws1.json`, `.github/**`, `pyproject.toml`, `setup.py`,
  `rl_engine/models/muse_glimmer/{workload.py,museglimmer_manifest.json,loader.py,operator_trace.py}`,
  `docs/contracts/*`. If you need a registration line or a tolerance row, write the exact
  snippet into your REPORT.md under "Integration requests" and the integrator applies it.
- Layout for new code (model-scoped on purpose — the community will later move rows into the
  generic op tree following `.claude/skills/ws1-single-card-kernel/SKILL.md`):
    rl_engine/models/muse_glimmer/ops/<op>.py        golden (`forward_fp32`) + strict op wrapper
    rl_engine/models/muse_glimmer/triton/<op>.py     Triton kernels
    tests/models/muse_glimmer/test_<op>.py           pytest (G1–G5 gates of the contract)
    benchmarks/models/muse_glimmer/bench_<op>.py     strict vs native timing on real shapes
    rl_engine/models/muse_glimmer/ops/REPORT_<op>.md your report
  Package `__init__.py` files: create `rl_engine/models/muse_glimmer/ops/__init__.py` and
  `.../triton/__init__.py` ONLY as empty SPDX-header files if missing (identical content across
  agents avoids merge conflicts); never put imports in them.
- Reuse existing repo code by importing it (det_gemm, embedding backward, logprob tiles,
  collectives...). Do not copy-paste large repo files into your directory.

## Engineering rules
- SPDX header `# SPDX-License-Identifier: Apache-2.0` + copyright line as in existing files;
  black line length 100; ruff E/F/B clean (`~/envs/vllm/bin/python -m ruff` is not installed —
  keep lines ≤100 and imports clean by hand); type hints; docstrings must state reduction order
  and cast points.
- Every op exposes `forward_fp32`, `forward`, deterministic `backward` (autograd.Function),
  and `trace()` as in the contract R4. No autotune. No `torch.matmul`/SDPA/cuBLAS inside the
  strict path (the golden may use `torch.mv`-style per-row ops but must be fixed-order).
- Tests: small shapes for breadth + at least one test at the real Muse Glimmer shape. Parity
  gates use `torch.equal`. Correctness gates use tolerances from
  `rl_engine/contracts/profiles/precision/ws1.json` (read-only) or, if no row fits, state the
  tolerance you used and why.
- Benchmark once at the end, on real shapes, strict vs the native torch equivalent (bf16
  `torch.matmul`/SDPA/etc.), report ms and the ratio. Adequate is fine; note the obvious
  next optimization but do not do it.
- Time-box: aim to finish in a few hours of work. If something blocks you for >45 minutes,
  write it down and move on to the rest; a partial op with an honest report beats a stall.

## Deliverables
1. Code + tests committed on your branch (one or a few commits; message prefix
   `poc(muse_glimmer): <row>`; end the message with the attribution lines
   `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
   `Claude-Session: https://claude.ai/code/session_01U3SxjVAWpLJB5L47T5WgDc`).
2. `rl_engine/models/muse_glimmer/ops/REPORT_<op>.md` with sections: Summary; Design (reduction
   order, casts, tiles); Contract deviations; Test commands and results (verbatim pytest output
   tail); Parity results (batch / sequence-split / repeat); Benchmark (table); Integration
   requests (exact registration snippets); Open issues; RFC feedback (what the RFC row text
   should say differently); Time spent.
3. Your final message: branch name, HEAD SHA, the list of files, and a ≤25-line summary of the
   report (pass/fail per gate, numbers, deviations). The integrator merges from your branch.

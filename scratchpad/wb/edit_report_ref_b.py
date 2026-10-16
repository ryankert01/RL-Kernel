import pathlib
import sys

p = pathlib.Path("rl_engine/models/muse_glimmer/reference/REPORT_full_model_chain.md")
s = p.read_text()
GOLDEN_CHAIN = sys.argv[1] if len(sys.argv) > 1 else "NOT RUN"
PROF_TABLE = pathlib.Path(sys.argv[2]).read_text().strip() if len(sys.argv) > 2 else ""

start = s.index("STATUS (follow-up, wave A, 2026-10-15)")
end = s.index("## Follow-up (wave A): scoring, re-run, profiling")
NEW = '''STATUS (wave B, 2026-10-15): complete. Every measurement of this row was re-run at the final code
(`39053b2`; golden runs from a clean `git archive` of `615ee61`, which differs only by added scripts),
after the O7 fix (c88607f) that changed strict Triton logprob bits. The shared token file is scored by
the strict reference on both backends (`xrt_reference_{triton,golden}.json`, 933 tokens each) and every
token is byte-equal across train / one-shot prefill / prompt prefill + decode / chunked prefills:
triton 48 (mode, sequence) checks incl. W−1/W/W+1 = 2047/2048/2049 (these three apply only to the
2439-token sequence), golden 28 checks incl. W = 2048 only. A second triton process gives the same 933
bits. The real-weights pytest gates pass on the triton op set: the 256+32 chain and the new batched-decode
gate (batch 1 == 2 == permuted 4 == reversed 16 == committed 16 with page eviction, and decode == train).
The 512+32 and 2100+32 triton chains with the HF anchor and the golden 2100+32 chain are re-run. The
single-GPU op-class profile (strict triton vs HF bf16 sdpa) is re-measured with the paged-prefill memory
fixed and 3 profiler passes per config (`reports/muse_glimmer_poc/data/reference_profile.json`). Tiny gates:
21/21 in both envs.

## Wave B review follow-ups (issues reference_profile:R1..R12)

Machine-readable: `reference_profile.json` → `notes.wave_b_review` (verdicts, the tiny tests before/after
with the reason each failed before, the real-weights pytest result) and `measured_commits` (derived from
every run file's own commit field, with `includes_o7_fix`).

| id | verdict | change | evidence |
|---|---|---|---|
| R1 | fixed | `KVPlan.validate()`: ids unique and in `[0, cache.max_seqs)`, equal `seqs`/`starts`/`lens` lengths, `lens >= 1`, `starts` == committed lengths; `decode_batch` checks the ids before reading lengths; every paged forward validates its plan (and that `positions` are the plan's) before any K/V write | `test_decode_batch_and_kv_plans_reject_bad_sequence_ids`: before (f91e8b6) duplicate ids ran (row 0 off by 0.018 on the tiny model, 0.23 on real weights per the review) → FAILED; after: `ValueError`, cache K/V, block table and lengths untouched |
| R2 | fixed | `measure(prepare=, release=)`: the paged-prefill KV cache is allocated per call outside the timed region, one alive at a time; `prepared_state_gib` = the one cache inside the peak | paged prefill peak 52.25 / 52.67 / 53.23 GiB vs train 52.22 / 52.57 / 53.03 at T = 512 / 2048 / 4096 = exactly one cache (0.026 / 0.102 / 0.204 GiB); wave A reported 52.28 / 53.16 / 54.33 (7 idle caches) |
| R3 | fixed | the collector replaces `kernel_profile.by_group_gpu_ms` with the regrouped values (+ `groups`), so the file holds one GPU grouping; profiles re-measured | `profile.strict_triton.runs[*].kernel_profile.by_group_gpu_ms == runs[*].by_group_gpu_ms` |
| R4 | fixed | real-weights pytest `test_batched_decode_is_batch_invariant_with_eviction` (triton; skipped for golden): 16 sequences, contexts 1..2100 (eviction asserted), 3 steps; per step batch 1 == 2 == permuted 4 == reversed 16 == committed 16 (fp32 bit patterns); committed decode logps of a 2100-token and a 17-token sequence == `logprobs(mode="train")` | `MUSEGLIMMER_CHECKPOINT=… MUSEGLIMMER_BACKEND=triton pytest test_full_model_chain.py`: 2 passed in 39.1 s |
| R5 | fixed | every run regenerated; `measured_commits` derived from the files; golden and HF files regenerated (golden: `commit_note` explains the archive run) | golden bits identical to wave A (933/933); HF bf16 bits identical (933/933) |
| R6 | fixed | xrt checks and `decode_batch_invariance` compare fp32 bit patterns | `test_xrt_scoring_compares_bit_patterns_and_rejects_vacuous_checks`: before, −0.0 vs +0.0 counted equal → FAILED |
| R7 | fixed | `checks.n_checks / n_skipped / n_tokens_compared`; `all_byte_equal_to_train` needs `n_checks > 0`; `run_chain --score-file` exits 1 when no check ran | same test: before, no check → "pass" |
| R8 | fixed | tiny tests assert properties that can fail: regions nest on the GPU timeline (exclusive ≥ 0), instrumented total within [0.5×, 5× + 5 ms] of the clean call, every layer's regions charged to its index; unattributed GPU time ≤ 10 %, GPU busy ≤ 1.25 × wall (one stream); `compare_xrt` against a payload with one flipped mantissa bit | 21/21 pass |
| R9 | fixed | STATUS wording (above) | – |
| R10 | fixed | corrected in place in the wave A text and in the profile notes below | – |
| R11 | fixed | `logprobs(mode="decode", timings=)` and the xrt checks record the prompt prefill and every decode step separately (`decode_step_seconds`, `decode_step_context`) | table below |
| R12 | fixed | the triton repeat is committed (`xrt_reference_triton_repeat.json`, second process, same commit); 3 profiler passes per config (`kernel_profile_repeats`); `KVPlan._seq_index` removed | repeat 933/933; strict GPU-busy spread < 1.1 % |

### Results (wave B; all observed in the runs under "How to reproduce (wave B)")

**Cross-runtime token file (933 scored tokens).**

| backend | commit | checks (mode × sequence) | byte-equal to train | wall | peak | decode s/step (median per sequence; long seq: (decode − prompt prefill)/177) |
|---|---|---|---|---|---|---|
| triton | 39053b2 | 48 (+24 skipped: window chunkings on the 8 short sequences) = 5199 token comparisons; prefill, decode, prompt_boundary, block64, ragged, W−1/W/W+1 | all | 180 s | 54.5 GiB | 0.125–0.129; long: prefill 1.08 s + 24.3 s / 177 = 0.137 |
| triton repeat (2nd process) | 39053b2 | 18 (prefill, decode) | all; 933/933 bits == the first process | 126 s | 54.5 GiB | – |
| golden | 615ee61 (archive) | 28 (+8 skipped) = 2977; prefill, decode, prompt_boundary, W | all | 1041 s | 57.0 GiB | 0.50–1.89; long: prefill 34.8 s + 364.9 s / 177 = 2.06 |

- O7 fix: the regenerated triton logprobs equal the pre-O7 file (f7d3723) on 739/933 tokens (the others
  differ by 1–2 ulp). Golden and HF bf16 bits are unchanged from wave A (933/933 each): neither uses the
  Triton O7 kernel.
- golden vs triton: 39/933 byte-equal (different op sets), max |Δlogp| 0.216, mean 0.0078.
- vs HF bf16 (sdpa, native anchor): 0/933 byte-equal; triton max 0.392, mean 0.0137; golden max 0.246, mean
  0.0147. HF's argmax equals the gold token on 77.2 % of the scored positions.

**Real-weights chain (triton at 39053b2, golden from the 615ee61 archive).**

| run | paths | byte-equal | train / prefill / decode s | peak GiB | HF anchor (completion) max / mean \\|Δlogp\\| | argmax all / completion |
|---|---|---|---|---|---|---|
| triton 512+32 + HF | 7 | 32/32 each | 1.05 / 0.39 / 3.88 | 52.2–52.3 | 0.0656 / 0.0254 | 98.90 % / 100 % |
| triton 2100+32 + HF | 10 incl. W−1/W/W+1 | 32/32 each | 1.74 / 1.10 / 4.26 | 52.2–52.7 | 0.0852 / 0.0187 | 98.08 % / 100 % |
| golden 2100+32 | 10 incl. W−1/W/W+1 | {GOLDEN_CHAIN} | – | – |
| pytest gate 256+32 (triton) | 7 | 32/32 each | 1.13 / 0.29 / 4.67 | 52.1–52.2 | – | – |

**Profile (A100-SXM4-80GB, triton backend; strict vs HF bf16 sdpa; `clean` = median wall of 3 prefill / 5
decode calls without regions; GPU-busy = kernel time, 3 torch.profiler passes).**

{PROF_TABLE}

Notes on the profile (wave B corrections included):
- The node was shared with other agents (load average 335–404 on 240 cores at the end of the run). Wall
  numbers of launch-bound configurations move with host load: strict decode B=1 149 ms here vs 132 ms in
  wave A, HF decode B=1 260 ms vs 130 ms (its B=16 step took 110 ms). Use the GPU-busy column for kernel cost
  (spread over the 3 passes: strict ≤ 1.1 %; HF up to 21.5 % at T = 512 and 2.1 % at T = 4096).
- Paged prefill now costs exactly its own KV cache on top of the train forward (see R2).
- GPU-time shares, forward T=4096: strict ffn_gemm 72.6 %, q/k/v/gate/o 16.7 %, lm_head + softcap + logp
  5.1 %, attention 4.0 % (local 53.2 ms, global 22.7 ms), ffn_act 0.8 %, norms 0.6 %, qknorm+rope 0.1 %;
  `det_gemm_tree_kernel` 1786.6 of 1896.7 ms. HF: ffn_gemm 51.3 %, norms + residual 13.5 %, attention 13.5 %
  (local 151.4 ms, global 13.1 ms), q/k/v/gate/o 11.5 %, lm head 4.9 %, qknorm+rope 3.3 %.
- `det_gemm_tree_kernel` share: 95.9 / 95.1 / 94.2 % (train) and 94.5 / 94.5 / 93.6 % (paged) at T = 512 /
  2048 / 4096. Throughput at T=4096: 217.2 TFLOP / 1.787 s ≈ 121.6 TFLOP/s; HF's cuBLAS GEMMs (GEMM classes,
  808.2 ms) ≈ 268.7 TFLOP/s.
- The GEMM gap (1786.6 − 808.2 = 978 ms) is LARGER than the whole GPU gap (1896.7 − 1222.7 = 674 ms): the
  strict non-GEMM ops are faster in sum (110 vs 414 ms), so the GEMM is the first target.
- Per layer at T=4096: strict local attention 1.36 ms vs HF 3.75 ms (HF's sliding layers pass an explicit
  mask, which rules out a flash kernel); strict GLOBAL attention 1.75 ms vs HF 0.88 ms, i.e. ≈ 2× slower than
  HF's sdpa (memory-efficient) kernel.
- Decode B=16: the reference's per-sequence paged-cache write costs 204.9 ms of wall and 20.7 ms of GPU
  (index kernels) per step, against 29.9 + 10.0 ms (wall) for the attention kernels. That is reference-harness
  overhead, not kernel cost; a single vectorised scatter per layer is expected to remove most of it (not
  measured).

### Provenance (wave B)
- `measured_commits` in `reference_profile.json` is derived from each run file's own `commit` field (taken
  when the run starts; `dirty` = tracked files modified at that moment): triton xrt, repeat, HF xrt, chains
  and profiles at 39053b2 (clean). The golden runs ran from `git archive 615ee61` in the scratchpad so the
  worktree could be edited meanwhile; that copy has no `.git`, so the run's own lookup returned an empty sha
  and the sha was filled in from the archive (`commit_note` in the files). 615ee61 = 39053b2 + scripts.
- Strict code (strict_model, chain_harness, golden_ops, paged_kv, weights, backend, ops/, triton/) is
  identical from 39053b2 to the final commit of this row.
- Removed as stale: `chain_triton_512_hf_pre_change` (307fcf2) and the wave A first triton scoring (69b4409);
  both predate c88607f and are superseded.

### How to reproduce (wave B)
```
WT=$(git rev-parse --show-toplevel); CK=~/models/Muse-Glimmer-30B; R=<runs dir>; L="~/bin/gpu-lease -n 1 --pool 3,7 --"
$L env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer --ignore=tests/models/muse_glimmer/distributed -q
#   -> 653 passed, 8 skipped (at 3404cf9)
$L env PYTHONPATH=$WT MUSEGLIMMER_CHECKPOINT=$CK MUSEGLIMMER_BACKEND=triton ~/envs/vllm/bin/python -m pytest \\
    tests/models/muse_glimmer/test_full_model_chain.py -q -s -rA          # -> 2 passed
$L env PYTHONPATH=$WT ~/envs/vllm/bin/python rl_engine/models/muse_glimmer/reference/run_chain.py --checkpoint $CK \\
    --backend triton --score-file reports/muse_glimmer_poc/data/xrt_tokens.json --score-out $R/xrt_reference_triton.json \\
    --score-check-modes prefill,decode,chunked:prompt_boundary,chunked:block64,chunked:ragged,chunked:window_boundary_2047,chunked:window_boundary_2048,chunked:window_boundary_2049
# repeat (--score-out $R/xrt_reference_triton_repeat.json --score-check-modes prefill,decode), golden
# (--backend golden ... --score-check-modes prefill,decode,chunked:prompt_boundary,chunked:window_boundary_2048),
# HF (python -m rl_engine.models.muse_glimmer.reference.xrt_scoring ...), the chains and the profiles:
# exact command lines in reference_profile.json -> commands
env PYTHONPATH=$WT ~/envs/vllm/bin/python rl_engine/models/muse_glimmer/reference/collect_results.py \\
    --runs $R --commands commands.json --notes notes.json
```

### Plot map additions (wave B)
- Decode cost vs context: `xrt_scoring.{triton,golden}.per_sequence[*].modes.decode.{decode_step_seconds,
  decode_step_context, prompt_prefill_seconds}`.
- GPU-view spread: `profile.<target>.runs[i].kernel_profile_repeats.{gpu_busy_ms, by_group_gpu_ms}`.
- Paged-prefill KV cache inside the peak: `profile.strict_triton.runs[i].prepared_state_gib`.
- Provenance per measurement: `measured_commits[name].{commit, includes_o7_fix}`.

'''
NEW = NEW.replace("{GOLDEN_CHAIN}", GOLDEN_CHAIN).replace("{PROF_TABLE}", PROF_TABLE)
s = s[:start] + NEW + s[end:]
p.write_text(s)
print("ok")

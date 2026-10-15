# REPORT — RFC #501 row `full_model_chain` (Muse-Glimmer-30B strict POC)

STATUS (follow-up, wave A, 2026-10-15): complete. The shared token file is scored by the strict
reference on both backends (`xrt_reference_{golden,triton}.json`, 933 tokens each) and every token is
byte-equal across train / one-shot prefill / prompt prefill + decode / chunked prefills, including the
2047/2048/2049 window boundaries. The real-weights chain still holds on the current branch (512+32 triton and
2100+32 triton and golden, every path byte-equal). Batched decode was added: batch 1 == batch 16 == a permuted
batch of 4, bitwise, on real weights. The single-GPU op-class profile of the strict model and of HF bf16 (sdpa)
is in `reports/muse_glimmer_poc/data/reference_profile.json`. Tiny gates: 19/19 (+ fingerprint 32 passed,
1 skipped).

## Follow-up (wave A): scoring, re-run, profiling

### What was done
1. **Scoring mode** `run_chain.py --score-file <muse_glimmer_xrt_v1> --score-out <path>` (code in
   `xrt_scoring.py`). It computes `log p(token[i] | token[:i])` for `i ∈ [completion_start, len)` with the TRAIN
   forward (varlen attention, every position in one pass). It writes `muse_glimmer_xrt_logprobs_v1`:
   `{schema, runtime: "reference", mode: "train", tp: 1, backend, sequences: [{id, logprobs_hex}]}` with
   `logprobs_hex[k] = '%08x' % uint32(fp32 bits)`. Extra keys (`logprobs` floats, `checks`, `commit`, `source.sha256`)
   are for plots and provenance. In the same run, `--score-check-modes` replays the tokens through other forms
   and compares every token bitwise with the train forward: `prefill` (one-shot paged), `decode` (prompt
   prefill, then one paged decode step per completion token), `per_token`, and `chunked:<prompt_boundary | halves
   | block64 | ragged | window_boundary_<b> | n1/n2/…>`. A chunking that does not fit a sequence is recorded
   as `skipped`.
   `python -m rl_engine.models.muse_glimmer.reference.xrt_scoring` writes the same format for HF bf16 (sdpa) as
   a native anchor (`xrt_reference_hf_bf16.json`). It is not a strict path.
2. **Re-ran the real-weights chain** on the current branch: triton 512+32 with the HF anchor (before and after
   this row's code changes), triton 2100+32 with the HF anchor (new), and golden 2100+32 (the window case from
   the original report). The HF anchor now stores per-position series: strict and HF logp, |Δlogp|,
   max|Δlogit| and argmax agreement.
3. **Batched decode.** `strict_model._run` now takes a `KVPlan` (several sequences of one paged cache packed
   into one call). It is still ONE composition: a `Session` is the one-sequence plan, and single-sequence bits
   are unchanged (the pre- and post-change 512+32 train logp series are identical). New
   `MuseGlimmerStrictModel.decode_batch(cache, seqs, tokens, targets, commit=…)` and `new_batch_cache`.
4. **Profiler** (`profiling.py`, CLI `run_profile.py`). Strict triton model and HF bf16 (sdpa) at prefill
   T ∈ {512, 2048, 4096} (and a strict paged prefill at the same T) and at decode batch 1 and 16 with context
   2048. Two views per configuration:
   - *wall*: CUDA events around every op call in nested regions. Each region gets its exclusive time, and the
     median is taken over 3 prefill / 5 decode repeats. A region also absorbs the CPU launch gaps inside it.
   - *gpu*: one extra call under `torch.profiler`. Each kernel's device time is charged to the innermost region
     that was open when its launch API ran (CUPTI correlation id). 100 % of kernels were matched in every
     configuration.

   `clean_total_ms` is the same call without regions. Peak memory is `torch.cuda.max_memory_allocated` over the
   clean repeats. Op classes are embedding, norm, qkvg_gemm, qknorm_rope, attention (global/local), o_gemm,
   ffn_gemm, ffn_act, lm_head_gemm, softcap_logprob, kv_write/kv_evict (strict paged), plus HF-only glue
   (layer_glue = residual adds, attn_glue = transposes, scale, cache update, sigmoid·mul). They are grouped for
   stacked bars as in `profile.stacked_bars`.
5. `collect_results.py` assembles everything into `reference_profile.json` (one object: row, status, commit,
   env, commands, notes, experiments) and copies the xrt files next to it.

### Results (all observed in the runs listed under "How to reproduce")
**Cross-runtime token file (933 scored tokens: 8 GSM8K chat prompts + gold answers; one 2439-token sequence
with completion from 2261, so it crosses W=2048).**

| backend | tokens | train == prefill | train == decode | chunked checks | wall time | peak |
|---|---|---|---|---|---|---|
| triton (final HEAD f7d3723) | 933 | 933/933 | 933/933 | prompt_boundary, block64, ragged 933/933; window_boundary_2047/2048/2049 178/178 (long seq) | 152 s | 54.5 GiB |
| triton (first run, 69b4409) | 933 | 933/933 | 933/933 | prompt_boundary 933/933; window 2047/2048/2049 178/178 | 128 s | 54.5 GiB |
| golden (69b4409) | 933 | 933/933 | 933/933 | prompt_boundary 933/933; window_boundary_2048 178/178 | 1176 s | 57.0 GiB |

- Repeat: the two triton runs (separate processes, different commits) are byte-equal on 933/933 tokens.
- golden vs triton: 32/933 byte-equal (different op sets, so not expected), max |Δlogp| 0.216, mean 0.0078.
- vs HF bf16 (sdpa, native anchor): 0/933 byte-equal. Triton max 0.392, mean 0.0137; golden max 0.246, mean
  0.0147. Per sequence, triton max |Δlogp| is 0.10–0.18 except gsm8k_test_7 (0.39); the long-window sequence
  is 0.12 / mean 0.0099. HF's argmax equals the gold token on 77 % of the scored positions.
- Decode is the slow form: triton 0.11–0.17 s/token (e.g. 29.3 s for the 178 long-window steps at context
  2261–2438), golden 0.6–1.0 s/token for short contexts and 2.3 s/token at 2.3k context (415.6 s for 178 steps).

**Real-weights chain re-run (current branch).**

| run | paths | byte-equal | train / prefill / decode s | peak GiB | HF anchor (completion) max / mean |Δlogp| | argmax all / completion |
|---|---|---|---|---|---|---|
| triton 512+32, before changes (307fcf2) | 7 | 32/32 each | 1.01 / 0.39 / 3.92 | 52.0–52.1 | 0.0656 / 0.0254 | 98.90 % / 100 % |
| triton 512+32, after (69b4409) | 7 | 32/32 each | 1.10 / 0.39 / 3.79 | 52.0–52.1 | 0.0656 / 0.0254 | 98.90 % / 100 % |
| triton 2100+32 (69b4409), new | 10 incl. W−1/W/W+1 | 32/32 each | 2.28 / 1.28 / 5.35 | 52.1–52.6 | 0.0852 / 0.0187 | 98.08 % / 100 % |
| golden 2100+32 (86d1087) | 10 incl. W−1/W/W+1 | 32/32 each | 40.98 / 32.08 / 103.88 | 52.3–54.3 | – | – |

The 512+32 HF numbers equal the original report's (0.836 / 0.257 max |Δlogit| all / completion), and the
pre- and post-change train logp series are identical. At 2132 tokens the HF anchor gives max |Δlogit| 0.637
(all) / 0.395 (completion) and all-position logp max 0.134 / mean 0.0218. Per-layer residual mean |Δ| vs HF-bf16
is 0.020 at layer 9, 0.023 at 19, 0.023 at 29, 0.022 at 39, 0.045 at 49 (rms 1.95 → 5.81). The strict
reference stays as close to HF past the window as before it. Pytest real-weights gate
(`test_full_model_chain.py`, triton, 256+32): 1 passed.

**Batched decode, contract G2 at the model level (real weights).** 16 sequences of 2048 random tokens in one
paged cache. One decode step at batch 16 equals the same step replayed at batch 1 for every sequence (16/16
bitwise), and a permuted batch of 4 (seqs [3, 0, 15, 8]) also matches 4/4. The tiny test checks the same over
3 committed steps with page eviction, against the single-sequence `logprobs(mode="decode")` path.

**Profile (A100-SXM4-80GB, triton backend, `clean_total_ms`; strict / HF bf16 sdpa).**

| config | strict ms | HF ms | strict/HF | strict GPU-busy ms | HF GPU-busy ms | strict peak GiB | HF peak GiB |
|---|---|---|---|---|---|---|---|
| forward T=512 | 269.8 | 149.6 | 1.80 | 251.8 | 139.8 | 52.10 | 56.52 |
| forward T=2048 | 972.8 | 603.5 | 1.61 | 963.3 | 598.3 | 52.44 | 59.41 |
| forward T=4096 | 1938.3 | 1248.1 | 1.55 | 1930.5 | 1249.7 | 52.91 | 63.26 |
| paged prefill T=512/2048/4096 (strict) | 308.0 / 1011.5 / 1980.4 | – | – | 255.6 / 961.6 / 1941.1 | – | 52.28 / 53.16 / 54.33 | – |
| decode B=1, ctx 2048 (per step) | 132.0 | 130.3 | 1.01 | 50.5 | 50.8 | 53.63* | 55.71 |
| decode B=16, ctx 2048 (per step) | 281.9 | 136.1 | 2.07 | 71.8 | 106.5 | 53.63* | 57.69 |

\*The decode peak includes the 16-sequence paged cache (≈1.7 GiB). Strict weights are 51.89 GiB. HF holds
55.55 GiB because it also loads the vision tower. HF's prefill peak grows with T because it materializes
full-vocab bf16 logits and their fp32 log_softmax. The strict head keeps one fp32 25344-column tile alive at a
time.

GPU-time shares (gpu view):
- **forward T=4096, strict:** ffn_gemm 72.6 %, q/k/v/gate/o GEMMs 16.7 %, lm_head + softcap + logp 5.1 %,
  attention 4.0 % (local 54.2 ms, global 23.2 ms), ffn_act 0.8 %, norms 0.6 %, qknorm+rope 0.2 %.
  `det_gemm_tree_kernel` alone is 1817.5 ms of 1930.5 ms.
- **forward T=4096, HF:** ffn_gemm 51.4 %, norms + residual 13.3 %, attention 13.5 % (local 155.2 ms, global
  13.4 ms), q/k/v/gate/o 11.6 %, lm head 5.0 %, qknorm+rope 3.3 %.

Derived GEMM throughput: 2·T·(52 × 483.9 M + 202 752 × 6656) = 217.2 TFLOP at T=4096.
- strict `det_gemm_tree_kernel` in 1.8175 s ≈ **119.5 TFLOP/s** (meets the contract's ≥ 80 TFLOP/s target).
- HF cuBLAS GEMMs in 0.829 s ≈ 262 TFLOP/s.

So the strict path is GEMM-bound and its GEMM is ≈ 2.2× slower than cuBLAS. Strict attention, norms and
qknorm/rope are each faster than HF's eager/sdpa equivalents. Per layer at T=4096:
- strict local 1.39 ms, global 1.79 ms;
- HF sdpa local 3.84 ms, global 0.89 ms (HF's sliding layers pass an explicit mask, and the explicit mask
  rules out a flash kernel).

Decode at batch 1 is launch-bound for both: GPU busy ≈ 38–39 % of the step, 2690 strict kernel launches and
4552 HF. The strict GEMM kernel time at decode (37.5 ms for ≈ 53 GB of weights) is ≈ 1.4 TB/s effective. At
batch 16 the strict step is dominated by the reference's paged-cache WRITE: `PagedKVCache.append` is called
in Python once per sequence per layer. It costs 158.4 ms of wall (kv_cache group) and 20.8 ms of GPU
(index kernels), against 23.8 + 8.0 ms for the attention kernels themselves. That is reference-harness
overhead, not kernel cost; a vLLM-style single scatter per layer removes it.

### Contract deviations / caveats
- Profile wall-view fractions include CPU launch gaps; read the gpu view for kernel cost. Instrumentation
  overhead (instrumented vs clean total) is ≤ 1 % at T ≥ 2048, +9 % at T=512 and +29 %/+12 % at decode B=1/16
  for strict; HF +5 % at T=512 and +51–55 % at decode.
- HF anchors use different attention backends: the chain runs use eager (as in the original report), while
  the xrt anchor and the HF profile use sdpa. Both are tolerance references only.
- Commit provenance: `xrt_reference_golden.json` says 86d1087 in its `commit` field. The run actually started
  from 69b4409: the sha was taken when the file was written, and that bug is fixed in f7d3723. The strict
  code (strict_model, chain_harness core, golden_ops, paged_kv, weights, backend, every ops/triton file) is
  identical from 69b4409 to HEAD. `reference_profile.json` → `notes.measured_commits` lists the commit of
  every run.

### RFC feedback (new)
- `full_model_chain`: make **batched decode** part of the gate (batch 1 == batch N == permuted sub-batch, per
  token, on one op set). It is cheap and caught nothing here, but it is the property rollout engines rely on.
  Also include **window-crossing real text** (here: 2439 tokens, completion from 2261) in the cross-runtime
  token file. Random tokens alone keep attention mass diffuse.
- Cross-runtime file format: `muse_glimmer_xrt_logprobs_v1` (fp32 bit patterns as hex) worked as-is. Add an
  optional `logprobs` float list and a `commit` captured at run start to the schema text.
- Performance contract: on A100 the strict forward is GEMM-bound (94–95 % of GPU time in `det_gemm_tree_kernel`
  at T = 512–4096). The 8-chunk tree GEMM reaches ≈ 120 TFLOP/s against ≈ 260 for cuBLAS, which accounts for
  essentially all of the 1.55× prefill gap to HF bf16. The RFC's performance item should target the GEMM, not
  attention or norms.
- Strict kernels should take host-side metadata (`max_seqlen_q`, sequence lengths) instead of reading it back
  from the device. `strict_gated_attention_fwd_*` calls `.item()` when `max_seqlen_q` is not passed (one sync
  per layer), and decode at batch 1 is ≈ 60 % CPU/launch time. CUDA-graph capture of the strict decode step is
  the obvious next step and needs sync-free wrappers.

### How to reproduce
```
WT=$(git rev-parse --show-toplevel); CK=~/models/Muse-Glimmer-30B; R=<runs dir>
~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_strict_model_tiny.py tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q
# -> 51 passed, 1 skipped (41.7 s)
~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$WT MUSEGLIMMER_CHECKPOINT=$CK MUSEGLIMMER_BACKEND=triton \
    ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_full_model_chain.py -q -s   # -> 1 passed
# scoring (golden: --backend golden, ~20 min; triton ~2.5 min)
~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python \
    rl_engine/models/muse_glimmer/reference/run_chain.py --checkpoint $CK --backend triton \
    --score-file reports/muse_glimmer_poc/data/xrt_tokens.json --score-out $R/xrt_reference_triton.json \
    --score-check-modes prefill,decode,chunked:prompt_boundary,chunked:block64,chunked:ragged,chunked:window_boundary_2047,chunked:window_boundary_2048,chunked:window_boundary_2049
~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m \
    rl_engine.models.muse_glimmer.reference.xrt_scoring --checkpoint $CK \
    --score-file reports/muse_glimmer_poc/data/xrt_tokens.json --score-out $R/xrt_reference_hf_bf16.json
# chain re-runs
... run_chain.py --checkpoint $CK --backend triton --prompt-len 512 --completion-len 32 --hf-compare --json-out $R/chain_triton_512_hf.json
... run_chain.py --checkpoint $CK --backend triton --prompt-len 2100 --completion-len 32 --hf-compare --json-out $R/chain_triton_2100_hf.json
... run_chain.py --checkpoint $CK --backend golden --prompt-len 2100 --completion-len 32 --json-out $R/chain_golden_2100.json
# profiles (one GPU each; ~2 min strict, ~1 min HF)
... run_profile.py --target strict --checkpoint $CK --json-out $R/profile_strict.json
... run_profile.py --target hf --checkpoint $CK --json-out $R/profile_hf.json
# assemble
env PYTHONPATH=$WT ~/envs/vllm/bin/python rl_engine/models/muse_glimmer/reference/collect_results.py \
    --runs $R --commands commands.json --notes notes.json
```
Exact command lines are in `reference_profile.json` → `commands`.

### Plot map for the blog (keys in `reference_profile.json` → `experiments`)
- Stacked bars, strict vs HF per config: `profile.stacked_bars.{strict_triton,hf_bf16_sdpa}.{gpu_ms,wall_ms,
  gpu_frac,wall_frac}[group][config]`, with configs in `profile.stacked_bars.configs`; totals in
  `clean_total_ms`, `gpu_busy_ms`, `peak_allocated_gib`.
- Per-layer bars: `profile.<target>.runs[i].per_layer_ms[class][layer]` (wall) and
  `.kernel_profile.per_layer_gpu_ms` (gpu); per layer type: `per_layer_type_ms` and `per_layer_type_gpu_ms`.
- Per-token drift: `xrt_scoring.{golden,triton,hf_bf16}.logprobs[seq_id]` and
  `xrt_scoring.{golden_vs_triton,triton_vs_hf_bf16,golden_vs_hf_bf16}.per_sequence[*].abs_diff`.
- Equality matrix (form × sequence): `xrt_scoring.{triton,golden}.per_sequence[*].modes`.
- HF anchor per position: `chain_triton_{512,2100}_hf.hf_anchor.series.{abs_dlogp,max_abs_dlogit,argmax_equal}`;
  per-layer residual drift: `hf_anchor.per_layer[*].{mean_abs,max_abs,ref_rms}`.
- Path timing bars: `chain_*.paths[*].{seconds,peak_alloc_gib,n_chunks}`.

### Files (this follow-up)
`reference/{strict_model.py (KVPlan, decode_batch, timer hook), xrt_scoring.py, profiling.py, run_profile.py,
collect_results.py, run_chain.py (--score-*), chain_harness.py (HF series)}`,
`tests/models/muse_glimmer/test_strict_model_tiny.py` (+4 tests),
`reports/muse_glimmer_poc/data/{reference_profile.json, xrt_reference_golden.json, xrt_reference_triton.json,
xrt_reference_hf_bf16.json}`.

---

## Original hand-back (HEAD 09eef9d)

STATUS: complete. Chain gate byte-equal on real weights for both backends; HF bf16 and HF fp32 anchors
measured; tiny gates 15/15; real-weights pytest gate passed (triton backend). (Committed by the integrator
from the implementing agent's hand-back; the agent's harness blocked it from writing REPORT files.)

## Summary
1. Strict 52-layer reference: ONE op composition for train-forward (varlen attention), one-shot/chunked prefill
   and decode via paged KV (page 16, 64-key blocks, window eviction); loads the real checkpoint after
   `verify_checkpoint` (627 bf16 tensors, 51.9 GiB, 11 s).
2. `backend="golden"` = pure-torch fp32 fixed-order fallbacks + repo `TritonDetGemmOp.forward_fp32` GEMM;
   `backend="triton"` = all 12 sibling ops (attention forward included).
3. REAL WEIGHTS, prompt 512 + 32 completion: train == prefill == 4 chunkings (prompt-boundary, halves, 64-aligned,
   ragged incl. 1-token chunks) == decode, **byte-equal 32/32 on BOTH backends**. Golden timings: train 4.7 s,
   prefill 4.1 s, decode 28 s (31 steps); full-Triton: train 1.2 s, prefill 3.0 s (incl. first compile), decode
   3.9 s (0.13 s/step); peak 52.0–52.6 GiB.
4. Prompt 2100 + 32 (golden, W=2048): 10 paths incl. chunk boundaries 2047/2048/2049 all byte-equal (train 36 s,
   decode 89 s, peak 54.3 GiB). Pages are really evicted on local layers (tiny test asserts it).
5. HF anchor (transformers 5.17.0 `MuseGlimmerForConditionalGeneration`, eager, bf16, same 544 tokens): golden
   max|Δlogp| 0.064 / mean 0.022 on the completion, argmax agreement 98.5 % (100 % on the completion);
   full-Triton 0.066 / 0.025 / 98.9 %. max|Δlogit| 0.53 (golden), 0.84 (Triton) — single bf16 outlier channels
   (ulp 2–8).
6. HF fp32 (CPU) cross-check: the strict paths are CLOSER to fp32 than HF-bf16 at every layer: completion
   max|Δlogp| golden 0.041 / Triton 0.046 / HF-bf16 0.046; mean 0.0125 / 0.016 / 0.019; argmax 99.63 / 99.63 /
   98.53 %. Layer-50 residual mean|Δ| 0.040 (golden), 0.040 (Triton), 0.050 (HF-bf16) at rms 7.3; the error grows
   ≈ linearly with depth, steeper after layer 40. A "≤1e-1 logits" bar is met by NO bf16 path, HF's own included
   (0.55 vs fp32) — logp is the robust metric.
7. Golden vs Triton per path: max|Δlogp| 0.048, mean 0.013 (identical for every path, since each backend is
   internally byte-equal).
8. Tiny config ([L,L,L,G], H=256, 8/2 heads, D=32, FFN 512, vocab 1024/1000, W=100): 15/15 tests —
   train == prefill == chunked == decode == per-token `torch.equal`, backward deterministic for all 51 parameters,
   drift localizer verified (injected error → `layers.0.attn`), HF tiny tolerance.

## Design
Files under `rl_engine/models/muse_glimmer/reference/`: `golden_ops.py` (fp32 fixed-order fallbacks with the
sibling names/signatures; `GoldenGemm`; Linear/LmHead holders), `paged_kv.py` (per-layer paged KV, page 16, block
tables, window eviction, reserved zero page), `backend.py` (`resolve_ops("golden"|"triton", allow_partial)` →
`OpSet` with per-node provenance), `weights.py` (`StrictConfig` official/tiny; `MuseGlimmerWeights.load_checkpoint`
/ `.random`), `strict_model.py` (`forward_train`, `prefill(chunk_sizes)`, `decode_step`,
`logprobs(prompt, completion, mode)`, node capture), `chain_harness.py` (`compare_paths`, `localize_first_drift`,
`compare_to_hf`, tiny HF helpers), `run_chain.py` (CLI `--checkpoint --backend --prompt-len --completion-len
--hf-compare --save-dir`); tests in `tests/models/muse_glimmer/`.

Reduction orders / casts (golden backend): row reductions = adjacent-pair halving then sequential (elementwise
only → batch/chunk/decode invariance by construction) [O1, O2, O4, O7]; GEMM = `TritonDetGemmOp.forward_fp32`
(one program per 64×64 tile, K in BLOCK_K=32 `tl.dot(allow_tf32=False)` steps, fp32 accumulate, no split-K, one
rounding; row-parallel = 8 fp32 K-chunk partials combined as ((0+1)+(2+3))+((4+5)+(6+7)) → bf16 once) [O3];
attention = 64-key blocks aligned at 0, ascending from floor(max(0, p−W+1)/64), every block padded to 64 keys,
fixed trees over D=128 and 64 keys, the contract's update order, p→bf16 before PV, o = acc/l → bf16,
y = bf16(fp32(o)·sigmoid(fp32(g))) with sigmoid = 1/(1+exp(−x)), lse = m + log l, m = −inf guard (α = 1, p = 0),
paged gather of 4 pages per block with −1 → zero page [O5]; Q/K norm pow −0.5, q *= 3.87 in fp32, fp32 rotate_half
RoPE with a cached fp32 CUDA table, one cast [O4]; centered norms rsqrt·(1+w), final pow −0.5·w, residual
bf16(fp32 + fp32), fused [F1] inside layers [O2]; SwiGLU bf16(g/(1+exp(−g))·u); lm_head 8 vocab tiles, fp32 logits,
T·tanh((z·m)/T), padded columns −inf, (M_t, S_t) tree [O6, O7].

## Contract deviations
1. Golden GEMM is the repo fp32 FMA Triton kernel (not per-chunk MMA, not a torch loop).
2. Golden attention uses explicit trees, not `tl.dot`.
3. The Triton backend applies the 8-chunk tree to column-parallel GEMMs too (`fast_det_gemm` does it for every K).
4. The tiny-config embedding backward is a local sorted-segment function (the repo canonical one is session-bound).
5. The deliberate fp32-vs-HF differences (single final rounding in the Q/K path, fp32 logits) are why HF is a
   tolerance anchor only.

## Sibling wiring facts the consumers need
- `fast_det_gemm` does the K tree internally → the row-parallel call is a plain call at TP1.
- The sibling lm_head rejects −1 targets → the model masks targetless rows before calling it.
- Paged `seq_lens` and varlen `q_pos` must be int32.
- The sibling attention kernels are hard-wired to 32/2/128 heads → tiny-config tests are golden-only.

## Test commands and results
```
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_strict_model_tiny.py tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q
# → 47 passed, 1 skipped in 27.57s
MUSEGLIMMER_CHECKPOINT=/home/ubuntu/models/Muse-Glimmer-30B MUSEGLIMMER_BACKEND=triton \
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_full_model_chain.py -q -s
# → | decode | - | yes | 32/32 | 0.000e+00 | 3.80 | 52.05 | … 1 passed in 21.76s
~/envs/vllm/bin/python rl_engine/models/muse_glimmer/reference/run_chain.py --checkpoint ~/models/Muse-Glimmer-30B \
    --backend golden|triton --prompt-len 512 --completion-len 32 --hf-compare --json-out out.json
```

## Parity results (real weights)
Golden 512+32 (all equal 32/32): train 4.70 s / 52.55 GiB; prefill 4.13; chunked prompt_boundary [512,31] 4.36;
halves [271,272] 4.93; block64 (9 chunks) 6.66; ragged [1,63,100,7,129,30,…] (11 chunks) 9.46; decode 28.27.
Triton (all sibling ops) 512+32 (all equal 32/32): train 1.20; prefill 2.97 (first compile); prompt_boundary 0.43;
halves 0.44; block64 1.11; ragged 2.60; decode 3.92; peak 52.0–52.1 GiB. Sibling ops + golden attention
(intermediate branch state): also 7/7 equal.
Golden 2100+32 (all equal 32/32): train 35.8; prefill 31.7; prompt_boundary 33.8; halves 36.2; block64 (34) 43.3;
ragged (39) 64.1; window_boundary_2047 [2047,84] 32.6; _2048 [2048,83] 33.5; _2049 [2049,82] 36.1; decode 89.4;
peak 54.3 GiB.
First drift never triggered on real weights; the localizer is verified on the tiny config (injected error →
`layers.0.attn` / attention_local, row 0). Golden vs Triton: max|Δlogp| 0.048, mean 0.013 for every path.

## HF comparison (544 tokens; completion = tokens 511..542)
max|Δlogit| all / completion, max|Δlogp| completion, mean|Δlogp|, argmax agreement (all):
golden vs HF-bf16 0.529 / 0.293, 0.064, 0.022, 0.985 | Triton vs HF-bf16 0.836 / 0.257, 0.066, 0.025, 0.989 |
golden vs HF-fp32 0.444 / 0.173, 0.041, 0.0125, 0.9963 | Triton vs HF-fp32 0.474 / 0.226, 0.046, 0.016, 0.9963 |
HF-bf16 vs HF-fp32 0.550 / 0.263, 0.046, 0.019, 0.9853 | golden vs Triton 0.415 / –, 0.048, 0.013, 0.9963.
Per-layer residual vs fp32 (mean/max |Δ|: golden | Triton | HF-bf16 | rms): emb 8.7e-4/0.062 (all) | 1.00;
L0 2.5e-3/0.29 | 2.5e-3/0.29 | 3.5e-3/0.33 | 1.12; L5 7.8e-3/0.75 | same | 1.4e-2/0.75 | 1.62; L10 1.16e-2/4.2 |
same | 2.0e-2/4.2 | 2.03; L20 1.54e-2/1.6 | 1.54e-2/1.6 | 2.4e-2/1.8 | 2.83; L30 1.64e-2/2.5 | 1.62e-2/2.0 |
2.2e-2/2.9 | 2.81; L40 1.87e-2/4.4 | 1.86e-2/4.4 | 2.3e-2/4.4 | 3.33; L45 2.59e-2/10.5 | 2.57e-2/10.5 |
3.3e-2/10.5 | 4.43; L50 4.02e-2/6.8 | 3.99e-2/5.4 | 5.0e-2/5.8 | 7.29; final norm 1.14e-2/0.65 | 1.16e-2/0.59 |
1.47e-2/0.80 | 1.01. Max diffs are single outlier channels (bf16 ulp 2–8); the mean grows ≈ linearly with depth.

## Benchmark / memory
Weights 51.9 GiB (11 s load); peaks 52.0–54.3 GiB; HF bf16 load 13–15 s, forward 0.48 s; HF fp32 on CPU load
7.6 s, forward 27.5 s (64 threads); golden train 4.7 s @544 / 35.8 s @2132, decode 0.9 s/step; Triton train 1.2 s,
decode 0.13 s/step.

## Integration requests
None required. Optional ws1.json row `muse_glimmer_full_model_hf_anchor`:
`{"logp_max_abs": 0.1, "logp_mean_abs": 0.03, "argmax_agreement_min": 0.97}` (observed 0.066 / 0.025 / 0.985).

## Open issues
Sibling attention kernels hard-wired to 32/2/128 heads (tiny tests golden-only); the full-model backward does not
fit on one GPU (tiny only); the golden decode is slow at long context (Python block loop).

## RFC feedback
- `full_model_chain`: define the gate as byte-equal selected-token logp across train / prefill / chunked
  (prompt-boundary, 64-aligned, ragged, W−1/W/W+1) / decode on ONE op set, plus a tolerance anchor vs HF-bf16 AND
  vs fp32 (HF-bf16 alone cannot distinguish a correct strict path from HF's own rounding); require node-named
  first-drift reports; state that lm_head must accept or mask targetless rows.
- `hybrid_attention_regression`: run T > W with chunk boundaries W−1/W/W+1 and assert real eviction on local
  layers; O5 must spell out the m = −inf guard (α = 1, p = 0) and the exact-zero contribution of masked/evicted
  (possibly NaN) keys.

Time spent ≈ 6 h.

import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/reference/REPORT_full_model_chain.md")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new)


rep('''## Follow-up (wave A): scoring, re-run, profiling
''', '''## Follow-up (wave A): scoring, re-run, profiling

*Wave A record. Every number below was re-measured in wave B (above); the wave A runs predate
the O7 fix (c88607f), so their triton logprob bits are stale. Claims the wave B review found
overstated are corrected in place and marked "(wave B correction)".*
''')

rep('''- Decode is the slow form: triton 0.11–0.17 s/token (e.g. 29.3 s for the 178 long-window steps at context
  2261–2438), golden 0.6–1.0 s/token for short contexts and 2.3 s/token at 2.3k context (415.6 s for 178 steps).''',
    '''- Decode is the slow form: triton 0.11–0.17 s/token (e.g. 29.3 s for the 178 long-window steps at context
  2261–2438), golden 0.6–1.0 s/token for short contexts and 2.3 s/token at 2.3k context (415.6 s for 178 steps).
  (wave B correction: those totals include the one-shot prompt prefill of 2261 tokens and there are
  177 decode steps; golden ≈ (415.6 − 43.9) / 177 = 2.10 s per decode step. Wave B times the prefill and
  every step separately, see above.)''')

rep('''So the strict path is GEMM-bound and its GEMM is ≈ 2.2× slower than cuBLAS. Strict attention, norms and
qknorm/rope are each faster than HF's eager/sdpa equivalents. Per layer at T=4096:''',
    '''So the strict path is GEMM-bound and its GEMM is ≈ 2.2× slower than cuBLAS. (wave B correction: strict
LOCAL attention, norms and qknorm/rope are faster than HF's equivalents; strict GLOBAL attention is ≈ 2×
slower per layer than HF's sdpa kernel, see below.) Per layer at T=4096:''')

rep('''(index kernels), against 23.8 + 8.0 ms for the attention kernels themselves. That is reference-harness
overhead, not kernel cost; a vLLM-style single scatter per layer removes it.''',
    '''(index kernels), against 23.8 + 8.0 ms for the attention kernels themselves. That is reference-harness
overhead, not kernel cost; a vLLM-style single scatter per layer is expected to remove it (not measured).''')

rep('''- Performance contract: on A100 the strict forward is GEMM-bound (94–95 % of GPU time in `det_gemm_tree_kernel`
  at T = 512–4096). The 8-chunk tree GEMM reaches ≈ 120 TFLOP/s against ≈ 260 for cuBLAS, which accounts for
  essentially all of the 1.55× prefill gap to HF bf16. The RFC's performance item should target the GEMM, not
  attention or norms.''',
    '''- Performance contract: on A100 the strict forward is GEMM-bound (93.5–95.9 % of GPU time in
  `det_gemm_tree_kernel` at T = 512–4096, train and paged; wave B numbers). The 8-chunk tree GEMM reaches
  ≈ 120 TFLOP/s against ≈ 265 for cuBLAS. The GEMM gap is LARGER than the whole prefill gap to HF bf16
  (T=4096, wave B: 978 ms of GEMM vs 674 ms total), partly offset by faster non-GEMM ops (110 vs 414 ms).
  The RFC's performance item should target the GEMM first; strict global attention (≈ 2× HF's sdpa per
  layer) is second.''')

rep('''- Commit provenance: `xrt_reference_golden.json` says 86d1087 in its `commit` field. The run actually started
  from 69b4409: the sha was taken when the file was written, and that bug is fixed in f7d3723. The strict
  code (strict_model, chain_harness core, golden_ops, paged_kv, weights, backend, every ops/triton file) is
  identical from 69b4409 to HEAD. `reference_profile.json` → `notes.measured_commits` lists the commit of
  every run.''',
    '''- Commit provenance (wave A): `xrt_reference_golden.json` said 86d1087 in its `commit` field while the run
  started from 69b4409 (the sha was taken when the file was written; fixed in f7d3723), and the hand-written
  `notes.measured_commits` disagreed with the HF file's own field (422c33b, dirty). Superseded in wave B:
  every file was regenerated and `measured_commits` is derived from the files.''')
p.write_text(s)
print("ok")

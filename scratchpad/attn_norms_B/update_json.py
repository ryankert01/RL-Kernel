"""Wave B update of reports/muse_glimmer_poc/data/kernels_attention.json (run from the worktree)."""
import json
import sys

ROOT = sys.argv[1]
J = f"{ROOT}/reports/muse_glimmer_poc/data/kernels_attention.json"
RAW = "reports/muse_glimmer_poc/data/kernels_attention_raw"
RAWB = f"{ROOT}/{RAW}/waveB"
tests_summary = json.loads(sys.argv[2])  # {"owned": "...", "full_suite": "...", ...}
commits = json.loads(sys.argv[3])  # {"fail_closed": sha, ...}

d = json.load(open(J))


def load(name):
    return json.load(open(f"{RAWB}/{name}"))


LOCAL_FWD = (
    "local layer: torch SDPA has no sliding-window kernel; the baseline is SDPA with a dense "
    "bool mask (math/efficient path, O(T^2) memory), not a flash-class kernel"
)
LOCAL_BWD = (
    "local layer: the SDPA backward baseline is the FULL-causal flash backward (no window), "
    "i.e. different work from the windowed strict backward"
)

# ---------------------------------------------------------------- wave A rows: mark local n/c
b = d["benchmarks"]
for sec in ("prefill_forward", "decode_forward", "tp_shard_forward"):
    for r in b[sec]:
        if r.get("window") is not None and "sdpa_ms" in r:
            r["ratio"] = None
            r["baseline_not_comparable"] = LOCAL_FWD
for sec in ("backward", "tp_shard_backward"):
    for r in b[sec]:
        if r.get("window") is not None and "sdpa_causal_bwd_ms" in r:
            r["ratio"] = None
            r["baseline_not_comparable"] = LOCAL_BWD
for r in b["backward_before_after"]:
    if r.get("window") is not None:
        r["v1_ratio_to_sdpa"] = None
        r["v2_ratio_to_sdpa"] = None
        r["baseline_not_comparable"] = LOCAL_BWD
b["description"] += (
    " WAVE B: every local-layer SDPA row now has ratio=null + baseline_not_comparable (dense-mask "
    "forward, full-causal backward); sdpa_ms values are kept as measured. Wave A rows are "
    "single-series means (no per-round samples, except backward_before_after); the wave B "
    "forward series (benchmarks_wave_b) use alternating rounds with raw samples and add the "
    "vLLM FA2 windowed baseline."
)

# ---------------------------------------------------------------- wave B benchmarks
base_fwd = load("bench_base_fwd.json")
new_fwd = load("bench_new_fwd.json")
run1 = load("bench_waveB.json")
run2 = load("bench_waveB_bwd2.json")
dec = load("decode_overhead_new.json")
for x in (base_fwd, new_fwd, run1, run2):
    x.pop("argv", None)
d["benchmarks_wave_b"] = {
    "description": (
        "CUDA-event ms, one leased A100 of pool 3,7. Forward rows: strict / SDPA / FA2 timed in "
        "5 alternating rounds x 30 iterations after 3 warm-ups; *_ms = median, *_ms_rounds = raw "
        "per-round samples, *_ms_range = [min, max]; *_max_abs_diff_y = baseline output vs strict "
        "y. FA2 = vllm.vllm_flash_attn.flash_attn_varlen_func (causal, window_size [2047, 0] on "
        "local layers; decode on the same paged cache). Local SDPA rows: ratio null (not "
        "comparable). Backward / TP rows: single series of 15 (backward) or 30 iterations, means; "
        "the backward wrapper syncs inside each call (prepare_sequence_meta), so those rows vary "
        "up to ~2x run to run (compare run_1 and run_2)."
    ),
    "paired_forward_before_after": {
        "note": (
            "same driver, same GPU, run one after the other: f91e8b6 wrappers (base) then this "
            "branch; prefill/decode forward only"
        ),
        "base_f91e8b6": {k: base_fwd[k] for k in ("prefill_forward", "decode_forward", "env",
                                                   "modules", "iters", "rounds")},
        "this_branch": {k: new_fwd[k] for k in ("prefill_forward", "decode_forward", "env",
                                                 "modules", "iters", "rounds")},
        "commit_measured": commits["bench"],
        "files": [f"{RAW}/waveB/bench_base_fwd.json", f"{RAW}/waveB/bench_new_fwd.json"],
    },
    "run_1_all_sections": {"commit_measured": commits["bench"], "data": run1,
                           "file": f"{RAW}/waveB/bench_waveB.json"},
    "run_2_backward_tp": {"commit_measured": commits["bench"], "data": run2,
                          "file": f"{RAW}/waveB/bench_waveB_bwd2.json"},
    "paged_decode_wrapper_overhead": {
        "commit_measured": commits["decode_overhead"],
        "records": dec["records"],
        "before_review_probe_e13ea68_full_wrapper_ms": {
            "B1_global": 0.321, "B1_local": 0.316, "B64_global": 0.323, "B64_local": 0.320,
            "source": "reviewer's probe_decode_overhead.py (wave A2 issue gated_attention_fwd:R6), "
                      "not re-run by this agent; the paired bench above is this agent's own "
                      "before/after",
        },
        "file": f"{RAW}/waveB/decode_overhead_new.json",
        "script": f"{RAW}/waveB/scripts/probe_decode_overhead_new.py",
    },
}

# ---------------------------------------------------------------- wave B summary
cmp = load("digest_cmp.json")
ir = load("ir_new.json")
nb, nn = load("norm_golden_mismatch_base.json"), load("norm_golden_mismatch_new.json")
before_tests = open(f"{RAWB}/before_tests.txt").read().splitlines()
d["wave_b"] = {
    "row": "attn_norms (gated attention fwd/bwd + norms), review wave A2 fixes",
    "base_commit": "f91e8b6 (feat/museglimmer-poc, merged first)",
    "commits": commits,
    "forward_bits_unchanged": {
        "method": (
            "gated_attention_bench digest (this branch's driver) run on a git-archive export of "
            "f91e8b6 and on this branch; sha256 of y, o, lse per case"
        ),
        "forward_cases": cmp["forward"]["cases"],
        "forward_identical": cmp["forward"]["identical"],
        "backward_cases": cmp["backward"]["cases"],
        "backward_identical": cmp["backward"]["identical"],
        "different": cmp["forward"]["different"] + cmp["backward"]["different"],
        "new_cases": (
            "31 forward cases added: TP shards 16/1, 8/1, 4/1 varlen TOKEN+HEAD and paged decode; "
            "paged chunked prefill (queries = sequence tail; TOKEN/HEAD/auto; interleaved page "
            "ids); local decode with evicted (-1, NaN-filled) pages"
        ),
        "files": [f"{RAW}/waveB/digest_base.json", f"{RAW}/waveB/digest_final.json",
                  f"{RAW}/waveB/digest_cmp.json", f"{RAW}/waveB/digest_new1.json"],
        "commit_measured": commits["final_digest"],
    },
    "fail_closed": {
        "kernel_checks": [
            "cu_seqlens_q (and cu_seqlens_k on varlen): [0]==0, non-decreasing, [-1]==Tq (Tk)",
            "grid coverage: len_q <= rows of grid axis 0 (under-reported max_seqlen_q)",
            "varlen: 0 <= q_pos < len_k per row",
            "paged: len_q <= seq_lens <= max_pages*16",
            "paged: block-table entry in [0, num_pages) for every visible key (never "
            "dereferenced otherwise)",
        ],
        "signal": "status word (int32, 1 = ok) + torch._assert_async(status, METADATA_ASSERT_MSG)",
        "host_paths": {
            "varlen_seq_meta": "SequenceMeta identity/version check, no sync",
            "varlen_max_seqlen_q_None": "validate_varlen_packing: one D2H copy, ValueError",
            "paged_max_seqlen_q_None": "validate_paged_packing: one D2H copy, ValueError",
            "autograd_without_meta_grads_needed": (
                "prepare_sequence_meta in forward, reused by backward (2 D2H per layer instead "
                "of 1 + 2)"
            ),
        },
        "cost_no_sync_path": "one 4-byte torch.ones fill + one torch._assert_async launch per call",
        "outputs_nan_prefill": "not added: the coverage check makes unwritten rows impossible "
                               "without the assert firing",
    },
    "tests": tests_summary,
    "tests_fail_before_pass_after": {
        "method": (
            "new test files copied into a git-archive export of f91e8b6 and run there "
            "(PYTHONPATH = that tree)"
        ),
        "result_on_f91e8b6": before_tests[-1] if before_tests else None,
        "lines": before_tests,
        "file": f"{RAW}/waveB/before_tests.txt",
        "note": (
            "the 10 tests that pass on f91e8b6 are coverage additions (layout scan, CP4/CP8 "
            "plumbing, controls, embedding OOB pinning), not behaviour changes"
        ),
    },
    "compiled_arithmetic": {
        "summary": {
            "exp": "ex2.approx.f32(x * log2 e)",
            "log": "libdevice __nv_logf, FTZ",
            "division": "div.rn.ftz.f32",
            "l_update": "fma.rn.f32 (contracted)",
            "pv": "tt.dot accumulates into acc*alpha",
        },
        "commit_measured": commits["ir"],
        "per_variant_ptx_f32_ops": ir["attention"],
        "norms_ptx_f32_ops": ir["norms"],
        "file": f"{RAW}/waveB/ir_new.json",
    },
    "norms_golden_vs_strict_mismatch_fraction_T4096": {"f91e8b6_golden": nb, "ieee_golden": nn,
                                                        "commit_measured": commits["norms_mismatch"],
                                                        "file_base": f"{RAW}/waveB/norm_golden_mismatch_base.json",
                                                        "file_new": f"{RAW}/waveB/norm_golden_mismatch_new.json"},
    "finding_verdicts": json.loads(open(sys.argv[4]).read()),
}

# ---------------------------------------------------------------- O5 text corrections
t = d["o5_replacement_text"]
a1 = "o = div_rn(acc, l) (IEEE round-to-nearest fp32 division)"
assert a1 in t
t = t.replace(a1, "o = div_rn(acc, l) (round-to-nearest fp32 division; compiled div.rn.ftz.f32, "
                  "subnormals flushed)")
a2 = "the kernel masks them anyway so eviction timing cannot change bits."
assert a2 in t
t = t.replace(a2, a2 + " Every block-table entry under a key that some query of the call can "
              "see must be a live page id in [0, num_pages); anything else (including -1) fails "
              "closed. exp/log are Triton's fp32 tl.exp/tl.log as compiled by the pinned Triton "
              "(tl.exp = ex2.approx(x*log2 e)); the bitwise gates are kernel vs kernel, the torch "
              "golden is a tolerance reference.")
d["o5_replacement_text"] = t

# ---------------------------------------------------------------- headline / misc
old_cp = "against cp_attention at CP1, simulated CP2/4/8 and real CP2 on two GPUs"
assert old_cp in d["headline"][1], d["headline"][1]
d["headline"][1] = d["headline"][1].replace(
    old_cp,
    "against simulated CP2/4/8, real CP2 on two GPUs and (wave B) the real cp_attention backward "
    "at CP4/CP8 driven on one GPU with a replayed all_gather (the cp_attention CP1 comparison is "
    "a plumbing check: the same calls by construction)",
)
d["headline"] = [
    "Wave B: the strict attention forward fails closed on every bad-metadata case the review "
    "found (cu_seqlens offsets, under-reported max_seqlen_q, q_pos outside the keys, seq_lens "
    "outside [len_q, max_pages*16], -1 or out-of-range page ids under visible keys): host "
    "ValueError where the wrapper syncs anyway, otherwise in-kernel checks + a device-side "
    "assert (no new sync on the hot path); 26 new tests fail on f91e8b6 and pass here.",
    "Forward bits unchanged by wave B: 97/97 forward (y, o, lse) and 14/14 backward digests "
    "identical to f91e8b6, including 31 new cases (TP shards, paged chunked prefill, evicted "
    "pages).",
] + d["headline"][:4] + [
    "Cost (wave B paired runs, alternating rounds): prefill forward at T = 4096 is 1.27x SDPA "
    "and 1.40x vLLM FA2 on global layers, 1.18x FA2 on local layers (the local SDPA ratios of "
    "wave A are withdrawn: dense-mask baseline); paged decode is 1.16-2.4x faster than at "
    "f91e8b6 (positions now derived in the kernel); the v2 backward is 2.5x the SDPA causal "
    "backward at T = 4096 (global; no comparable local backward baseline in the envs).",
]
d["commit_note"] = (
    d.get("commit_note", "") + " | WAVE B: 'commit' (d0cc564) dates the wave A measurements; "
    "wave B measurements carry their own commit_measured fields; wave_b.commits lists the "
    "branch commits. Raw files: reports/muse_glimmer_poc/data/kernels_attention_raw/ (waveA = "
    "copies of the ephemeral scratchpad files named in 'commands', waveB = this wave's runs)."
)
d["raw_dir"] = RAW
d["commands"]["scratch_note"] = (
    "the $SCRATCH directory is ephemeral; copies of the referenced files are committed in "
    f"{RAW}/waveA (final_tests.log -> final_tests.log.txt, cp2.log -> cp2.log.txt)"
)
d["commands"]["wave_b"] = {
    "base_tree": "git archive --format=tar -o $S/base.tar f91e8b6 rl_engine; tar -xf $S/base.tar "
                 "-C $S/base_tree; cp <branch>/rl_engine/models/muse_glimmer/ops/"
                 "gated_attention_bench.py into it",
    "digest_base": "PYTHONPATH=$S/base_tree ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/vllm/bin/"
                   "python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench digest "
                   "--out digest_base.json",
    "digest_final": "PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/vllm/bin/python -m "
                    "rl_engine.models.muse_glimmer.ops.gated_attention_bench digest --out "
                    "digest_final.json; python scripts/cmp_digest.py digest_base.json "
                    "digest_final.json digest_cmp.json",
    "bench_paired": "same driver with PYTHONPATH=$S/base_tree then $WT: bench --iters 30 "
                    "--rounds 5 --sections prefill decode",
    "bench_full": "PYTHONPATH=$WT ... gated_attention_bench bench --iters 30 --rounds 5 (run 1); "
                  "--sections backward tp (run 2)",
    "decode_overhead": "python scripts/probe_decode_overhead_new.py decode_overhead_new.json",
    "ir": "python scripts/probe_ir_new.py ir_new.json",
    "norms_mismatch": "python scripts/probe_norm_golden_mismatch.py <out> (both trees)",
    "tests_owned": "PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/vllm/bin/python -m "
                   "pytest -q tests/models/muse_glimmer/test_gated_attention_fwd.py "
                   "tests/models/muse_glimmer/test_gated_attention_bwd.py "
                   "tests/models/muse_glimmer/test_norms.py",
    "tests_before": "same test files copied into $S/base_tree/tests/models/muse_glimmer; "
                    "PYTHONPATH=$S/base_tree pytest -k '<new tests>' -rA",
}
d["tests"]["final_single_gpu_files"]["log"] = f"{RAW}/waveA/final_tests.log.txt"
d["tests"]["final_cp_two_gpu"]["log"] = f"{RAW}/waveA/final_tests.log.txt"
d["tests"]["wave_b"] = tests_summary
d["not_run"] = d["not_run"] + [
    "wave B: test_norms.py::test_fused_rejects_two_gpus (needs two visible GPUs; this row's "
    "leases are one GPU at a time)",
    "wave B: distributed CP/TP tests were not re-run (one GPU at a time); the CP backward "
    "plumbing at CP4/CP8 is covered on one GPU by test_cp_backward_plumbing_cp4_cp8",
    "wave B: no windowed BACKWARD baseline (vLLM's FA2 build is forward-only)",
]
d["requests_to_integrator"] = d["requests_to_integrator"] + json.loads(sys.argv[5])
d["suggested_figures"] = [
    f for f in d["suggested_figures"] if not f["title"].startswith("Backward time before/after")
] + [
    {"title": "Backward time before/after and vs SDPA (global only; local SDPA is n/c)",
     "data": "benchmarks.backward_before_after[*] with window == null (series v1/v2/SDPA)"},
    {"title": "Prefill forward: strict vs SDPA vs FA2, global and local (median + range)",
     "data": "benchmarks_wave_b.paired_forward_before_after.this_branch.prefill_forward "
             "(strict_ms, sdpa_ms (global only), flash_ms; *_ms_range for error bars)"},
    {"title": "Paged decode before/after (positions in the kernel)",
     "data": "benchmarks_wave_b.paired_forward_before_after.{base_f91e8b6,this_branch}."
             "decode_forward[*].strict_ms by B and window"},
    {"title": "Forward digests vs f91e8b6 (all identical)",
     "data": "wave_b.forward_bits_unchanged (cases / identical)"},
]
if "rl_engine/models/muse_glimmer/ops/REPORT_norms.md" not in d["reports"]:
    d["reports"].append("rl_engine/models/muse_glimmer/ops/REPORT_norms.md")
json.dump(d, open(J, "w"), indent=1)
print("updated", J)

"""Assemble reports/muse_glimmer_poc/data/kernels_attention.json from this session's raw outputs."""
import json
import sys

SP = sys.argv[1]
OUT = sys.argv[2]
CODE_COMMIT = sys.argv[3]
WT = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1"


def load(name):
    with open(f"{SP}/{name}") as fh:
        return json.load(fh)


parity = load("parity.json")
cp2 = load("cp2_torchrun.json")
digest_old = load("digest_old.json")
digest_new = load("digest_new2.json")
digest_cmp = load("digest_cmp.json")
bwd_diff = load("bwd_diff.json")
bwd_ab = load("bwd_ab.json")
bench = load("bench_new2.json")
probe = load("probe_sigmoid.json")


def total(records, keys):
    return sum(int(r[k]) for r in records for k in keys if k in r)


# ------------------------------------------------------------------ parity matrices
tp_matrix = []
for r in parity["tp"]:
    sums = {}
    for rk in r["ranks"]:
        for k, v in rk.items():
            if k in ("rank", "q_heads", "kv_head") or k.startswith("numel"):
                continue
            sums[k] = sums.get(k, 0) + int(v)
    row = {"case": r["case"], "tp": r["tp"], "window": r["window"],
           "mismatches_summed_over_ranks": sums}
    for k in ("dk_tp_tree_vs_tp1", "dv_tp_tree_vs_tp1", "dk_replica_tree_vs_tp1",
              "dv_replica_tree_vs_tp1", "numel_dk"):
        if k in r:
            row[k] = r[k]
    tp_matrix.append(row)
tp_total = sum(sum(v for v in row["mismatches_summed_over_ranks"].values()) for row in tp_matrix)
tp_total += total(tp_matrix, ("dk_tp_tree_vs_tp1", "dv_tp_tree_vs_tp1", "dk_replica_tree_vs_tp1",
                              "dv_replica_tree_vs_tp1"))
cp_sim_total = total(parity["cp"], ("dq", "dk", "dv", "dgate"))
cp2_keys = [f"{a}_mismatch" for a in ("y", "o", "lse")] + [
    f"d{n}{s}" for n in ("q", "k", "v", "gate")
    for s in ("_mismatch", "_vs_one_shot_mismatch", "_cp1_vs_one_shot_mismatch")
]
cp2_total = total(cp2["records"], cp2_keys)
cp2_matrix = [
    {k: r[k] for k in ("layer", "T", "cp", "rank", *cp2_keys, "repeat_equal", "cp_forward_ms",
                       "cp_fwd_bwd_ms", "ag_kv_ms", "ag_positions_ms", "ag_kv_bytes_per_rank",
                       "dq_numel_local", "dk_numel_local") if k in r}
    for r in cp2["records"]
]

# ------------------------------------------------------------------ bits vs old
bwd_changes = []
for r in bwd_diff["records"]:
    if "old_error" in r:
        bwd_changes.append({"T": r["T"], "window": r["window"], "old_error": r["old_error"][:120]})
        continue
    rec = {"T": r["T"], "window": r["window"]}
    for k, v in r.items():
        if k.endswith(("_mismatch", "_numel", "_max_abs_diff", "_max_rel_diff", "_max_ulp")):
            rec[k] = v
    bwd_changes.append(rec)

fwd_same = sum(1 for v in digest_cmp["forward"].values() if isinstance(v, dict) and v and all(
    x is True for x in v.values()))

# ------------------------------------------------------------------ findings
findings = [
    {"id": "gated_attention_bwd:1", "severity": "blocking", "verdict": "confirmed_fixed",
     "evidence": "ops/gated_attention_autograd.py imports rl_engine.models.muse_glimmer.ops."
     "gated_attention and calls strict_gated_attention_fwd_varlen; no try/except fallback (an "
     "import failure now fails the import). trace(): forward_is_fallback False, forward_impl = "
     "ops.gated_attention.strict_gated_attention_fwd_varlen, forward trace nested under "
     "'forward'. test_autograd_forward_is_the_strict_kernel: GatedAttentionFn y == strict kernel "
     "y (torch.equal), lens [300, 64], global + local: PASS (143 passed in the final run).",
     "tests": ["test_autograd_forward_is_the_strict_kernel", "test_trace_is_stable"]},
    {"id": "gated_attention_bwd:2", "severity": "should_fix", "verdict": "confirmed_fixed",
     "evidence": "test_autograd_function_binds_strict_backward now takes o/lse from the strict "
     "forward (asserts the Function's y equals it, i.e. the same saved tensors) and keeps "
     "torch.equal for dq/dk/dv/dgate between the Function and the direct entry, plus a separate "
     "normalized accuracy check vs the FP64 reference; PASS global + local.",
     "tests": ["test_autograd_function_binds_strict_backward[global|local]"]},
    {"id": "gated_attention_bwd:3", "severity": "should_fix", "verdict": "confirmed_fixed",
     "evidence": "ops/attention_golden_bwd.golden_forward_fixed_order is a varlen adapter over "
     "ops.gated_attention.gated_attention_forward_fp32 (no second golden); "
     "test_golden_adapter_is_the_forward_golden asserts bitwise equality per sequence (packed, "
     "offset positions), PASS; every backward test derives o/lse from the strict forward kernel "
     "(strict_bwd helper).",
     "tests": ["test_golden_adapter_is_the_forward_golden[global|local]"]},
    {"id": "gated_attention_bwd:4", "severity": "should_fix", "verdict": "confirmed_fixed",
     "evidence": "preprocess calls the shared Triton helper sigmoid_rn = div_rn(1, 1 + exp(-g)) "
     "that the forward epilogue also calls; trace()['cast_points'] names it. Probe (one-key "
     "sequences, y = bf16(o*sg_fwd) vs do = bf16(o*sg_bwd) with dy := o, 16,777,216 values): "
     f"v2 {probe['new_preprocess_vs_forward_y']} mismatches, v1 "
     f"{probe['old_preprocess_vs_forward_y']}; the same check is "
     "test_backward_sigmoid_is_the_forward_sigmoid (PASS, would fail on v1). It changed dgate bits "
     "(e.g. 100/8,388,608 at T=2048) and, through do, dq bits.",
     "tests": ["test_backward_sigmoid_is_the_forward_sigmoid"]},
    {"id": "gated_attention_bwd:5", "severity": "should_fix", "verdict": "confirmed_fixed",
     "evidence": "The tautological TP loop is gone. Kernels and wrappers take num_q_heads / "
     "num_kv_heads ((16,1), (8,1), (4,1) shards); test_tp_head_shards_equal_tp1_slices runs real "
     "per-rank calls and compares dq / dgate / delta / dK-dV group partials to the TP1 slices "
     "and the rank nodes folded by runtime/tp_canonical_tree.combine_tree to TP1 dk/dv "
     f"(torch.equal, PASS); the parity driver counts {tp_total} mismatching bit patterns over "
     "TP2/4/8 x 3 shapes x 2 layer kinds.",
     "tests": ["test_tp_head_shards_equal_tp1_slices[global|local x 2|4|8]"]},
    {"id": "gated_attention_bwd:6", "severity": "nit", "verdict": "confirmed_fixed",
     "evidence": "The vacuous ws1 allclose after the normalized bound was removed; the module "
     "docstring states the normalized bound max|err|/max|ref| <= 1e-2 is the gate. The suggested "
     "'attention_gradient' profile row is not added (contracts/profiles not owned by this row; "
     "listed under requests_to_integrator).",
     "tests": ["test_backward_matches_reference[*]"]},
    {"id": "gated_attention_bwd:7", "severity": "nit", "verdict": "confirmed_fixed",
     "evidence": "Code side: the direct entry and the preprocess reject a non-bf16 dy with "
     "ValueError (test_rejects_non_bf16_dy PASS); the do / p / dS / dgate casts and the div_rn "
     "sigmoid are declared in the module docstring and trace()['cast_points']. Contract side: the "
     "exact O5 replacement text naming these casts is in requests_to_integrator (the contract "
     "file is not owned by this row, so it is not yet amended).",
     "tests": ["test_rejects_non_bf16_dy"]},
    {"id": "gated_attention_bwd:8", "severity": "nit", "verdict": "confirmed_fixed",
     "evidence": "prepare_sequence_meta raises unless cu_seqlens_k[-1] == Tk; _check_inputs "
     "requires k, v, cu_seqlens_q/k and q_pos on q's device. "
     "test_rejects_inconsistent_cu_seqlens_k_and_devices (cu_k[-1] too small, too large, CPU "
     "cu_seqlens_q) PASS.",
     "tests": ["test_rejects_inconsistent_cu_seqlens_k_and_devices"]},
    {"id": "gated_attention_bwd:9", "severity": "nit", "verdict": "confirmed_fixed",
     "evidence": "SequenceMeta (prepare_sequence_meta, bound to its tensors by identity + version "
     "counters, carries qpos0 on device and max_seqlen_q) is accepted by every backward entry and "
     "by gated_attention(..., seq_meta); test_seq_meta_removes_host_sync runs the direct backward "
     "and the autograd forward + backward under torch.cuda.set_sync_debug_mode('error') without "
     "error and checks the grads equal the validating path; a meta for other tensors raises. "
     "Caveat: torch's sync debug mode is a prototype and does not detect every sync; without a "
     "meta the backward still does its two validating D2H copies.",
     "tests": ["test_seq_meta_removes_host_sync[global|local]"]},
]

# ------------------------------------------------------------------ integrator texts
O5_TEXT = r"""### O5 `gated_attention`  (rows: global_gated_attention, local_gated_attention_train/rollout)
Inputs q̂ (bf16, from O4), K, V (bf16, from O4/O3), gate pre-activation g = attn_gate GEMM output
(bf16, H_Q×128 channels, from O3 on the input_layernorm output). Softmax scale s = 128^(-1/2) (fp32).
Mask: global layer: k_pos ≤ q_pos. Local layer: q_pos − W < k_pos ≤ q_pos (W=2048 keys incl. self).
Positions are absolute positions of the sequence (training, packed/chunked prefill, decode and CP
ranks use the same positions).
Head geometry: one call holds (H_Q, H_KV) ∈ {(32,2) TP1, (16,1) TP2, (8,1) TP4, (4,1) TP8}; rank r
of TP t holds Q heads [32r/t, 32(r+1)/t) and KV head floor(2r/t) (replicated on t/2 ranks when
t > 2); local q head h reads local KV head h // (H_Q/H_KV). Any other geometry is rejected. No
operation below mixes heads except the K/V-gradient fold, so every per-head output of a rank equals
the matching slice of the TP1 call bit for bit.
Forward (per query row, fp32):
  - KV is traversed in blocks of BLOCK_N=64 keys aligned at absolute key position 0, ascending.
    For a query at position p the first visited block is floor(max(0, p−W+1)/64) on local layers
    and 0 on global layers; the last is floor(p/64). Fully-masked blocks are a bitwise no-op, so
    visiting extra blocks (because a query tile spans several positions) never changes a row's bits.
  - scores = tl.dot(q̂_tile, K_blockᵀ) fp32 (bf16×bf16, K=128, no TF32); scaled by s in fp32;
    masked to −inf.
  - m_new = max(m, rowmax(scores)); m_safe = 0 if m_new = −inf else m_new; α = exp(m − m_safe);
    p = exp(scores − m_safe); l = l·α + rowsum(p); acc = acc·α + tl.dot(p→bf16, V_block) (K=64);
    m = m_new. (p→bf16 before the PV dot is a declared cast.)
  - o = div_rn(acc, l) (IEEE round-to-nearest fp32 division) → bf16. σ(g) = div_rn(1, 1 + exp(−g))
    in fp32 (sigmoid_rn; the backward uses the same function). y = (o_bf16→fp32)·σ(g→fp32) → bf16,
    elementwise over the H_Q×128 channels. lse = m + ln(l) fp32 per (token, q head), l accumulated
    from the fp32 p.
  - Tiles: TOKEN tile = 64 consecutive queries of one q head (prefill/training); HEAD tile = 16 rows
    holding the q heads of one GQA group at one query position (decode / chunks of ≤16 queries; at
    TP4/TP8 the rows beyond the group of 8/4 are masked and never stored). Both tiles use the same
    num_warps (4): rowsum(p) is reduced along the warp split of the key axis, so every tile variant
    must share the warp split; the compiled MMA layout (warpsPerCTA) is checked by the tests.
  - Rollout: K/V come from a paged cache (page = 16 tokens; BLOCK_N=64 = 4 pages via a block
    table); arithmetic unchanged. Sliding-window eviction: pages whose every key position ≤ p−W may
    be freed; the kernel masks them anyway so eviction timing cannot change bits.
Backward (training only; FlashAttention-2 recompute, no atomics, no split-K, no autotune):
  - Preprocess per (token, q head), fp32: σ = sigmoid_rn(g); dgate = dy·o·σ·(1−σ) → bf16;
    dO = dy·σ → bf16 (declared cast); D = Σ_d dO_bf16·o_bf16 (tl.sum over the 128 lanes). dy must be
    bf16.
  - dQ, per 64-query tile (aligned at the call's first query) and q head: key blocks of 64 ascending
    over the forward's block range; s = tl.dot(Q, Kᵀ)·s masked to −inf; p = exp(s − lse) (p = 0 when
    lse = −inf); dP = tl.dot(dO, Vᵀ); dS = p·(dP − D); dQ += tl.dot(dS→bf16, K) fp32; dQ·s → bf16.
  - dK/dV, query-chunk rule (replaces "accumulated over query blocks in ascending order", which is
    not CP-invariant): the queries of a sequence of L_k keys are split into 8 canonical contiguous
    chunks on absolute positions, chunk c = [floor(c·L_k/8), floor((c+1)·L_k/8)) (empty chunks are
    exact zero leaves), each cut into 64-query tiles aligned at the chunk start (the last tile ends
    at the chunk end). Leaf (c, g) for canonical Q-head group g (4 consecutive q heads = one TP8
    shard), per 64-key block: for the 4 heads of g ascending, for the tiles of chunk c ascending:
    sᵀ = tl.dot(K_blk, Q_tileᵀ)·s masked; pᵀ = exp(sᵀ − lse); dV += tl.dot(pᵀ→bf16, dO_tile);
    dPᵀ = tl.dot(V_blk, dO_tileᵀ); dSᵀ = pᵀ·(dPᵀ − D); dK += tl.dot(dSᵀ→bf16, Q_tile);
    leaf dK = dK·s, leaf dV = dV (fp32). Group partial P_g = ((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7))
    (O3 tree, fp32). dK/dV of KV head h = (P_g0+P_g1)+(P_g2+P_g3) over the 4 groups that read h
    (O3 tree, fp32) → bf16 once.
  - Distribution (same expression at every degree, including the single-call kernel): a CP rank i
    of c (allgather layout, T divisible by c) owns chunks [8i/c, 8(i+1)/c); its call against the
    gathered K/V gives, per group, the sub-tree root of its chunks (the other chunks are exact zero
    leaves); rank roots are exchanged by a rank-ordered all_gather and folded in the same tree. A TP
    rank r of t folds its groups [8r/t, 8(r+1)/t) into its node of the per-KV-head tree (TP1/TP2:
    the full 4-leaf tree; TP4: a 2-leaf node; TP8: the leaf); the t/2 replicas of a KV head
    (TP4/TP8) all_gather their fp32 nodes in rank order and fold them with the same tree. No
    NCCL all_reduce / reduce_scatter on dK/dV. Query positions inside a backward call are
    contiguous and ascending per sequence.
Sequence-split invariance: a token's out bits are identical whether it is processed in one
prefill, in chunked prefills of any split, or as a single decode step, given identical K/V bits."""

SIGNATURE_TEXT = r"""Per-rank head calls (replace the KV-replication / head all-gather workaround). Rank r of TP t:
H_Q = 32 // t, H_KV = 2 // t if t <= 2 else 1, Q heads [r*H_Q, (r+1)*H_Q), KV head kvh = (2*r) // t,
gate / y / o columns [r*H_Q*128, (r+1)*H_Q*128). Inputs are the rank's own slices (contiguous).

from rl_engine.models.muse_glimmer.ops.gated_attention import (
    strict_gated_attention_fwd_varlen, strict_gated_attention_fwd_paged)
from rl_engine.models.muse_glimmer.triton.gated_attention_bwd import (
    gated_attention_bwd_preprocess, strict_gated_attention_bwd_partials, kv_partials_subtree,
    strict_gated_attention_bwd, prepare_sequence_meta)
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.ops.gated_attention_autograd import gated_attention

# forward, training / prefill (q_r (T,H_Q,128), k_r/v_r (Tk,H_KV,128), gate_r (T,H_Q*128) bf16)
y, o, lse = strict_gated_attention_fwd_varlen(
    q_r, k_r, v_r, gate_r, cu_seqlens_q, cu_seqlens_k, q_pos, *, window,   # None | 2048
    max_seqlen_q=None, tile_mode=None, num_q_heads=H_Q, num_kv_heads=H_KV)
#   -> y (T,H_Q*128) bf16, o (T,H_Q*128) bf16, lse (T,H_Q) fp32  == TP1 slices bitwise
# forward, vLLM paged rollout (k_cache_r/v_cache_r (pages,16,H_KV,128): the rank's KV head only)
y, o, lse = strict_gated_attention_fwd_paged(
    q_r, gate_r, k_cache_r, v_cache_r, block_table, seq_lens, cu_seqlens_q, *, window,
    max_seqlen_q=None, tile_mode=None, num_q_heads=H_Q, num_kv_heads=H_KV)

# backward, TP1 / TP2 (the rank owns whole KV heads): fused entry or the autograd op
dq, dk, dv, dgate = strict_gated_attention_bwd(
    q_r, k_r, v_r, gate_r, o, lse, dy_r, cu_seqlens_q, cu_seqlens_k, q_pos, *, window,
    num_q_heads=H_Q, num_kv_heads=H_KV, seq_meta=None)
y = gated_attention(q_r, k_r, v_r, gate_r, cu_seqlens_q, cu_seqlens_k, q_pos, window, seq_meta)
#   (head counts taken from q_r.shape[1] / k_r.shape[1]; raises for TP4/TP8 shards)

# backward, TP4 / TP8 (KV head replicated on t/2 ranks): partials + replica tree
dgate, dO, delta = gated_attention_bwd_preprocess(dy_r, o, gate_r, num_q_heads=H_Q)
dq, dk_part, dv_part = strict_gated_attention_bwd_partials(
    q_r, k_r, v_r, dO, lse, delta, cu_seqlens_q, cu_seqlens_k, q_pos, *, window,
    num_q_heads=H_Q, num_kv_heads=H_KV, seq_meta=None)          # (8/t, Tk, 1, 128) fp32
node_k, node_v = kv_partials_subtree(dk_part), kv_partials_subtree(dv_part)   # (Tk, 1, 128) fp32
# all_gather node_k / node_v (fp32, rank order) over the t/2 ranks with the same kvh, then
dk = combine_tree(nodes_k).to(torch.bfloat16)                   # identical on every replica
dv = combine_tree(nodes_v).to(torch.bfloat16)                   # == TP1 dk[:, kvh] bitwise

# optional, once per step (removes every host sync of the op):
meta = prepare_sequence_meta(cu_seqlens_q, cu_seqlens_k, q_pos, num_k_tokens=Tk)
Measured (one GPU, per-rank calls): 0 mismatching bit patterns vs TP1 for y/o/lse/dq/dgate/partials
and for the replica-tree dk/dv at TP2/4/8. Not covered: how the binding turns the replicated dk/dv
into k/v-projection weight and input gradients without double counting across replicas."""

requests = [
    "Contract O5: replace the whole O5 section of docs/contracts/muse-glimmer-30b-strict-contract.md "
    "with the text in `o5_replacement_text` (adds head geometry, the -inf guard, div_rn division and "
    "sigmoid, tile/warp rule, backward casts, the query-chunk dK/dV rule and the per-KV-head / "
    "replica tree; removes 'dK/dV accumulated over query blocks in ascending order' and '8 partial "
    "dK/dV per KV head').",
    "Bindings (vLLM + Megatron, next wave): switch from KV replication / head all-gather to "
    "per-rank head calls with the signatures in `tp_rank_call_signature`; TP4/TP8 must fold the "
    "fp32 dK/dV nodes of the t/2 KV-head replicas with combine_tree in rank order (no NCCL "
    "all_reduce) before rounding.",
    "runtime/kv_replication.py docstring refers to the v1 (8, Tk, 2, 128) partial layout and to "
    "the sibling kernel pinning 32/2 heads; both are outdated (layout is (H_Q/4, Tk, H_KV, 128); "
    "the wrappers take head counts) - owner of that file should update it and can drop the "
    "MUSEGLIMMER_STRICT_ATTN_FWD gate.",
    "Optional profile row for gradient accuracy (finding bwd:6): \"attention_gradient\": "
    "{\"bfloat16\": {\"mode\": \"normalized_tolerance\", \"max_err_over_max_ref\": 1.0e-2}} in "
    "rl_engine/contracts/profiles/precision/ws1.json (tests already apply this bound).",
    "Run on 4+ GPUs (not possible in pool 4-5): tests/models/muse_glimmer/distributed CP4 and "
    "TP2xCP2 cases, and check_cp_parity.py --backward at nproc 4 and 8.",
]

doc = {
    "row": "kernels_attention (global_gated_attention, local_gated_attention_train/rollout, "
           "cp_global_attention, cp_local_attention; gated attention fwd/bwd + TP head shards)",
    "status": "complete",
    "commit": CODE_COMMIT,
    "commit_note": "kernels, wrappers, tests and parity driver as measured; the JSON itself is "
                   "added in the following commit on branch worktree-wf_5e0d8071-7a1-1",
    "base_commit": "c88607f (feat/museglimmer-poc at merge time; old kernels = v1 there)",
    "env": {
        "gpu": "NVIDIA A100-SXM4-80GB (sm80)",
        "torch": bench["env"]["torch"],
        "triton": bench["env"]["triton"],
        "cuda": bench["env"]["cuda"],
        "python_env": "~/envs/vllm (vLLM 0.31.0 env)",
        "gpu_pool": "4-5 via ~/bin/gpu-lease (-n 1; -n 2 --contiguous for CP2)",
        "backward_kernel_id": bench["modules"]["backward_kernel_id"],
    },
    "commands": {
        "worktree": WT,
        "scratch": SP,
        "old_tree": "git archive c88607f rl_engine | tar -x -C $SCRATCH/old_tree (v1 kernels)",
        "tests_single_gpu": "PYTHONPATH=$WT ~/bin/gpu-lease -n 2 --contiguous --pool 4-5 -- "
        "~/envs/vllm/bin/python -m pytest -q tests/models/muse_glimmer/test_gated_attention_fwd.py "
        "tests/models/muse_glimmer/test_gated_attention_bwd.py",
        "tests_cp2": "PYTHONPATH=$WT ~/bin/gpu-lease -n 2 --contiguous --pool 4-5 -- "
        "~/envs/vllm/bin/python -m pytest -q -rs tests/models/muse_glimmer/distributed/"
        "test_cp_global_attention.py tests/models/muse_glimmer/distributed/"
        "test_cp_local_attention.py tests/models/muse_glimmer/distributed/test_cp_tp_topology.py",
        "parity": "PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 4-5 -- ~/envs/vllm/bin/python -m "
        "rl_engine.models.muse_glimmer.ops.gated_attention_bench parity --out $SCRATCH/parity.json",
        "cp2_torchrun": "PYTHONPATH=$WT ~/bin/gpu-lease -n 2 --contiguous --pool 4-5 -- "
        "~/envs/vllm/bin/python -m torch.distributed.run --nproc_per_node=2 "
        "rl_engine/models/muse_glimmer/runtime/check_cp_parity.py --backward --lengths 2048 2100 "
        "4096 8192 --json $SCRATCH/cp2_torchrun.json",
        "digest_new": "PYTHONPATH=$WT ... gated_attention_bench digest --out $SCRATCH/digest_new2.json",
        "digest_old": "PYTHONPATH=$SCRATCH/old_tree ~/bin/gpu-lease -n 1 --pool 4-5 -- "
        "~/envs/vllm/bin/python $SCRATCH/drv/ga_bench_driver.py digest --out "
        "$SCRATCH/digest_old.json (driver copied out of the worktree, old modules on PYTHONPATH)",
        "bwd_diff": "PYTHONPATH=$WT ... gated_attention_bench bwd-diff --old-module "
        "$SCRATCH/old_tree/rl_engine/models/muse_glimmer/triton/gated_attention_bwd.py --out "
        "$SCRATCH/bwd_diff.json",
        "bwd_ab": "PYTHONPATH=$WT ... gated_attention_bench bwd-ab --old-module <same> --iters 10 "
        "--rounds 7 --out $SCRATCH/bwd_ab.json",
        "bench": "PYTHONPATH=$WT ... gated_attention_bench bench --iters 30 --out "
        "$SCRATCH/bench_new2.json",
        "sigmoid_probe": "PYTHONPATH=$WT ... ~/envs/vllm/bin/python $SCRATCH/probe_sigmoid.py "
        "<old bwd file> $SCRATCH/probe_sigmoid.json",
    },
    "headline": [
        "TP head shards are bitwise: at TP2/4/8 every rank's call on its 16/1, 8/1 or 4/1 head "
        "shard equals the TP1 slice for y, o, lse, dq, dgate and the dK/dV group partials, and the "
        "KV-head replicas' fp32 nodes folded in the canonical tree equal TP1 dk/dv "
        f"({tp_total} mismatching bit patterns over 3 shapes x 2 layer kinds, plus HEAD-tile "
        "paged decode).",
        "The one-shot backward kernel now equals the CP path bit for bit: 0 mismatches in "
        "dq/dk/dv/dgate against cp_attention at CP1, simulated CP2/4/8 and real CP2 on two GPUs "
        "at T = 2048/2100/4096/8192, global and local (the v1 kernel differed from its own CP1 "
        "path in 1557/5540/20440 dK elements at T = 2048/4096/8192).",
        f"Forward bits are unchanged ({fwd_same}/66 digests identical to the v1 forward); the "
        "backward changed dq and dgate (the gate sigmoid is now the forward's div_rn sigmoid) and "
        "dk/dv (query-chunk rule); the largest change is 5.1e-3 of max|dk| (0.125 absolute), "
        "about the size by which the v1 one-shot dk differed from its own CP1 path (4.7e-3, "
        "0.125 absolute).",
        "All 9 review findings on the backward are confirmed fixed with tests; the final run "
        "passed 143 single-GPU tests and 18 two-GPU CP tests (5 skipped: they need 4 GPUs).",
        "Cost: the v2 backward is 5-14 % slower than v1 at T >= 1024 (2.52x / 2.08x the SDPA "
        "causal backward at T = 4096 global / local) with 2.5x the transient memory; prefill "
        "forward is 1.31x SDPA at T = 4096.",
    ],
    "suggested_figures": [
        {"title": "TP head-shard parity matrix (mismatches per output x TP degree x shape)",
         "data": "parity.tp_matrix[*].mismatches_summed_over_ranks + *_tree_vs_tp1"},
        {"title": "One-shot vs CP backward: v1 defect vs v2 (dK mismatches by T)",
         "data": "bits_vs_old.backward_diff[*]['defect_old_oneshot_vs_old_cp1/dk_mismatch'] vs "
                 "parity.cp_simulated[*].dk (all 0)"},
        {"title": "Which backward bits changed vs v1 (dq/dk/dv/dgate, split by cause)",
         "data": "bits_vs_old.backward_diff (full/*, preprocess/*, core_same_do/*)"},
        {"title": "Backward time before/after and vs SDPA",
         "data": "benchmarks.backward_before_after[*] (x=T, series v1/v2/SDPA, per window)"},
        {"title": "Per-rank attention time vs TP degree",
         "data": "benchmarks.tp_shard_forward / tp_shard_backward"},
    ],
    "parity": {
        "comparison": "bit-pattern mismatch counts (float tensors viewed as int16/int32)",
        "tp_matrix": tp_matrix,
        "tp_raw_per_rank": parity["tp"],
        "tp_total_mismatches": tp_total,
        "cp_simulated": parity["cp"],
        "cp_simulated_total_mismatches": cp_sim_total,
        "cp2_two_gpu_torchrun": cp2_matrix,
        "cp2_two_gpu_total_mismatches": cp2_total,
        "controls_association": parity["controls"],
    },
    "bits_vs_old": {
        "old": "c88607f forward + backward v1",
        "forward_digests_identical": f"{fwd_same}/66",
        "forward_digest_cases": digest_cmp["forward"],
        "backward_digest_cases": digest_cmp["backward"],
        "backward_diff": bwd_changes,
        "sigmoid_probe": probe,
        "summary": "forward unchanged (y, o, lse). backward: dgate changed only via the "
                   "sigmoid; dq only via do (the dQ kernel is bit-identical given the same do); "
                   "dk/dv via the query-chunk rule and do. Size over T = 2..8192: max |diff| "
                   "dq 0.0039, dk 0.125, dv 0.0625, dgate 0.0039 (<= 5.1e-3 of max |old|); "
                   "dgate and do differ by at most 1 bf16 ulp per element, dq/dk/dv also flip "
                   "near-zero (cancellation) elements, max ulp distance ~3e4 there. v1 could not "
                   "compile T = 1 (CompilationError); v2 handles it.",
    },
    "benchmarks": {
        "description": "CUDA-event ms on one leased A100 of pool 4-5. bench: 30 iterations "
                       "(backward 15) after 3 warm-ups; SDPA = F.scaled_dot_product_attention "
                       "(GQA) + the same gate for the forward, causal GQA backward without window "
                       "or gate for the backward; the local SDPA forward baseline is a dense bool "
                       "mask (no windowed flash kernel). backward_before_after: v1 and v2 timed "
                       "alternately in one process, 7 rounds x 10 iterations, medians.",
        "prefill_forward": bench["prefill_forward"],
        "decode_forward": bench["decode_forward"],
        "backward": bench["backward"],
        "backward_before_after": bwd_ab["records"],
        "tp_shard_forward": bench["tp_shard_forward"],
        "tp_shard_backward": bench["tp_shard_backward"],
        "tp_shard_backward_note": "TP4/TP8 rows time the rank's local work (preprocess + kernels "
                                  "+ its tree node); the cross-rank fold is not timed",
    },
    "findings": findings,
    "tests": {
        "final_single_gpu_files": {
            "log": f"{SP}/final_tests.log",
            "summary_line": "143 passed, 1 warning in 54.78s",
        },
        "final_cp_two_gpu": {
            "log": f"{SP}/final_tests.log",
            "summary_line": "18 passed, 5 skipped in 32.21s (skipped: CP4 x4, TP2xCP2 need 4 GPUs)",
        },
        "earlier_run_after_fold_change_one_gpu": "157 passed, 9 skipped (fwd + bwd + "
                                                 "distributed files on one GPU)",
    },
    "not_run": [
        "CP4 and CP8 (pytest and torchrun) and the TP2xCP2 4-GPU test: the row's pool has 2 GPUs",
        "TP2/4/8 as multi-process NCCL jobs: proven with per-rank calls on one GPU only",
        "30B real-weights attention through the vLLM / Megatron bindings with per-rank heads "
        "(next wave)",
    ],
    "o5_replacement_text": O5_TEXT,
    "tp_rank_call_signature": SIGNATURE_TEXT,
    "requests_to_integrator": requests,
    "reports": [
        "rl_engine/models/muse_glimmer/ops/REPORT_gated_attention_bwd.md",
        "rl_engine/models/muse_glimmer/ops/REPORT_gated_attention_fwd.md",
        "rl_engine/models/muse_glimmer/runtime/REPORT_ws2_cp.md",
    ],
}
with open(OUT, "w") as fh:
    json.dump(doc, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("wrote", OUT, "tp_total", tp_total, "cp_sim_total", cp_sim_total, "cp2_total", cp2_total,
      "fwd_same", fwd_same)

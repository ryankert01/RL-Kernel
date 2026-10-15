# Muse-Glimmer-30B strict POC — numerical contract (v1)

Every operator agent builds against THIS document. The strict path defines its own numerics;
it does not have to reproduce Hugging Face bit-for-bit. Correctness is `allclose` against an
FP64 (or fp32 fixed-order) reference; parity is `torch.equal` between two strict executions
(training vs prefill vs decode, batch 1 vs batch N, TP1 vs TP2/4/8, CPU-side chunking).

Pinned facts (from `rl_engine/models/muse_glimmer/museglimmer_manifest.json`):
hidden H=6656, layers L=52, heads Hq=32, kv heads Hkv=2 (group 16), head dim D=128,
Q/O width 4096, K/V width 256, FFN F=19968, vocab V=202048 (padded Vp=202752, 8 tiles of 25344),
window W=2048, rope_theta=5e5, rms_norm_eps=1e-5, post_norm_eps=1e-8, qk_scale=3.87,
output_multiplier m=0.19611613513818404, softcap T=20.0, layer i is global iff i % 4 == 3.
Weights and activations are bf16 unless stated. "fp32 fixed-order" means: accumulate in fp32,
reduction order fixed by the kernel and independent of batch size, batch position, TP degree
and sequence chunking.

## Global rules

R1. No cuBLAS/cuDNN/`torch.matmul`/SDPA in a strict op. GEMMs go through `det_gemm`
    (repo Triton kernel) with fixed tiles, no split-K, fp32 accumulation, one rounding to bf16
    at the declared output cast.
R2. TF32 disabled; no fast-math; `tl.dot` with `allow_tf32=False`; exp/log in fp32
    (`tl.exp`/`tl.log` fp32; no `exp2` tricks unless BOTH golden and kernel use them identically).
R3. Row-local ops (norms, gates, swiglu, softcap, rope) process one logical row end-to-end
    inside one program with a fixed block layout → batch invariance by construction.
R4. Each op exposes: `forward_fp32(*tensors) -> fp32` (golden, hand-written fixed-order torch,
    no matmul shortcuts), `forward(*bf16) -> bf16` (strict kernel), `backward` deterministic
    (assembled from det_gemm + row-local fp32 VJPs — never cuBLAS), and a `trace()` dict with
    `{op, dtype_in, dtype_acc, dtype_out, reduction_order, tile_config}` for provenance.
R5. Casts happen only where this document says "→ bf16".
R6. Kernel launch configuration (BLOCK sizes, num_warps, num_stages) is a compile-time constant
    per op per shape class, recorded in `trace()`. Autotuning is forbidden at runtime.

## Operators

### O1 `normalized_embedding`  (row: normalized_embedding)
x = E[token] (bf16 row of H) → fp32; ms = mean(x²) fixed-order over H; y = x * (ms + eps)^(-1/2),
eps = rms_norm_eps; → bf16. No weight, no sqrt(H) scaling.
Backward: dE[token] rows via sorted-segment accumulation (reuse repo canonical embedding backward);
the norm VJP is row-local fp32.

### O2 `centered_rmsnorm` / `fused_add_rmsnorm`  (row: fused_add_rmsnorm)
Decoder layer has four norms: input (eps=1e-5), post_attention (eps=1e-8), pre_feedforward
(eps=1e-5), post_feedforward (eps=1e-8); all are centered: y = x̂ * (1 + w), x̂ = x * rsqrt(mean(x²)+eps),
computed in fp32 (x→fp32, w→fp32), → bf16. Residual stream r is kept in bf16 (as HF does):
r' = r + post_norm(branch_out), both bf16 operands added in fp32 → bf16.
Fused form [F1]: `fused_post_norm_residual_pre_norm(branch_out, r, w_post, eps_post, w_pre, eps_pre)`
returns (r' bf16, pre_norm(r') bf16). Final norm: y = x̂ * w (NOT 1+w), eps=1e-5, → bf16.
Backward: row-local fp32 VJP for each norm; residual grads pass through.

### O3 `det_gemm` qualification  (row: dense_gemm_qualification)
Shapes per layer (N, K): q (4096,6656), k (256,6656), v (256,6656), attn_gate (4096,6656),
o (6656,4096), mlp_gate (19968,6656), up (19968,6656), down (6656,19968); lm_head (202752,6656)
with the padded rows zero. Column-parallel shards split N (per-element arithmetic unchanged).
Row-parallel (o, down) split K into 8 canonical chunks (512 / 2496): y = Σ_{c=0..7} y_c, with
y_c = det_gemm over chunk c in fp32, and the 8 fp32 partials summed in a FIXED binary tree
((0+1)+(2+3))+((4+5)+(6+7)) → bf16 once. At TP1 the 8 chunks are computed locally; at TP t each
rank owns 8/t consecutive chunks, reduces them locally in tree order, then the deterministic
all-gather + the same tree over ranks. This is `tp_canonical_tree` (WS2) and must be byte-equal
to TP1.
Backward: dX = det_gemm(dY, W), dW = det_gemm(dYᵀ, X) with the same chunking rule on K.

### O4 `qknorm_scaled_rope`  (row: qknorm_scaled_rope)
Input q (…,32,128), k (…,2,128) bf16 from O3. Per head: fp32, x̂ = x * (mean(x²)+eps)^(-1/2),
eps=1e-5, weightless. q̂ *= 3.87 (fp32). On LOCAL layers only: rotate_half RoPE over all 128
channels, cos/sin computed in fp32 as cos(p * θ^(-2i/128)) for absolute position p (table built
once by a fixed fp32 torch-CUDA function, cached, identical object on both runtimes),
rotation in fp32. → bf16 once at the end. Global layers: no rotation, same norm/scale.
Backward: row-local fp32 VJP (rotation is orthogonal; norm VJP standard).

### O5 `gated_attention`  (rows: global_gated_attention, local_gated_attention_train/rollout)
Inputs q̂ (bf16, from O4), K, V (bf16, from O4/O3), gate pre-activation g = attn_gate GEMM output
(bf16, 4096 = 32×128, from O3 on the input_layernorm output). Softmax scale s = 128^(-1/2) (fp32).
Mask: global layer: k_pos ≤ q_pos. Local layer: q_pos − W < k_pos ≤ q_pos (W=2048 keys incl. self).
Positions are absolute positions of the sequence (packed/chunked prefill uses the same positions).
Kernel arithmetic (online softmax, fp32):
  - KV is traversed in blocks of BLOCK_N=64 keys aligned at absolute key position 0, ascending.
    For a query at position p the first visited block is floor(max(0, p−W+1)/64) on local layers
    and 0 on global layers; the last is floor(p/64). Fully-masked blocks are a bitwise no-op
    (score=-inf → exp=0; running max unchanged; rescale factor exp(0)=1) so visiting extra
    blocks (because a query tile spans several positions) never changes a row's bits.
  - scores = tl.dot(q̂_tile, K_blockᵀ) fp32 (K=128) ; scaled by s in fp32 ; masked to -inf.
  - running max m, running sum l, acc (fp32, D=128) updated per block in this exact order:
    m_new = max(m, rowmax(scores)); α = exp(m − m_new); p = exp(scores − m_new);
    l = l·α + rowsum(p); acc = acc·α + tl.dot(p→bf16, V_block) ; m = m_new.
    (p is rounded to bf16 before the PV dot; this is the declared cast. Both golden and kernel
    do it.)
  - out = acc / l (fp32) → bf16 ; then gate: y = out * sigmoid(g→fp32) in fp32 → bf16
    (elementwise over the 4096 flattened channels; gate uses the attention module's INPUT
    i.e. input_layernorm output, computed by det_gemm in O3).
  - Query tile BLOCK_M=16 for decode (M padded) and BLOCK_M=64 for prefill/training; the per-row
    arithmetic above is identical for both because each row's ops do not depend on M. The
    tl.dot shapes have K=128 (QK) and K=64 (PV) in both configs. Agents MUST verify prefill vs
    decode bitwise on real shapes and report if the compiler breaks this (then fall back to a
    single BLOCK_M).
  - GQA: q head h reads kv head h // 16.
  - Rollout: K/V come from a paged cache (page = 16 tokens, so BLOCK_N=64 = 4 pages gathered via
    a block table); arithmetic unchanged. Sliding-window eviction: pages whose every key position
    ≤ p−W may be freed; the kernel masks them anyway so eviction timing cannot change bits.
Backward (training only): standard FlashAttention-2 style dQ/dK/dV with fp32 accumulation,
recomputing p from the saved (m, l) in the same order; dK/dV accumulated over query blocks in
ascending order with no atomics (one program per KV block iterating all query blocks). dgate =
out·sigmoid'(g)·dy row-local. K/V gradient rule for TP (WS2 kv_head_replication): dK/dV are
produced per canonical Q-head group (4 heads = TP8 shard) and combined in the canonical tree
at every physical TP including TP1 — i.e. the kernel emits 8 partial dK/dV per KV head (one
per 4-head group) and O3's tree sums them.
Sequence-split invariance: a token's out bits are identical whether it is processed in one
prefill, in chunked prefills of any split, or as a single decode step, given identical K/V bits.

### O6 `swiglu_ffn`  (row: swiglu_ffn)
g = det_gemm(x, W_gate), u = det_gemm(x, W_up) (bf16 outputs); h = silu(g→fp32) * (u→fp32)
→ bf16 ; y = row-parallel det_gemm(h, W_down) with the O3 K-chunk tree. silu(x) = x / (1 + exp(−x))
in fp32. Backward via det_gemm and row-local fp32 VJP of silu*mul.

### O7 `lm_head_softcap_logprob`  (rows: lm_head_softcap_logprob, vocab_parallel_softcap_logp)
z_tile = det_gemm(h, W_lm[tile]) over each of the 8 canonical vocab tiles (25344 rows each; the
last tile holds the 704 zero-padded rows) → fp32 (NOT rounded to bf16: the logits are consumed
in fp32). ẑ = T · tanh(z · m / T) in fp32 (tanh via fp32 `tl` math; golden uses torch fp32 tanh
— agents must confirm bitwise agreement between tl.tanh-equivalent and torch.tanh or implement
tanh as (1−e^{−2x})/(1+e^{−2x}) identically on both sides). Padded logits are excluded (masked)
from the logsumexp. Per tile: M_t = max ẑ, S_t = Σ exp(ẑ − M_t) fixed-order. Combine the 8
(M_t, S_t) in the fixed tree of O3 → logsumexp Z. logp(y) = ẑ[y] − Z (fp32). Output fp32 per
selected token. TP t: each rank owns 8/t tiles, partials exchanged by deterministic all-gather,
tree identical at every t.
Backward: dh = det_gemm(dẑ, W_lm) with dẑ = (onehot − softmax)·dlogp·(dẑ/dz = m·(1−tanh²)),
tile-chunked with the same tree; dW_lm = det_gemm(dẑᵀ, h) per tile.

## Model assembly (training forward of one layer, in order)
x_n = input_norm(r)                                   O2
q,k,v,g = det_gemm(x_n, Wq|Wk|Wv|Wg) — four GEMMs     O3  (never a fused QKV or QKVG GEMM)
q̂,k̂ = qknorm_scaled_rope(q,k, pos, local?)            O4
a = gated_attention(q̂,k̂,v,g, mask(local?/global))      O5
o = row-parallel det_gemm(a, Wo) (tree)               O3
r = r + post_attention_norm(o)  ;  x_f = pre_ffn_norm(r)   O2 (fused [F1])
f = swiglu_ffn(x_f)                                   O6
r = r + post_ffn_norm(f)                              O2
After layer 51: h = final_norm(r); logp = lm_head_softcap_logprob(h, y)     O2, O7

## Parity gates every agent must ship as tests (small shapes OK, real shapes at least once)
G1 forward correctness vs `forward_fp32`/FP64 within `rl_engine/contracts/profiles/precision/ws1.json`
   tolerances (do not edit tolerances);
G2 batch invariance: same logical row at batch 1 vs {2, 4, 16} and at different batch positions,
   `torch.equal`;
G3 sequence-split invariance (stateful ops: O5; O4 via positions): one-shot vs chunked vs
   per-token, `torch.equal`;
G4 backward: deterministic repeat (`torch.equal` across two runs) and allclose vs fp32 reference;
G5 trace(): emitted and stable.

import time, torch
torch.backends.cuda.matmul.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as op
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as dgt

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(123)
w = (torch.randn(op.VOCAB_SIZE, op.HIDDEN_SIZE, device=dev, generator=g) * 0.02).to(torch.bfloat16)
w_padded = op.pad_lm_head_weight(w); del w
T = 4096
h = torch.randn(T, op.HIDDEN_SIZE, device=dev, generator=g).to(torch.bfloat16)
targets = torch.randint(0, op.VOCAB_SIZE, (T,), device=dev, generator=g)
print("backend:", op.gemm_backend_name(None))

t0 = time.perf_counter()
with torch.no_grad():
    full = op.lm_head_softcap_logprob(h, w_padded, targets)
torch.cuda.synchronize()
print(f"T=4096 forward {time.perf_counter()-t0:.2f}s; last gemm launch class={dgt.last_launch().get('shape_class')} cfg={dgt.last_launch().get('config')}")
with torch.no_grad():
    full2 = op.lm_head_softcap_logprob(h, w_padded, targets)
print("repeat T=4096 equal:", torch.equal(full, full2))

# batch invariance across GEMM shape classes (decode16 / decode64 / prefill)
with torch.no_grad():
    for n in (1, 16, 17, 64, 65, 100, 256):
        for start in (0, 1000, 4096 - n):
            sub = op.lm_head_softcap_logprob(h[start:start+n], w_padded, targets[start:start+n])
            eq = torch.equal(sub, full[start:start+n])
            nd = (sub != full[start:start+n]).sum().item()
            md = (sub - full[start:start+n]).abs().max().item()
            print(f"rows[{start}:{start+n}] as T={n} ({dgt.last_launch().get('shape_class')}) vs T=4096: equal={eq} ndiff={nd} maxdiff={md:.3e}")

# backward: fast dW path (K=T=256, never exercised by the tests) vs legacy fallback at the same T
def run(Tb, gemm, seed=5):
    gg = torch.Generator(device=dev).manual_seed(seed)
    dlogp = torch.randn(Tb, device=dev, generator=gg)
    hh = h[:Tb].clone().requires_grad_(True)
    ww = w_padded.clone().requires_grad_(True)
    lp = op.lm_head_softcap_logprob(hh, ww, targets[:Tb], gemm=gemm)
    lp.backward(dlogp)
    return lp.detach(), hh.grad, ww.grad

torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
t0 = time.perf_counter()
lp_a, dh_a, dw_a = run(256, None)
torch.cuda.synchronize()
print(f"T=256 fwd+bwd fast: {time.perf_counter()-t0:.2f}s peak extra {(torch.cuda.max_memory_allocated()-base)/2**20:.0f} MiB; dW gemm class={dgt.last_launch().get('shape_class')} plan={dgt.last_launch().get('plan')}")
lp_b, dh_b, dw_b = run(256, None)
print("T=256 fast repeat equal: logp", torch.equal(lp_a, lp_b), "dh", torch.equal(dh_a, dh_b), "dw", torch.equal(dw_a, dw_b))
print("dw pad rows zero:", torch.count_nonzero(dw_a[op.VOCAB_SIZE:]).item() == 0, "finite:", torch.isfinite(dw_a.float()).all().item(), torch.isfinite(dh_a.float()).all().item())
lp_l, dh_l, dw_l = run(256, op.legacy_det_gemm_fp32)
print("T=256 legacy-vs-fast: logp equal", torch.equal(lp_a, lp_l), "max|d|", (lp_a-lp_l).abs().max().item(),
      "| dh equal", torch.equal(dh_a, dh_l), "ndiff", (dh_a != dh_l).sum().item(), "max|d|", (dh_a.float()-dh_l.float()).abs().max().item(),
      "| dw equal", torch.equal(dw_a, dw_l), "ndiff", (dw_a != dw_l).sum().item(), "max|d|", (dw_a.float()-dw_l.float()).abs().max().item())
del lp_l, dh_l, dw_l
# T=255 -> dW silently takes the legacy kernel (K=255 % 256 != 0)
lp_c, dh_c, dw_c = run(255, None)
print("T=255 ran; dW gemm last launch class (should still be the T=255 dh launch, since dW went legacy):", dgt.last_launch().get('shape'))
del lp_c, dh_c, dw_c, dw_a, dw_b, dh_a, dh_b

# T=4096 fwd+bwd memory/time with the fast backend
torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
t0 = time.perf_counter()
lp4, dh4, dw4 = run(4096, None)
torch.cuda.synchronize()
print(f"T=4096 fwd+bwd fast: {time.perf_counter()-t0:.2f}s peak extra {(torch.cuda.max_memory_allocated()-base)/2**20:.0f} MiB; logp equal fwd-only: {torch.equal(lp4, full)}")

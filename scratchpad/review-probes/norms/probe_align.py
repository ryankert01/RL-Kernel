"""Probe: do Triton pointer-alignment specializations change the row bits?
Same logical rows, same weights, but the weight / input / ids / dy tensors are views at
non-16-byte-aligned storage offsets (what a flattened parameter buffer or an id slice can give)."""
import torch
from rl_engine.models.muse_glimmer.ops import norms as N

H = N.HIDDEN
T = 64


def bf16(shape, seed, scale=1.0, shift=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale + shift).to(torch.bfloat16)


x = bf16((T, H), 1, scale=2.0)
w = bf16((H,), 2, scale=0.1)
ref = N.centered_rmsnorm(x, w, eps=1e-5)

# weight at element offsets 1 (2 B), 4 (8 B), 8 (16 B) inside a bigger buffer
for off in (1, 2, 4, 8):
    buf = torch.zeros(H + 64, device="cuda", dtype=torch.bfloat16)
    buf[off:off + H] = w
    wv = buf[off:off + H]
    assert wv.is_contiguous() and wv.data_ptr() % 16 == (off * 2) % 16
    y = N.centered_rmsnorm(x, wv, eps=1e-5)
    print(f"w offset {off} elems ({off*2} B): torch.equal vs aligned =", torch.equal(y, ref),
          "| mismatching elems:", (y != ref).sum().item())

# input rows at non-16B-aligned storage offset
for off in (1, 4, 8):
    buf = torch.zeros(T * H + 64, device="cuda", dtype=torch.bfloat16)
    buf[off:off + T * H] = x.reshape(-1)
    xv = buf[off:off + T * H].view(T, H)
    assert xv.is_contiguous()
    y = N.centered_rmsnorm(xv, w, eps=1e-5)
    print(f"x offset {off} elems ({off*2} B): torch.equal vs aligned =", torch.equal(y, ref),
          "| mismatching elems:", (y != ref).sum().item())

# backward dy at non-aligned offset; and w offset in backward
dy = bf16((T, H), 3)
xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
N.centered_rmsnorm(xx, ww, eps=1e-5).backward(dy)
dx_ref, dw_ref = xx.grad, ww.grad
for off in (1, 4):
    buf = torch.zeros(T * H + 64, device="cuda", dtype=torch.bfloat16)
    buf[off:off + T * H] = dy.reshape(-1)
    dyv = buf[off:off + T * H].view(T, H)
    xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    N.centered_rmsnorm(xx, ww, eps=1e-5).backward(dyv)
    print(f"bwd dy offset {off}: dx equal =", torch.equal(xx.grad, dx_ref), "dw equal =",
          torch.equal(ww.grad, dw_ref))
for off in (1, 4):
    buf = torch.zeros(H + 64, device="cuda", dtype=torch.bfloat16)
    buf[off:off + H] = w
    xx = x.clone().requires_grad_(True)
    wv = buf[off:off + H].detach().requires_grad_(True)
    N.centered_rmsnorm(xx, wv, eps=1e-5).backward(dy)
    print(f"bwd w offset {off}: dx equal =", torch.equal(xx.grad, dx_ref), "dw equal =",
          torch.equal(wv.grad, dw_ref))

# embedding: ids slice at 8-byte offset (int64), table aligned
table = bf16((4099, H), 7, scale=0.02)
ids = torch.randint(0, 4099, (T + 1,), device="cuda", generator=torch.Generator(device="cuda").manual_seed(8))
e_ref = N.normalized_embedding(table, ids[1:].clone())
e_view = N.normalized_embedding(table, ids[1:])  # int64 already -> no copy, 8-byte aligned ptr
print("embedding ids 8B-offset view vs clone:", torch.equal(e_ref, e_view),
      "ptr%16 =", ids[1:].data_ptr() % 16)

# fused op with w_post / w_pre at odd offsets
b = bf16((T, H), 11)
r = bf16((T, H), 12, scale=3.0)
f_ref = N.fused_post_norm_residual_pre_norm(b, r, w, 1e-8, w, 1e-5)
for off in (1, 4):
    buf = torch.zeros(H + 64, device="cuda", dtype=torch.bfloat16)
    buf[off:off + H] = w
    wv = buf[off:off + H]
    f = N.fused_post_norm_residual_pre_norm(b, r, wv, 1e-8, wv, 1e-5)
    print(f"fused w offset {off}: rnew equal =", torch.equal(f[0], f_ref[0]), "pre equal =",
          torch.equal(f[1], f_ref[1]))

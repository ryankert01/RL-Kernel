import torch
import triton
import triton.language as tl

print(torch.__version__, triton.__version__)
s = torch.ones((1,), dtype=torch.int32, device="cuda")
torch._assert_async(s, "ok message")
torch.cuda.synchronize()
print("ok passes")


@triton.jit
def k(x_ptr):
    tl.device_assert(tl.load(x_ptr) == 1, "device assert fired")


z = torch.zeros((1,), dtype=torch.int32, device="cuda")
k[(1,)](z)
torch.cuda.synchronize()
print("tl.device_assert without debug: no error (no-op)")
s.zero_()
torch._assert_async(s, "strict gated attention: invalid metadata")
try:
    torch.cuda.synchronize()
    print("NO ERROR")
except Exception as e:  # noqa: BLE001
    print("raised", type(e).__name__, str(e)[:300])

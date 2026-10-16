"""Does a first rope_rows() call inside CUDA-graph capture poison the canonical-table cache?"""
import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

dev = torch.device("cuda")
pos = torch.arange(16, device=dev) + 1000
# fill the caching allocator with junk so uninitialized memory is visibly non-table
junk = torch.full((2 * 131072 * 128,), 7.0, device=dev)
del junk
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
err = None
with torch.cuda.stream(s):
    try:
        with torch.cuda.graph(g):
            c_rows, s_rows = op.rope_rows(pos)  # first call: builds the canonical table in capture
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
print("capture error:", err)
if err is None:
    cc, sc = op.canonical_rope_table(dev)
    ref_c, ref_s = op._rope_table_in(
        torch.arange(131072, device=dev), theta=op.ROPE_THETA, dim=128, device=dev, dtype=torch.float32
    )
    print("cached table == reference BEFORE replay:", torch.equal(cc, ref_c), torch.equal(sc, ref_s),
          "mismatching elements:", int((cc != ref_c).sum()))
    eager_c, _ = op.rope_rows(pos)
    print("eager rope_rows after capture == reference rows:", torch.equal(eager_c, ref_c[pos]))
    g.replay()
    torch.cuda.synchronize()
    print("cached table == reference AFTER replay:", torch.equal(cc, ref_c))
    print("sha256 now:", op.rope_table_sha256(dev) == op.CANONICAL_ROPE_TABLE_SHA256)

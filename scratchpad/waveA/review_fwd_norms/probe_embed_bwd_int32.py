# normalized_embedding backward reuses the repo sorted-segment kernel, whose row offsets are
# int32 (`row * hidden`). With H = 6656 the offset overflows once a backward call holds
# >= 2**31 / 6656 = 322,639 token rows. Rows are all 1.0, so the exact dweight is the token count.
from __future__ import annotations

import json

import torch

from rl_engine.backends.shared.triton.embedding.embedding import _embedding_grad_weight

DEV = torch.device("cuda")
H = 6656
n = 330_000
split = 322_000
ids = torch.zeros(n, dtype=torch.int64, device=DEV)
ids[split:] = 1
rows = torch.ones(n, H, dtype=torch.float32, device=DEV)
out = {"n_rows": n, "int32_limit_rows": (2**31) // H}
try:
    dw = _embedding_grad_weight(ids, rows, weight_shape=(4, H), weight_dtype=torch.float32)
    torch.cuda.synchronize()
    out.update(
        raised=None,
        token0_expected=split,
        token0_unique_values=dw[0].unique().tolist()[:5],
        token1_expected=n - split,
        token1_unique_values=dw[1].unique().tolist()[:5],
    )
except Exception as e:  # noqa: BLE001
    out.update(raised=repr(e)[:300])
print(json.dumps(out))

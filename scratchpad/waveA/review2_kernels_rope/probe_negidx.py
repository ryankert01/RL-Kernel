"""rope_rows(check_range=False) with a negative / too-large position: does index_select fail
loudly (as the docstring claims) or wrap?  Run in its own process (a device assert kills it)."""
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
which = sys.argv[1]
p = {"neg": [-1, 3], "big": [op.ROPE_TABLE_POSITIONS, 3]}[which]
out = {"case": which, "positions": p}
try:
    c, s = op.rope_rows(torch.tensor(p, device=DEV), check_range=False)
    torch.cuda.synchronize()
    full_c, _ = op.canonical_rope_table(DEV)
    out["raised"] = False
    out["row0_equals_last_table_row"] = bool(torch.equal(c[0], full_c[-1]))
except Exception as e:  # noqa
    out["raised"] = True
    out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
print(json.dumps(out))

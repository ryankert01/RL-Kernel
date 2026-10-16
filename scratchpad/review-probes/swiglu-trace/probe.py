import json
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as m
from rl_engine.models.muse_glimmer.ops import det_gemm as dg
print("fast_det_gemm has .trace:", hasattr(dg.fast_det_gemm, "trace"))
g = m.default_gemm()
print("default_gemm type:", type(g).__name__)
t = m.trace()
print("gemm trace keys:", sorted(t["tile_config"]["gemm"].keys()))
print(json.dumps(t["tile_config"]["gemm"], indent=1))
print("module trace tile_config keys:", list(dg.trace()["tile_config"].keys()))

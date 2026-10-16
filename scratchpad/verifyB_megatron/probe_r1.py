"""R1 probe: every way a strict op can go missing must raise in strict mode."""
import json
import os
import sys

import torch  # noqa: F401

from rl_engine.models.muse_glimmer.runtime import megatron_strict as ms
from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk

results = {}
real = sk._optional_module


def run(tag, broken=None, env=None, quiet_fake=False):
    sk.reset_resolution()
    saved = {k: os.environ.get(k) for k in ("MUSE_STRICT_ATTN", "MUSE_STRICT_GEMM")}
    for k in saved:
        os.environ.pop(k, None)
    if env:
        os.environ.update(env)

    def fake(path):
        if broken and path.endswith(broken):
            if not quiet_fake:
                sk._IMPORT_ERRORS[path] = "ImportError: simulated"
            return None
        return real(path)

    sk._optional_module = fake
    try:
        try:
            ops = sk.resolve_ops()
            r = {"raised": None, "all_strict": ops.provenance()["all_strict"]}
        except Exception as e:  # noqa: BLE001
            r = {"raised": type(e).__name__, "msg": str(e)[:300]}
        try:
            ms.install_museglimmer_megatron_strict(object(), hf_path=None)
            r["binding"] = "no raise"
        except Exception as e:  # noqa: BLE001
            r["binding"] = type(e).__name__ + ": " + str(e)[:120]
    finally:
        sk._optional_module = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    results[tag] = r
    print(tag, r)


run("baseline")
run("attn_off", env={"MUSE_STRICT_ATTN": "off"})
run("attn_fallback", env={"MUSE_STRICT_ATTN": "fallback"})
run("attn_sibling_old_value", env={"MUSE_STRICT_ATTN": "sibling"})
run("gemm_cublas", env={"MUSE_STRICT_GEMM": "cublas"})
run("gemm_triton_repo", env={"MUSE_STRICT_GEMM": "triton_repo"})
run("gemm_bogus", env={"MUSE_STRICT_GEMM": "bogus"})
for mod in (
    "ops.norms", "ops.qknorm_rope", "ops.swiglu_ffn", "ops.det_gemm", "ops.canonical_tree",
    "ops.gated_attention", "triton.gated_attention_bwd", "ops.lm_head_softcap_logprob",
    "triton.softcap_logprob",
):
    run("missing_" + mod, broken=mod)
    run("missing_silent_" + mod, broken=mod, quiet_fake=True)
json.dump(results, open(sys.argv[1], "w"), indent=1)

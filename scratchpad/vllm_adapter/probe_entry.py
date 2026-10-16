import sys
import time

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a7944ab684b3629cf")
import warnings

warnings.filterwarnings("ignore")
t0 = time.time()
try:
    import rl_engine.integrations.engines.rollout.vllm.runtime as rt

    print("shared runtime import OK in %.1fs" % (time.time() - t0), rt.register_vllm_plugin)
except Exception as exc:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    print("shared runtime import FAILED:", type(exc).__name__, exc)
try:
    from rl_engine.distributed.algorithms.collectives import collective_for_group  # noqa: F401

    print("collectives import OK")
except Exception as exc:  # noqa: BLE001
    print("collectives import FAILED:", type(exc).__name__, exc)
try:
    from rl_engine.backends.shared.triton.gemm import det_gemm as repo_det

    print("repo triton det_gemm kernel:", hasattr(repo_det, "_det_gemm_fp32_kernel"))
except Exception as exc:  # noqa: BLE001
    print("repo det_gemm import FAILED:", type(exc).__name__, exc)

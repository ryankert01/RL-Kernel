# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Behavioural before/after probe for the wave B review of the rope / reference rows.

Uses only names that exist both before (f91e8b6) and after the fixes, so the SAME script runs
against an old tree (``git archive f91e8b6 | tar -x -C <old>``; ``PYTHONPATH=<old>``) and the
current one, and records what each build DOES (raises / silently accepts, how many elements
differ). One JSON object per run.

usage: PYTHONPATH=<tree> python probe_wave_b.py <out.json>   (one CUDA GPU)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
HQ, HK, D = 32, 2, 128
ENV = "MUSEGLIMMER_O4_REQUIRE_LAYER_KIND"


def _qk(t: int, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(t, HQ, D, generator=g) * 2).to(torch.bfloat16).to(DEV)
    k = (torch.randn(t, HK, D, generator=g) * 2).to(torch.bfloat16).to(DEV)
    return q, k


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e)[:160]}"


def _sha(*ts: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def r1_layer_kind() -> dict:
    """local=None on a sliding layer with no table: silent NoPE unless the env var makes the
    layer kind mandatory."""
    q, k = _qk(4096, 5)
    cos, sin = op.rope_table(torch.arange(4096, device=DEV), device=DEV)
    ref = op.qknorm_scaled_rope(q, k, cos, sin)
    out: dict = {}
    for env in ("unset", "1"):
        if env == "unset":
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = env
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                got = op.qknorm_scaled_rope(q, k, None, None)
                rec = {"raised": False,
                       "q_hat_elements_differing_from_rope_path": int((got[0] != ref[0]).sum()),
                       "elements": got[0].numel()}
            except Exception as e:  # noqa: BLE001
                rec = {"raised": True, "error": _err(e)}
        rec["warnings"] = [str(w.message)[:120] for w in caught]
        out[f"env_{env}"] = rec
    os.environ.pop(ENV, None)
    return out


def r2_builders() -> dict:
    from rl_engine.models.muse_glimmer.reference import golden_ops

    out: dict = {}
    for name, fn in (
        ("ops.qknorm_rope.rope_table", lambda dev: op.rope_table(torch.arange(8), device=dev)),
        ("reference.golden_ops.rope_table",
         lambda dev: golden_ops.rope_table(torch.arange(8), theta=5e5, dim=D, device=dev)),
    ):
        try:
            fn("cpu")
            out[f"{name}(cpu)"] = "accepted"
        except Exception as e:  # noqa: BLE001
            out[f"{name}(cpu)"] = "raised " + _err(e)
    before = set(op._CANONICAL_TABLES)
    op.rope_table(torch.arange(16, device=DEV), dim=96, device=DEV)  # a key nobody built
    out["rope_table_reads_the_canonical_cache"] = len(set(op._CANONICAL_TABLES) - before) > 0
    return out


def r3_in_place() -> dict:
    out: dict = {}
    c, _s = op.canonical_rope_table(DEV, dim=64)  # private key, never the model's table
    c.mul_(1 + 2**-20)
    try:
        op.canonical_rope_table(DEV, dim=64)
        out["in_place_change_detected"] = False
    except Exception as e:  # noqa: BLE001
        out["in_place_change_detected"] = True
        out["in_place_error"] = _err(e)
    # .data write on the model's table: verify=True must re-hash (restored afterwards)
    cos, sin = op.canonical_rope_table(DEV, verify=True)
    pinned = op.CANONICAL_ROPE_TABLE_SHA256
    old = cos.data[7, 3].clone()
    try:
        cos.data[7, 3] = old + 1.0
        try:
            op.canonical_rope_table(DEV, verify=True)
            out["verify_after_data_write_raised"] = False
        except Exception as e:  # noqa: BLE001
            out["verify_after_data_write_raised"] = True
            out["verify_error"] = _err(e)[:120]
        try:
            out["logged_digest_equals_pinned_after_write"] = op.rope_table_sha256(DEV) == pinned
        except Exception as e:  # noqa: BLE001
            out["logged_digest_after_write"] = _err(e)
        out["actual_digest_equals_pinned_after_write"] = _sha(cos, sin) == pinned
    finally:
        cos.data[7, 3] = old
    out["restored_digest_equals_pinned"] = _sha(cos, sin) == pinned
    return out


def r4_capture() -> dict:
    """First use of a table key inside CUDA-graph capture."""
    out: dict = {}
    theta = 12345.0  # a key nobody built
    pos = torch.arange(8, device=DEV)
    q, k = _qk(64, 9)
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g):
            op.rope_rows(pos, theta=theta)
        out["first_build_in_capture_raised"] = False
    except Exception as e:  # noqa: BLE001
        out["first_build_in_capture_raised"] = True
        out["error"] = _err(e)
    torch.cuda.synchronize()
    key = (torch.cuda.current_device(), theta, D)
    if not out["first_build_in_capture_raised"] and key in op._CANONICAL_TABLES:
        c, s = op._CANONICAL_TABLES[key]
        ref_c, ref_s = op._rope_table_in(torch.arange(op.ROPE_TABLE_POSITIONS, device=DEV),
                                         theta=theta, dim=D, device=DEV, dtype=torch.float32)
        out["cached_table_correct_before_replay"] = bool(torch.equal(c, ref_c)
                                                         and torch.equal(s, ref_s))
        p2 = torch.arange(64, device=DEV) + 1000
        rc, rs = op.rope_rows(p2, theta=theta)
        out["eager_rows_after_capture_wrong_elements"] = int(
            (rc != ref_c[p2]).sum() + (rs != ref_s[p2]).sum())
    out["cached_after"] = key in op._CANONICAL_TABLES
    return out


def inference_mode_build() -> dict:
    with torch.inference_mode():
        c, _ = op.canonical_rope_table(DEV, theta=777.0)
    return {"table_is_inference_tensor": bool(c.is_inference())}


def ref_decode_batch_duplicate_ids() -> dict:
    from rl_engine.models.muse_glimmer.reference.strict_model import (
        MuseGlimmerStrictModel,
        Session,
    )
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

    cfg = StrictConfig.tiny()
    model = MuseGlimmerStrictModel(MuseGlimmerWeights.random(cfg, device=DEV, seed=1,
                                                             layout="both"), "golden")
    g = torch.Generator(device="cpu").manual_seed(3)
    p = torch.randint(0, cfg.vocab_size, (40,), generator=g)
    cache = model.new_batch_cache(64, 2)
    model.prefill(Session(cache=cache, seq=0), p, targets=torch.cat([p[1:], p[:1]]))
    model.prefill(Session(cache=cache, seq=1), p[:9], targets=torch.cat([p[1:9], p[:1]]))
    one = model.decode_batch(cache, [0], [5], [6], commit=False).cpu()
    out: dict = {}
    try:
        dup = model.decode_batch(cache, [0, 0], [5, 77], [6, 6], commit=False).cpu()
        out["duplicate_ids_raised"] = False
        out["row0_equals_batch1"] = bool(torch.equal(dup[0:1].view(torch.int32),
                                                     one.view(torch.int32)))
        out["row0_abs_diff"] = float((dup[0] - one[0]).abs())
    except Exception as e:  # noqa: BLE001
        out["duplicate_ids_raised"] = True
        out["error"] = _err(e)
    # negative ids are NOT probed on old builds: they trip a device-side assert (kills CUDA)
    return out


def ref_xrt_checks(tmp_dir: str) -> dict:
    from types import SimpleNamespace

    from rl_engine.models.muse_glimmer.reference.xrt_scoring import score_token_file

    class Stub:
        device = DEV
        ops = SimpleNamespace(backend="stub", provenance={})

        def logprobs(self, prompt, completion, *, mode, chunk_sizes=None, **_kw):
            out = torch.zeros(int(completion.numel()))
            if mode == "prefill":
                out[0] = -0.0
            return out

    path = os.path.join(tmp_dir, "probe_tokens.json")
    with open(path, "w") as f:
        json.dump({"schema": "muse_glimmer_xrt_v1", "sequences": [
            {"id": "a", "token_ids": list(range(12)), "completion_start": 8}]}, f)
    a = score_token_file(Stub(), path, check_modes=("prefill",))
    b = score_token_file(Stub(), path, check_modes=())
    return {
        "minus_zero_vs_plus_zero_counted_equal": a["checks"]["sequences"][0]["modes"]["prefill"][
            "byte_equal"],
        "no_checks_reported_as_pass": b["checks"]["all_byte_equal_to_train"],
    }


def main() -> int:
    out_path = sys.argv[1]
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    res: dict = {"python": sys.executable, "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(), "module": op.__file__}
    for name, fn in (
        ("r1_layer_kind", r1_layer_kind),
        ("r2_builders", r2_builders),
        ("r3_in_place", r3_in_place),
        ("r4_capture", r4_capture),
        ("inference_mode_build", inference_mode_build),
        ("ref_decode_batch_duplicate_ids", ref_decode_batch_duplicate_ids),
        ("ref_xrt_checks", lambda: ref_xrt_checks(os.path.dirname(os.path.abspath(out_path)))),
    ):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = {"probe_error": _err(e)}
    res["wall_s"] = round(time.time() - t0, 1)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

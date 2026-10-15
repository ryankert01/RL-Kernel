# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CLI: Muse-Glimmer-30B on Megatron Bridge — strict binding checks, xrt scoring and benchmarks.

One process drives a :class:`StrictTrainer` (TP rank 0 in-process, rank 1 spawned at TP2) and
runs, in order, the requested checks. Every number goes to ``--results`` (one JSON object,
rewritten after each check so a crash keeps what finished)::

    identity     627 HF text tensors reconstructed from the Bridge model vs safetensors bytes
    xrt          strict scoring of the shared token file (LoRA B = 0) -> xrt_megatron_strict_tp<k>.json
    repeat       second strict scoring, bitwise mismatch count
    lora_noop    LoRA disabled vs LoRA with B = 0, bitwise mismatch count
    native       Bridge native forward on the same tokens -> xrt_megatron_native_tp<k>.json + diff
    grad_repeat  B ~ N(0, std): two recompute backward passes of a fixed surrogate loss,
                 bitwise LoRA-grad comparison; grads saved to --grads-out for the TP diff
    step         one AdamW step on that loss; post-step LoRA hash and logprob change
    bench        strict vs native forward / forward+backward(+step) time and peak memory at
                 several token counts

Example (TP1; TP2 is the same with ``-n 2 --tp 2``)::

    ~/bin/gpu-lease -n 1 --pool 2-3 -- env PYTHONPATH=$PWD ~/envs/megatron/bin/python -m \\
        rl_engine.models.muse_glimmer.runtime.run_megatron_strict --tp 1 \\
        --checks identity,xrt,repeat,lora_noop,native,grad_repeat,step,bench \\
        --results out/tp1.json --grads-out out/grads_tp1.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime.megatron_strict import hex_bits
from rl_engine.models.muse_glimmer.runtime.strict_trainer import StrictTrainer

_REPO = Path(__file__).resolve().parents[4]
DEFAULT_XRT = _REPO / "reports" / "muse_glimmer_poc" / "data" / "xrt_tokens.json"
DEFAULT_XRT_OUT = _REPO / "reports" / "muse_glimmer_poc" / "data"
XRT_OUT_SCHEMA = "muse_glimmer_xrt_logprobs_v1"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for mod in ("triton", "transformer_engine", "megatron.core", "megatron.bridge", "transformers"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            info[mod.replace(".", "_")] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            info[mod.replace(".", "_")] = f"unavailable: {exc}"
    return info


def load_xrt(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "muse_glimmer_xrt_v1":
        raise ValueError(f"{path}: not a muse_glimmer_xrt_v1 file")
    return data


def bit_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    a = a.detach().float().cpu().contiguous()
    b = b.detach().float().cpu().contiguous()
    if a.shape != b.shape:
        return max(a.numel(), b.numel())
    return int((a.view(torch.int32) != b.view(torch.int32)).sum().item())


def write_xrt(path: Path, xrt: dict[str, Any], lps: list[torch.Tensor], *, mode: str, tp: int,
              backend: str, extra: dict[str, Any]) -> None:
    seqs = []
    for s, lp in zip(xrt["sequences"], lps, strict=True):
        expected = len(s["token_ids"]) - int(s["completion_start"])
        if lp.numel() != expected:
            raise ValueError(f"{s['id']}: {lp.numel()} logprobs, expected {expected}")
        seqs.append({"id": s["id"], "logprobs_hex": hex_bits(lp), "logprobs": lp.float().tolist()})
    payload = {
        "schema": XRT_OUT_SCHEMA,
        "runtime": "megatron_bridge",
        "mode": mode,
        "tp": tp,
        "backend": backend,
        "checkpoint_revision": xrt.get("checkpoint_revision"),
        "sequences": seqs,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def sha256_fp32(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().view(torch.int32).numpy().tobytes()).hexdigest()


def surrogate_weights(spans: list[int]) -> list[torch.Tensor]:
    """Fixed per-token loss weights g (a GRPO-like signed pattern, deterministic, TP-free)."""
    out = []
    for i, n in enumerate(spans):
        sign = 1.0 if i % 2 == 0 else -1.0
        out.append(torch.full((n,), sign / n, dtype=torch.float32))
    return out


def surrogate_loss(lps: list[torch.Tensor], weights: list[torch.Tensor]) -> torch.Tensor:
    total = None
    for lp, g in zip(lps, weights, strict=True):
        term = (lp * g.to(lp.device)).sum()
        total = term if total is None else total + term
    return total


class Results:
    def __init__(self, path: Path, base: dict[str, Any]) -> None:
        self.path = path
        self.data = base
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def _sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/models/Muse-Glimmer-30B"))
    ap.add_argument("--tp", type=int, default=1, choices=(1, 2))
    ap.add_argument("--xrt-file", default=str(DEFAULT_XRT))
    ap.add_argument("--xrt-out-dir", default=str(DEFAULT_XRT_OUT))
    ap.add_argument("--results", required=True, help="JSON file with every measured number")
    ap.add_argument("--grads-out", default=None, help="torch.save of the TP-gathered fp32 LoRA grads")
    ap.add_argument(
        "--checks", default="identity,xrt,repeat,lora_noop,native,grad_repeat,step,bench",
        help="comma list (see module docstring)",
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-b-std", type=float, default=1e-3, help="LoRA B init for grad checks")
    ap.add_argument("--micro-batch-tokens", type=int, default=8192)
    ap.add_argument("--bench-tokens", default="512,1024,2048,4096")
    ap.add_argument("--no-checkpoint", action="store_true", help="disable per-layer recompute")
    args = ap.parse_args(argv)
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    known = {"identity", "xrt", "repeat", "lora_noop", "native", "grad_repeat", "step", "bench"}
    if set(checks) - known:
        ap.error(f"unknown checks {sorted(set(checks) - known)}")

    xrt = load_xrt(Path(args.xrt_file))
    seqs = [list(map(int, s["token_ids"])) for s in xrt["sequences"]]
    starts = [int(s["completion_start"]) for s in xrt["sequences"]]
    ids = [s["id"] for s in xrt["sequences"]]
    spans = [len(s) - st for s, st in zip(seqs, starts, strict=True)]
    tp = args.tp
    res = Results(
        Path(args.results),
        {
            "row": "megatron_runtime",
            "tp": tp,
            "status": "running",
            "commit": _git_sha(),
            "env": env_info(),
            "command": " ".join([sys.executable, "-m", __spec__.name if __spec__ else __file__, *sys.argv[1:]]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inputs": {
                "xrt_file": str(args.xrt_file),
                "sequences": [{"id": i, "len": len(s), "completion_start": st} for i, s, st in zip(ids, seqs, starts, strict=True)],
                "total_tokens": int(sum(len(s) for s in seqs)),
                "scored_tokens": int(sum(spans)),
            },
            "config": {k: v for k, v in vars(args).items()},
            "checks": {},
        },
    )
    C = res.data["checks"]

    t0 = time.perf_counter()
    trainer = StrictTrainer(
        args.model, tensor_parallel_size=tp, gpu_ids=list(range(tp)), lora_rank=args.lora_rank,
        lr=args.lr, strict=True, seed=args.seed, checkpoint_layers=not args.no_checkpoint,
        micro_batch_tokens=args.micro_batch_tokens,
    )
    res.data["load"] = {
        "init_seconds": time.perf_counter() - t0,
        "bridge_load_seconds": trainer._state.load_seconds,
        "memory_after_load": trainer.memory(reset=True),
    }
    res.data["provenance"] = trainer.provenance()
    res.flush()
    print(f"[load] {res.data['load']}", flush=True)
    strict_backend = "triton_strict" if res.data["provenance"]["ops"].get("all_strict") else "mixed"
    base: list[torch.Tensor] | None = None

    try:
        if "identity" in checks:
            rep = trainer.weight_identity(args.model)
            C["identity"] = rep
            res.flush()
            print(f"[identity] {rep['equal']}/{rep['total']} equal; mismatches="
                  f"{[m['name'] for m in rep['mismatches']]}", flush=True)

        if {"xrt", "repeat", "lora_noop", "native"} & set(checks):
            trainer.memory(reset=True)
            ta = _sync_time()
            base = trainer.score(seqs, starts)
            tb = _sync_time()
            C["xrt_strict"] = {
                "seconds": tb - ta,
                "memory": trainer.memory(reset=True),
                "lora": "enabled, B = 0",
                "per_sequence": [
                    {"id": i, "n": int(lp.numel()), "sum_logp": float(lp.double().sum()),
                     "min_logp": float(lp.min())}
                    for i, lp in zip(ids, base, strict=True)
                ],
            }
            if "xrt" in checks:
                out = Path(args.xrt_out_dir) / f"xrt_megatron_strict_tp{tp}.json"
                write_xrt(out, xrt, base, mode="strict", tp=tp, backend=strict_backend,
                          extra={"commit": res.data["commit"], "ops": res.data["provenance"]["ops"],
                                 "lora": "rank 16, B = 0 (exact no-op)"})
                C["xrt_strict"]["file"] = str(out)
            res.flush()
            print(f"[xrt] strict scored {sum(spans)} tokens in {tb - ta:.2f}s", flush=True)

        if "repeat" in checks:
            again = trainer.score(seqs, starts)
            mism = [bit_mismatches(a, b) for a, b in zip(again, base, strict=True)]
            C["repeat"] = {"mismatches_per_sequence": mism, "total_mismatches": sum(mism),
                           "tokens": sum(spans), "torch_equal": sum(mism) == 0}
            res.flush()
            print(f"[repeat] mismatches {sum(mism)}/{sum(spans)}", flush=True)

        if "lora_noop" in checks:
            trainer.set_lora_enabled(False)
            without = trainer.score(seqs, starts)
            trainer.set_lora_enabled(True)
            mism = [bit_mismatches(a, b) for a, b in zip(without, base, strict=True)]
            C["lora_noop"] = {"mismatches_per_sequence": mism, "total_mismatches": sum(mism),
                              "tokens": sum(spans), "torch_equal": sum(mism) == 0}
            res.flush()
            print(f"[lora_noop] mismatches {sum(mism)}/{sum(spans)}", flush=True)

        if "native" in checks:
            trainer.memory(reset=True)
            ta = _sync_time()
            nat = trainer.native_logprobs(seqs, starts)
            tb = _sync_time()
            out = Path(args.xrt_out_dir) / f"xrt_megatron_native_tp{tp}.json"
            write_xrt(out, xrt, nat, mode="native", tp=tp, backend="megatron_bridge_te_native",
                      extra={"commit": res.data["commit"],
                             "note": "Bridge MuseGlimmerModel forward (TE kernels, bf16 logits, "
                                     "fp32 log_softmax), no LoRA"})
            per_seq = []
            for i, a, b in zip(ids, nat, base, strict=True):
                d = (a.double() - b.double()).abs()
                per_seq.append({"id": i, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                "bit_mismatches": bit_mismatches(a, b), "n": int(a.numel()),
                                "abs_diff_series": [float(x) for x in d.tolist()]})
            C["native"] = {
                "seconds": tb - ta,
                "memory": trainer.memory(reset=True),
                "file": str(out),
                "max_abs_vs_strict": max(p["max_abs"] for p in per_seq),
                "mean_abs_vs_strict": float(
                    sum(p["mean_abs"] * p["n"] for p in per_seq) / sum(p["n"] for p in per_seq)
                ),
                "bit_mismatches_vs_strict": sum(p["bit_mismatches"] for p in per_seq),
                "per_sequence": per_seq,
            }
            res.flush()
            print(f"[native] max|d| {C['native']['max_abs_vs_strict']:.4g} mean|d| "
                  f"{C['native']['mean_abs_vs_strict']:.4g}", flush=True)

        weights = surrogate_weights(spans)
        if "grad_repeat" in checks or "step" in checks:
            trainer.reset_lora_b(args.grad_b_std, args.seed + 1)
            C["grad_setup"] = {"lora_b_std": args.grad_b_std, "lora_b_seed": args.seed + 1,
                               "loss": "sum_i sum_t lp[i,t] * g[i,t], g = (+1 if i even else -1)/n_i"}

        if "grad_repeat" in checks:
            runs = []
            for rep_i in range(2):
                trainer.memory(reset=True)
                ta = _sync_time()
                lps = trainer.token_logprobs(seqs, starts)
                tb = _sync_time()
                info = trainer.backward(surrogate_loss(lps, weights))
                tc = _sync_time()
                g = trainer.lora_grads()
                trainer.zero_grad()
                runs.append({"lps": [lp.detach().cpu() for lp in lps], "grads": g["grads"],
                             "info": info, "fwd_s": tb - ta, "bwd_s": tc - tb,
                             "memory": trainer.memory(reset=True),
                             "replicated_checked": g["replicated_checked"],
                             "replicated_differing": g["replicated_differing"]})
                print(f"[grad_repeat] pass {rep_i}: {info}", flush=True)
            g1, g2 = runs[0]["grads"], runs[1]["grads"]
            keys = sorted(g1)
            per_tensor = []
            for k in keys:
                a, b = g1[k], g2[k]
                per_tensor.append({
                    "key": k, "numel": int(a.numel()), "repeat_mismatches": bit_mismatches(a, b),
                    "abs_max": float(a.abs().max()), "l2": float(a.double().norm()),
                    "sha256_fp32": sha256_fp32(a),
                })
            lp_mism = sum(bit_mismatches(a, b) for a, b in zip(runs[0]["lps"], runs[1]["lps"], strict=True))
            C["grad_repeat"] = {
                "tensors": len(keys),
                "elements": int(sum(p["numel"] for p in per_tensor)),
                "repeat_mismatching_elements": int(sum(p["repeat_mismatches"] for p in per_tensor)),
                "repeat_mismatching_tensors": int(sum(1 for p in per_tensor if p["repeat_mismatches"])),
                "zero_tensors": int(sum(1 for p in per_tensor if p["abs_max"] == 0.0)),
                "logprob_repeat_mismatches": lp_mism,
                "lora_logprobs_hex": [hex_bits(lp) for lp in runs[0]["lps"]],
                "passes": [{k: v for k, v in r.items() if k not in ("lps", "grads")} for r in runs],
                "per_tensor": per_tensor,
                "aggregate_sha256": hashlib.sha256("".join(p["sha256_fp32"] for p in per_tensor).encode()).hexdigest(),
            }
            if args.grads_out:
                Path(args.grads_out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"grads": g1, "lps": runs[0]["lps"], "ids": ids}, args.grads_out)
                C["grad_repeat"]["grads_file"] = args.grads_out
            del runs, g1, g2
            res.flush()
            print(f"[grad_repeat] mismatching elements {C['grad_repeat']['repeat_mismatching_elements']}"
                  f"/{C['grad_repeat']['elements']}", flush=True)

        if "step" in checks:
            trainer.memory(reset=True)
            _, sha0 = trainer.export_lora()
            ta = _sync_time()
            lps = trainer.token_logprobs(seqs, starts)
            tb = _sync_time()
            info = trainer.step(surrogate_loss(lps, weights))
            tc = _sync_time()
            state, sha1 = trainer.export_lora()
            after = trainer.score(seqs, starts)
            changes = [float((a.double() - b.detach().cpu().double()).abs().max()) for a, b in zip(after, lps, strict=True)]
            C["step"] = {
                "info": info, "fwd_s": tb - ta, "step_s": tc - tb, "memory": trainer.memory(reset=True),
                "lora_sha256_before": sha0, "lora_sha256_after": sha1,
                "export_tensors": len(state),
                "export_elements": int(sum(t.numel() for t in state.values())),
                "post_step_max_abs_logp_change_per_sequence": changes,
                "post_step_logprobs_hex": [hex_bits(lp) for lp in after],
            }
            res.flush()
            print(f"[step] {info} sha {sha0[:12]} -> {sha1[:12]}", flush=True)

        if "bench" in checks:
            trainer.reset_lora_b(0.0, 0)
            long_seq = max(seqs, key=len)
            token_pool = (long_seq * (1 + 8192 // len(long_seq)))
            counts = [int(x) for x in args.bench_tokens.split(",") if x.strip()]
            rows = []
            warm = token_pool[: min(counts)]
            trainer.score([warm], [1])
            lp = trainer.token_logprobs([warm], [1])
            trainer.step(-lp[0].mean())
            for n in counts:
                seq = token_pool[:n]
                row: dict[str, Any] = {"tokens": n}
                trainer.memory(reset=True)
                trainer.score([seq], [1])  # warm the M-dependent shape classes
                trainer.memory(reset=True)
                ta = _sync_time()
                trainer.score([seq], [1])
                row["strict_forward_ms"] = (_sync_time() - ta) * 1e3
                row["strict_forward_peak_gib"] = max(m["peak_allocated_gib"] for m in trainer.memory(reset=True))
                ta = _sync_time()
                lp = trainer.token_logprobs([seq], [1])
                tb = _sync_time()
                info = trainer.step(-lp[0].mean())
                tc = _sync_time()
                row["strict_train_logprobs_ms"] = (tb - ta) * 1e3
                row["strict_recompute_fwd_bwd_ms"] = info["fwd_bwd_s"] * 1e3
                row["strict_optimizer_ms"] = info["optimizer_s"] * 1e3
                row["strict_step_total_ms"] = (tc - ta) * 1e3
                row["strict_recompute_mismatches"] = info["recompute_logp_mismatches"]
                row["strict_train_peak_gib"] = max(m["peak_allocated_gib"] for m in trainer.memory(reset=True))
                trainer.native_logprobs([seq], [1])
                trainer.memory(reset=True)
                ta = _sync_time()
                trainer.native_logprobs([seq], [1])
                row["native_forward_ms"] = (_sync_time() - ta) * 1e3
                row["native_forward_peak_gib"] = max(m["peak_allocated_gib"] for m in trainer.memory(reset=True))
                try:
                    nb = trainer.native_fwd_bwd(seq)
                    row["native_fwd_bwd_forward_ms"] = nb["forward_s"] * 1e3
                    row["native_fwd_bwd_backward_ms"] = nb["backward_s"] * 1e3
                    row["native_fwd_bwd_peak_gib"] = max(m["peak_allocated_gib"] for m in trainer.memory(reset=True))
                except Exception as exc:  # noqa: BLE001 - record and continue
                    row["native_fwd_bwd_error"] = repr(exc)[:500]
                    torch.cuda.empty_cache()
                rows.append(row)
                print(f"[bench] {row}", flush=True)
                C["bench"] = {"rows": rows, "note": (
                    "strict_step_total = no-grad strict forward (token_logprobs) + recompute "
                    "forward/backward with per-layer activation recompute + AdamW; native_fwd_bwd = "
                    "Bridge forward + backward of activation grads through all layers with full "
                    "per-layer recompute (frozen base weights, no LoRA)")}
                res.flush()
        res.data["status"] = "complete"
    except BaseException as exc:
        res.data["status"] = "failed"
        res.data["error"] = repr(exc)[:2000]
        raise
    finally:
        if res.data["status"] == "complete":
            res.data["provenance_end"] = trainer.provenance()
        res.flush()
        trainer.close()
    print(f"[run] wrote {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

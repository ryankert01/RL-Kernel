from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
bench_new = Path(sys.argv[2]).read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('''    grad_repeat  B ~ N(0, std): two recompute backward passes of a fixed surrogate loss,
                 bitwise LoRA-grad comparison; grads saved to --grads-out for the TP diff
    step         one AdamW step on that loss; post-step LoRA hash and logprob change
    bench        strict vs native forward / forward+backward(+step) time and peak memory at
                 several token counts

Example (TP1; TP2 is the same with ``-n 2 --tp 2``)::

    ~/bin/gpu-lease -n 1 --pool 2-3 -- env PYTHONPATH=$PWD ~/envs/megatron/bin/python -m \\\\
        rl_engine.models.muse_glimmer.runtime.run_megatron_strict --tp 1 \\\\
        --checks identity,xrt,repeat,lora_noop,native,grad_repeat,step,bench \\\\
        --results out/tp1.json --grads-out out/grads_tp1.pt
"""''', '''    grad_repeat  B ~ N(0, std): two recompute backward passes of a fixed surrogate loss,
                 bitwise LoRA-grad comparison; grads saved to --grads-out for the TP diff
    packing      the same backward at micro_batch_tokens 512 / default / 8192: LoRA grads and
                 logprobs must be bitwise identical (packing-invariant gradient rule)
    step         one AdamW step on that loss; post-step LoRA hash and logprob change
    bench        strict vs native forward / forward+backward(+step) time and peak memory at
                 several token counts; every phase warmed up at each size, then repeated
                 (raw samples + median)

Example (TP1; TP2 is the same with ``-n 2 --tp 2``)::

    ~/bin/gpu-lease -n 1 --pool 0-2 -- env PYTHONPATH=$PWD ~/envs/megatron/bin/python -m \\\\
        rl_engine.models.muse_glimmer.runtime.run_megatron_strict --tp 1 \\\\
        --checks identity,xrt,repeat,lora_noop,native,grad_repeat,packing,step,bench \\\\
        --results out/tp1.json --grads-out out/grads_tp1.pt
"""''')

rep('''import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any''', '''import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any''')

rep('''def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"
''', '''def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty() -> bool | None:
    """True when tracked files differ from HEAD (the measured code is not exactly the commit)."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO), "status", "--porcelain", "--untracked-files=no"], text=True
        )
    except Exception:  # noqa: BLE001
        return None
    return bool(out.strip())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
''')

rep('''    ap.add_argument(
        "--checks", default="identity,xrt,repeat,lora_noop,native,grad_repeat,step,bench",
        help="comma list (see module docstring)",
    )''', '''    ap.add_argument(
        "--checks", default="identity,xrt,repeat,lora_noop,native,grad_repeat,packing,step,bench",
        help="comma list (see module docstring)",
    )''')

rep('''    ap.add_argument("--micro-batch-tokens", type=int, default=4096)
    ap.add_argument("--bench-tokens", default="512,1024,2048,4096")
    ap.add_argument("--no-checkpoint", action="store_true", help="disable per-layer recompute")
    args = ap.parse_args(argv)
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    known = {"identity", "xrt", "repeat", "lora_noop", "native", "grad_repeat", "step", "bench"}''',
    '''    ap.add_argument("--micro-batch-tokens", type=int, default=DEFAULT_MICRO_BATCH_TOKENS)
    ap.add_argument("--packing-budgets", default="512,8192",
                    help="extra micro_batch_tokens values compared by the packing check")
    ap.add_argument("--bench-tokens", default="512,1024,2048,4096,8192")
    ap.add_argument("--bench-warmup", type=int, default=1)
    ap.add_argument("--bench-repeats", type=int, default=3)
    ap.add_argument("--no-checkpoint", action="store_true", help="disable per-layer recompute")
    args = ap.parse_args(argv)
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    known = {"identity", "xrt", "repeat", "lora_noop", "native", "grad_repeat", "packing", "step",
             "bench"}''')

rep('''from rl_engine.models.muse_glimmer.runtime.strict_trainer import StrictTrainer''',
    '''from rl_engine.models.muse_glimmer.runtime.strict_trainer import (
    DEFAULT_MICRO_BATCH_TOKENS,
    StrictTrainer,
)''')

rep('''            "status": "running",
            "commit": _git_sha(),''', '''            "status": "running",
            "commit": _git_sha(),
            "dirty": _git_dirty(),''')

rep('''            if "xrt" in checks:
                out = Path(args.xrt_out_dir) / f"xrt_megatron_strict_tp{tp}.json"
                write_xrt(out, xrt, base, mode="strict", tp=tp, backend=strict_backend,
                          extra={"commit": res.data["commit"], "ops": res.data["provenance"]["ops"],
                                 "lora": "rank 16, B = 0 (exact no-op)"})
                C["xrt_strict"]["file"] = str(out)''', '''            if "xrt" in checks:
                out = Path(args.xrt_out_dir) / f"xrt_megatron_strict_tp{tp}.json"
                write_xrt(out, xrt, base, mode="strict", tp=tp, backend=strict_backend,
                          extra={"commit": res.data["commit"], "dirty": res.data["dirty"],
                                 "ops": res.data["provenance"]["ops"],
                                 "attention_heads_per_rank":
                                     res.data["provenance"].get("attention_heads_per_rank"),
                                 "lora": "rank 16, B = 0 (exact no-op)"})
                C["xrt_strict"]["file"] = str(out)
                C["xrt_strict"]["file_sha256"] = file_sha256(out)''')

rep('''            write_xrt(out, xrt, nat, mode="native", tp=tp, backend="megatron_bridge_te_native",
                      extra={"commit": res.data["commit"],
                             "note": "Bridge MuseGlimmerModel forward (TE kernels, bf16 logits, "
                                     "fp32 log_softmax), no LoRA"})''', '''            write_xrt(out, xrt, nat, mode="native", tp=tp, backend="megatron_bridge_te_native",
                      extra={"commit": res.data["commit"], "dirty": res.data["dirty"],
                             "note": "Bridge MuseGlimmerModel forward (TE kernels, bf16 logits, "
                                     "fp32 log_softmax), no LoRA"})''')
rep('''            C["native"] = {
                "seconds": tb - ta,
                "memory": trainer.memory(reset=True),
                "file": str(out),''', '''            C["native"] = {
                "seconds": tb - ta,
                "memory": trainer.memory(reset=True),
                "file": str(out),
                "file_sha256": file_sha256(out),''')

rep('''        weights = surrogate_weights(spans)
        if "grad_repeat" in checks or "step" in checks:''', '''        weights = surrogate_weights(spans)
        if {"grad_repeat", "packing", "step"} & set(checks):''')

rep('''        if "step" in checks:
            trainer.memory(reset=True)''', '''        if "packing" in checks:
            budgets = [args.micro_batch_tokens] + [
                int(b) for b in args.packing_budgets.split(",") if b.strip()
            ]
            runs = []
            for budget in budgets:
                trainer.set_micro_batch_tokens(budget)
                trainer.memory(reset=True)
                lps = trainer.token_logprobs(seqs, starts)
                info = trainer.backward(surrogate_loss(lps, weights))
                g = trainer.lora_grads()["grads"]
                trainer.zero_grad()
                runs.append({"budget": budget, "lps": [lp.detach().cpu() for lp in lps], "grads": g,
                             "info": info, "memory": trainer.memory(reset=True)})
                print(f"[packing] budget {budget}: {info}", flush=True)
            trainer.set_micro_batch_tokens(args.micro_batch_tokens)
            ref = runs[0]
            per_budget = []
            for r in runs:
                per_budget.append({
                    "micro_batch_tokens": r["budget"],
                    "micro_batches": r["info"]["micro_batches"],
                    "recompute_logp_mismatches": r["info"]["recompute_logp_mismatches"],
                    "grad_bit_mismatching_elements_vs_default": int(sum(
                        bit_mismatches(r["grads"][k], ref["grads"][k]) for k in ref["grads"]
                    )),
                    "logprob_bit_mismatches_vs_default": int(sum(
                        bit_mismatches(a, b) for a, b in zip(r["lps"], ref["lps"], strict=True)
                    )),
                    "aggregate_sha256": hashlib.sha256("".join(
                        sha256_fp32(r["grads"][k]) for k in sorted(r["grads"])
                    ).encode()).hexdigest(),
                    "memory": r["memory"],
                })
            C["packing"] = {
                "elements": int(sum(t.numel() for t in ref["grads"].values())),
                "per_budget": per_budget,
                "bitwise_invariant": all(
                    b["grad_bit_mismatching_elements_vs_default"] == 0
                    and b["logprob_bit_mismatches_vs_default"] == 0 for b in per_budget
                ),
            }
            del runs, ref
            res.flush()
            print(f"[packing] bitwise invariant: {C['packing']['bitwise_invariant']}", flush=True)

        if "step" in checks:
            trainer.memory(reset=True)''')

start = s.index('''        if "bench" in checks:''')
end = s.index('''        res.data["status"] = "complete"''')
s = s[:start] + bench_new + s[end:]
p.write_text(s)
print("ok")

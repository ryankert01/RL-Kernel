"""Print the bench rows, vision drop, packing and step info of run_tp*.json."""
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    print("=====", path.split("/")[-1], d.get("status"), d.get("commit", "")[:8], "dirty", d.get("dirty"))
    c = d.get("checks", d)
    prov = d.get("provenance") or d.get("provenance_end") or {}
    print("vision_drop:", json.dumps(prov.get("vision_drop")), "mem:", json.dumps(prov.get("memory_gib")))
    print("micro_batch_tokens", prov.get("micro_batch_tokens"), "ops all_strict",
          (prov.get("ops") or {}).get("all_strict"), "allow_fallback", prov.get("allow_fallback"))
    if "packing" in c:
        for b in c["packing"]["per_budget"]:
            print("  packing", b["micro_batch_tokens"], b["micro_batches"], b["grad_bit_mismatching_elements_vs_default"],
                  b["logprob_bit_mismatches_vs_default"], b["recompute_logp_mismatches"], b["aggregate_sha256"][:12])
    if "step" in c:
        s = c["step"]
        print("  step grad_norm", s["info"].get("grad_norm"), "sha", s["lora_sha256_after"][:12],
              "mism", s["info"].get("recompute_logp_mismatches"))
    if "bench" in c:
        for r in c["bench"]["rows"]:
            med = r["median"]
            print("  bench", r["tokens"], "n", {k: len(v) for k, v in r["samples"].items()}.get("strict_step_total_ms"),
                  "strict_step_ms", round(med.get("strict_step_total_ms", -1)),
                  "fwd_bwd", round(med.get("strict_recompute_fwd_bwd_ms", -1)),
                  "native_fb", round(med.get("native_fwd_bwd_forward_ms", -1) + med.get("native_fwd_bwd_backward_ms", 0)),
                  "peak", {k: round(v, 1) for k, v in r["peak_gib"].items()}, "errors", list(r["errors"]))
    print("top keys", list(d.keys()))
    print("check keys", list(c.keys()) if isinstance(c, dict) else None)

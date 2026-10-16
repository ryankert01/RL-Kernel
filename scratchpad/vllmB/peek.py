"""Print a compact summary of run_vllm_strict run files (usage: peek.py <run.json>...)."""

import json
import sys


def chk(c):
    if not isinstance(c, dict):
        return c
    return (
        f"{c.get('mismatched')}/{c.get('tokens')} (nonzero {c.get('nonzero_tokens')}, "
        f"<-1e-3 {c.get('tokens_lp_below_1e-3')}, >0 {c.get('positive_tokens')})"
    )


for path in sys.argv[1:]:
    try:
        r = json.load(open(path))
    except Exception as exc:  # noqa: BLE001
        print(path, "unreadable", exc)
        continue
    print(f"== {path}: {r.get('mode')} tp{r.get('tp')} status={r.get('status')} "
          f"commit={str(r.get('commit'))[:8]} dirty={r.get('git_dirty')} err={r.get('error')}")
    for g in r.get("generate", []):
        print(f"  gen T={g['temperature']}: dvp {chk(g.get('decode_vs_prefill'))} tok/s {g['tokens_per_s']:.1f}")
    if r.get("lora_zero"):
        z = r["lora_zero"]
        print(f"  lora_zero identity={z.get('hash_identity')} vs_no_lora {chk(z.get('vs_no_lora'))}")
    if r.get("lora_random"):
        lr = r["lora_random"]
        print(f"  lora_random identity={lr.get('hash_identity')} workers={lr.get('worker_hashes')}"
              f" changed {chk(lr.get('scores_vs_no_lora'))}")
        for g in lr.get("generate", []):
            print(f"    lora gen T={g['temperature']}: dvp {chk(g.get('decode_vs_prefill'))}")
        for u in lr.get("unmerged_rule", []) or []:
            print(f"    rule layer {u['layer']}: all_equal={u['all_equal']}")
    for c in r.get("contract_shapes", []):
        st = c["engine_steps"]
        print(f"  {c['label']}: req {c['requests']} dvp {chk(c['decode_vs_prefill'])} "
              f"maxbatch {st['max_decode_batch']} steps {st['steps']} prefill-steps "
              f"{st['steps_with_prefill_tokens']} pos {c['decode_positions_min_max']}")
    if r.get("xrt"):
        print(f"  xrt -> {r['xrt'].get('out')} {r['xrt'].get('score_ms'):.0f} ms")
    for key in ("bench", "bench_lora"):
        b = r.get(key)
        if b:
            print(f"  {key}: tok/s {b['tokens_per_s']:.1f} repeats {[round(x,1) for x in b['tokens_per_s_repeats']]}"
                  f" spread {b['tokens_per_s_spread_pct']:.1f}% ttft {b['ttft_ms']:.0f} decode med "
                  f"{b['decode_step_ms_median']} mem {[round(m['max_allocated_gib'],1) for m in b['memory_per_rank']]}")
    meta = r.get("meta") or {}
    if meta:
        print(f"  meta lora_sha256={meta.get('lora_sha256')} src={meta.get('lora_sha256_source')}")

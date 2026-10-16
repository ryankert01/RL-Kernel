import json
import sys

new = json.load(open(sys.argv[1]))
old = json.load(open(sys.argv[2]))
print(new["env"], new["modules"]["backward_kernel_id"], old["modules"]["backward_kernel_id"])
print("prefill forward")
for r in new["prefill_forward"]:
    print(f"  T={r['T']:5d} win={r['window']} strict={r['strict_ms']:.3f} sdpa={r['sdpa_ms']:.3f} "
          f"ratio={r['ratio']:.2f}")
print("decode forward")
for r in new["decode_forward"]:
    print(f"  B={r['B']:4d} ctx={r['ctx']} win={r['window']} strict={r['strict_ms']:.3f} "
          f"sdpa={r['sdpa_ms']:.3f} ratio={r['ratio']:.2f}")
print("backward before/after")
ob = {(r["T"], r["window"]): r for r in old["backward"]}
for r in new["backward"]:
    o = ob[(r["T"], r["window"])]
    print(f"  T={r['T']:5d} win={r['window']} v1={o['strict_ms']:.3f} v2={r['strict_ms']:.3f} "
          f"x{r['strict_ms'] / o['strict_ms']:.2f} sdpa={r['sdpa_causal_bwd_ms']:.3f} "
          f"ratio v1={o['ratio']:.2f} v2={r['ratio']:.2f} peakGiB v1={o['transient_peak_GiB']:.3f} "
          f"v2={r['transient_peak_GiB']:.3f}")
print("tp shards")
for key in ("tp_shard_forward", "tp_shard_backward"):
    for r in new[key]:
        print(f"  {key} tp={r['tp']} H={r['H_Q']}/{r['H_KV']} win={r['window']} "
              f"strict={r['strict_ms']:.3f}" + (f" ratio={r['ratio']:.2f}" if "ratio" in r else ""))

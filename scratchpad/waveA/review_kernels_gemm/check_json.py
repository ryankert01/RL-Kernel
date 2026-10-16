import json

d = json.load(open("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-11/reports/muse_glimmer_poc/data/kernels_gemm.json"))
bl = d["bit_ledger"]
rows = bl["rows"]
print("summary", bl["summary"])
print("rows", len(rows), "changed", sum(r["changed"] for r in rows),
      "fwd rows", sum(r["path"].startswith("fwd") for r in rows),
      "fwd changed", sum(r["changed"] for r in rows if r["path"].startswith("fwd")))
prefixes = {}
for r in rows:
    p = r["path"].split("/")[0]
    prefixes.setdefault(p, [0, 0])
    prefixes[p][0] += 1
    prefixes[p][1] += r["changed"]
print("by prefix (n, changed):", prefixes)
print("changed paths:")
for r in rows:
    if r["changed"]:
        print("  ", r["path"], r["before"], r["after"])
print("unchanged non-fwd sample:")
for r in rows:
    if not r["changed"] and not r["path"].startswith("fwd"):
        print("  ", r["path"])
print()
print("headline:")
for h in d["headline"]:
    print(" -", h)
print("suggested_figures:")
for f in d["suggested_figures"]:
    print(" -", json.dumps(f)[:400])
it = d["det_gemm_1_iters_probe"]
print("before backward_paths:")
for r in it["before"]["backward_paths"]:
    print("  ", r)
print("before forward_sweep (first 6):")
for r in it["before"]["forward_sweep"][:6]:
    print("  ", r)
print("ir before:", it["before"]["ir"])
print("ir after:", it["after"]["ir"])
print("benchmarks lm_head:")
for r in d["benchmarks"]["lm_head"]:
    print("  ", r)
print("benchmarks ffn:")
for r in d["benchmarks"]["ffn"]:
    print("  ", r)

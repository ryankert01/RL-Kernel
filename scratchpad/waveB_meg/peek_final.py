import json
import sys

d = json.load(open(sys.argv[1]))
print(list(d))
print("status:", d["status"])
print("commits:", d["commits"])
print("runs:", {k: (v["status"], v["commit"][:8], v.get("dirty")) for k, v in d["runs"].items()})
print("xrt_files:", {k: (v["commit"][:8] if v["commit"] else None, v["sha256"][:12], v["postdates_o7_fix"])
                     for k, v in d["xrt_files"].items()})
print("cross_tp:", list(d["cross_tp"]))
print("bench rows tp1:", [r["tokens"] for r in d["runs"]["tp1"]["checks"]["bench"]["rows"]])
print("tests:", [t["outcome"] for t in d["tests"]["unit_tests"]])

import json
import sys

d = json.load(open(sys.argv[1]))
print(list(d.keys()))
for p in d["parts"]:
    print({k: p[k] for k in ("part", "path", "commit", "worktree_dirty", "load_s", "weights_gib",
                             "gpu_lease", "status")})
print(d["commands"])
for s in d["settings"]:
    print(s["engine"], s["setting"], s["n_bits_differ"], s.get("wall_ms"), s.get("peak_mem_gib"))
print(d["records"][0].keys())
print(d["baselines"]["hf"].keys(), d["baselines"]["hf"]["per_sequence"][0].keys())
print(d["analysis"].keys())
print(json.dumps(d["tf32_probe"]))
print(d["layer_records"][0].keys(), len(d["layer_records"]))
print(d["env"].keys())

"""Fill the commit field of a run file produced from `git archive <sha>` (no .git there, so the
run's own git lookup returned an empty sha)."""
import json
import sys

path, sha = sys.argv[1], sys.argv[2]
d = json.load(open(path))
c = d.get("commit") or {}
assert not c.get("sha"), f"{path} already has a commit: {c}"
d["commit"] = {"sha": sha, "dirty": False}
d["commit_note"] = (
    f"run from `git archive {sha[:7]}` (an exact, clean copy of that commit, in the session "
    "scratchpad) so that the worktree could be edited during the ~25 min golden runs; the run's "
    "own git lookup found no repository (sha ''), the sha was filled in from the archive")
json.dump(d, open(path, "w"), indent=1 if path.endswith("golden.json") else 2)
print("filled", path)

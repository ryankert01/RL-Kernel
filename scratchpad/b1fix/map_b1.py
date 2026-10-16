import difflib, json, re, sys
SP = "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad"
TF17 = "/home/ubuntu/envs/vllm/lib/python3.12/site-packages/transformers"
files = {
    "modeling": (f"{SP}/hf/modeling_v5.19.0.py", f"{SP}/hf/modeling_v5.17.0.py"),
    "masking_utils.py": (f"{SP}/hf/masking_utils.py", f"{TF17}/masking_utils.py"),
    "cache_utils.py": (f"{SP}/hf/cache_utils.py", f"{TF17}/cache_utils.py"),
}
src = {k: (open(a).read().splitlines(), open(b).read().splitlines()) for k, (a, b) in files.items()}
def mapper(k):
    a, b = src[k]
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                m[i1 + d + 1] = (j1 + d + 1, True)
        else:
            for d in range(i2 - i1):
                m[i1 + d + 1] = (min(j1 + d, max(j2 - 1, j1)) + 1, False)
    return m
maps = {k: mapper(k) for k in src}
audit = json.load(open("/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7/reports/muse_glimmer_poc/data/rfc_audit.json"))
b1 = [i for i in audit["items"] if i["id"] == "B1"][0]["corrected"]
# file context: a backticked file name right before a run of L refs applies until ';' or ')'
ctx_re = re.compile(r"`(masking_utils\.py|cache_utils\.py|config\.json)`\s+((?:L\d+(?:[–-]\d+)?(?:,\s*)?)+)")
spans = {}
for m in ctx_re.finditer(b1):
    for r in re.finditer(r"L\d+(?:[–-]\d+)?", m.group(2)):
        spans[m.start(2) + r.start()] = m.group(1)
# unqualified refs that sit in the cache paragraph (they describe DynamicSlidingWindowLayer / StaticCache)
cache_para = b1.index("**Local-layer window and eviction.**")
cache_end = b1.index("**Projection grouping.**")
report = []
def sub(mo):
    pos = mo.start()
    f = spans.get(pos)
    a, b = int(mo.group(1)), int(mo.group(3) or mo.group(1))
    if f is None and cache_para < pos < cache_end and (a in (254, 531, 579, 610)):
        f = "cache_utils.py"
    if f == "config.json":
        return mo.group(0)
    k = f or "modeling"
    na, oka = maps[k][a]; nb, okb = maps[k][b]
    old_a, new_a = src[k][0][a - 1].strip(), src[k][1][na - 1].strip()
    report.append((k, f"{a}-{b}", f"{na}-{nb}", oka and okb, old_a[:60], new_a[:60]))
    return f"L{na}" + (f"{mo.group(2)}{nb}" if mo.group(3) else "")
out = re.sub(r"L(\d+)(?:([–-])(\d+))?", sub, b1)
for r in report:
    print(("OK  " if r[3] else "CHK ") + " | ".join(map(str, r[:3])) + "\n     19: " + r[4] + "\n     17: " + r[5])
open("b1_renumbered.txt", "w").write(out)

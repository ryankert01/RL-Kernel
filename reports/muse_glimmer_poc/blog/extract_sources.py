# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Turn the text sources of the first POC session into JSON for the blog.

* bench_ws1.log          -> data/ws1_bench.json   (every markdown table, keyed by benchmark script)
* RFC_AUDIT_DIGEST.md    -> data/rfc_audit.json   (one record per verified RFC edit)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

POC = Path(__file__).resolve().parents[1]
DATA = POC / "data"


def parse_tables(text: str) -> list[dict]:
    tables, cur, heading = [], None, ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading = s.lstrip("# ").strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cur is None:
                cur = {"heading": heading, "columns": cells, "rows": []}
            elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            else:
                cur["rows"].append(dict(zip(cur["columns"], cells)))
        elif cur is not None:
            tables.append(cur)
            cur = None
    if cur is not None:
        tables.append(cur)
    return tables


def bench() -> dict:
    text = (POC / "bench_ws1.log").read_text()
    out = {"source": "reports/muse_glimmer_poc/bench_ws1.log", "gpu": "NVIDIA A100-SXM4-80GB",
           "torch": "2.13.0+cu130", "triton": "3.7.1", "benchmarks": {}}
    for block in text.split("===== ")[1:]:
        name, _, body = block.partition("\n")
        out["benchmarks"][name.strip()] = {"tables": parse_tables(body), "raw": body.strip()}
    return out


def audit() -> dict:
    text = (POC / "RFC_AUDIT_DIGEST.md").read_text()
    items, section = [], ""
    for chunk in re.split(r"\n(?=## |### )", text):
        if chunk.startswith("## "):
            section = chunk[3:].splitlines()[0].strip()
            continue
        m = re.match(r"### (\w+) \[(\w+)\] — verdict: (\w+)", chunk)
        if not m:
            continue
        rec = {"id": m.group(1), "kind": m.group(2), "verdict": m.group(3), "section": section}
        for key in ("RFC", "Problem", "Proposed", "Evidence", "CORRECTED"):
            mm = re.search(rf"^{key}: (.*?)(?=\n\n(?:RFC|Problem|Proposed|Evidence|CORRECTED):|\n\n###|\Z)", chunk, re.S | re.M)
            if mm:
                rec[key.lower()] = mm.group(1).strip()
        items.append(rec)
    return {"source": "reports/muse_glimmer_poc/RFC_AUDIT_DIGEST.md", "items": items}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    b = bench()
    (DATA / "ws1_bench.json").write_text(json.dumps(b, indent=1, ensure_ascii=False) + "\n")
    a = audit()
    (DATA / "rfc_audit.json").write_text(json.dumps(a, indent=1, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for it in a["items"]:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print("ws1 benchmarks:", list(b["benchmarks"]), "| audit items:", len(a["items"]), kinds)


if __name__ == "__main__":
    main()

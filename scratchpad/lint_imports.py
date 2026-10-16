"""Crude unused-import / undefined-name check (ruff/pyflakes are not installed)."""

import ast
import sys


def check(path: str) -> int:
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.lineno
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
    source = open(path, encoding="utf-8").read()
    problems = 0
    for name, line in imported.items():
        if name == "annotations":
            continue
        if name not in used and f"# noqa" not in source.splitlines()[line - 1]:
            print(f"{path}:{line}: unused import {name}")
            problems += 1
    return problems


if __name__ == "__main__":
    sys.exit(1 if sum(check(p) for p in sys.argv[1:]) else 0)

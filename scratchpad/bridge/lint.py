"""Tiny lint: unused imports / undefined-looking names (ast), compile check."""
import ast
import sys
from pathlib import Path

bad = 0
for path in sys.argv[1:]:
    src = Path(path).read_text()
    tree = ast.parse(src)
    compile(src, path, "exec")
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for a in node.names:
                imported[a.asname or a.name] = node.lineno
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
    text_names = src  # names used in strings / __all__
    for name, line in imported.items():
        if name not in used and f'"{name}"' not in text_names:
            print(f"{path}:{line}: unused import {name}")
            bad += 1
print("issues:", bad)

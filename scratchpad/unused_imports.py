"""Crude pyflakes substitute: report imported names never referenced in the module."""
import ast
import sys

for path in sys.argv[1:]:
    tree = ast.parse(open(path).read())
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported[a.asname or a.name] = node.lineno
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    # names re-exported via __all__ count as used
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            for elt in getattr(node.value, "elts", []):
                if isinstance(elt, ast.Constant):
                    used.add(elt.value)
    for name, line in sorted(imported.items(), key=lambda kv: kv[1]):
        if name not in used and name != "annotations":
            print(f"{path}:{line}: unused import {name}")
print("unused-import check done")

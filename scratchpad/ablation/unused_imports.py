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
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    src = open(path).read()
    for name, line in imported.items():
        if name not in used and f'"{name}"' not in src:
            print(f"{path}:{line}: unused import {name}")

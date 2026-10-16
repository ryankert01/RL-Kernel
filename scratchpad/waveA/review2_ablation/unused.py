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
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    src = open(path).read()
    for name, line in imported.items():
        if name not in names and name not in attrs and src.count(name) < 2:
            print(f"{path}:{line}: possibly unused {name}")
        elif name not in names and name not in attrs:
            # may be in __all__ strings only
            print(f"{path}:{line}: not referenced as a name: {name}")

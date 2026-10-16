"""Runner: python run_probe.py <module_file> <probe names...>."""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("probe_mod", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sys.argv[2:]:
    mod.run(name, getattr(mod, name))

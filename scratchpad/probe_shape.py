import sys, pytest
sys.argv=["x"]
from rl_engine.models.muse_glimmer import workload as w
from rl_engine.models.muse_glimmer.loader import verify_checkpoint
import importlib.util, pathlib, tempfile
spec=importlib.util.spec_from_file_location("t","tests/models/muse_glimmer/test_museglimmer_fingerprint.py"); t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
d=pathlib.Path(tempfile.mkdtemp())
fc=t.fake_checkpoint.__wrapped__(d) if hasattr(t.fake_checkpoint,'__wrapped__') else None

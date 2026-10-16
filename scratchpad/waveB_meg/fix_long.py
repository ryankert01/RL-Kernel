from pathlib import Path

R = Path("rl_engine/models/muse_glimmer/runtime")
edits = {
    R / "strict_kernels.py": [
        ('''            source=OpSource("gated_attention", False, fallback.source.detail + f"; missing {missing}"),''',
         '''            source=OpSource(
                "gated_attention", False, fallback.source.detail + f"; missing {missing}"
            ),'''),
        ('''        return [f"{name} -> {src.label}" for name, src in self.all_sources().items() if not src.strict]''',
         '''        return [
            f"{name} -> {src.label}" for name, src in self.all_sources().items() if not src.strict
        ]'''),
    ],
    R / "strict_trainer.py": [
        ('''            out["recompute_check"] = "bitwise; a mismatch raises StrictRecomputeMismatch (fail closed)"''',
         '''            out["recompute_check"] = (
                "bitwise; a mismatch raises StrictRecomputeMismatch (fail closed)"
            )'''),
    ],
    Path("tests/models/muse_glimmer/test_megatron_strict_binding.py"): [
        ('''                torch.nn.Linear(1024, 1024, device="cuda"), torch.nn.Linear(1024, 1024, device="cuda")''',
         '''                torch.nn.Linear(1024, 1024, device="cuda"),
                torch.nn.Linear(1024, 1024, device="cuda"),'''),
    ],
}
for path, pairs in edits.items():
    s = path.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60])
        s = s.replace(old, new)
    path.write_text(s)
print("ok")

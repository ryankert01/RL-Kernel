from pathlib import Path

root = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime")
fixes = {
    "vllm_strict.py": [
        ('''            "attention_heads_per_rank": {"num_q_heads": heads_per_rank, "num_kv_heads": kv_per_rank},''',
         '''            "attention_heads_per_rank": {
                "num_q_heads": heads_per_rank,
                "num_kv_heads": kv_per_rank,
            },'''),
        ('''        state.ops.det_gemm(hidden, weight[i * TILE_ROWS : (i + 1) * TILE_ROWS], out_dtype=torch.float32)''',
         '''        state.ops.det_gemm(
            hidden, weight[i * TILE_ROWS : (i + 1) * TILE_ROWS], out_dtype=torch.float32
        )'''),
    ],
    "strict_rollout_engine.py": [
        ('''            raise RuntimeError(f"{len(files)} workers answered, expected {self.tensor_parallel_size}")''',
         '''            raise RuntimeError(
                f"{len(files)} workers answered, expected {self.tensor_parallel_size}"
            )'''),
        ('''        reports = list(self.llm.collective_rpc(worker_native_lora_report, args=(request.lora_int_id,)))''',
         '''        reports = list(
            self.llm.collective_rpc(worker_native_lora_report, args=(request.lora_int_id,))
        )'''),
    ],
}
for name, pairs in fixes.items():
    p = root / name
    s = p.read_text()
    for old, new in pairs:
        assert old in s, (name, old)
        s = s.replace(old, new, 1)
    p.write_text(s)
print("ok")

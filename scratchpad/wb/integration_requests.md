### Wave B: integration requests (other owners' files; exact patches)

Line numbers are for f91e8b6; the bindings are being edited by other agents, so match by context.

1. **Fallback O4 functions accept `local=` and the fallback table builders stop building
   private tables** (`runtime/fallback_ops.py`, `runtime/strict_kernels.py`; kernels_rope:R1/R2).
   ```python
   # runtime/fallback_ops.py: replace the body of rope_table (line ~278)
   def rope_table(positions, *, theta, dim, device):
       from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows
       return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)

   # runtime/fallback_ops.py::qknorm_scaled_rope (line ~305) and
   # runtime/strict_kernels.py::_fb_qknorm_scaled_rope (line ~345): add the keyword and check
       *, eps: float, qk_scale: float, local: bool | None = None,
   ) -> tuple[torch.Tensor, torch.Tensor]:
       if local is not None and bool(local) != (cos is not None):
           raise ValueError(f"local={local} but cos/sin are {'None' if cos is None else 'given'}")

   # runtime/strict_kernels.py::_fb_rope_table (line ~331): same delegation as fallback_ops
   def _fb_rope_table(positions, *, theta, dim, device):
       from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows
       return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)
   ```
   Both builders are bitwise identical to the canonical table on CUDA today
   (`test_other_repo_table_builders_are_bitwise_identical_on_cuda`), so no bit changes; a CPU
   device then raises instead of building a table that differs in ~19 % of its elements.
2. **Every call site states the layer kind** (kernels_rope:R1), after item 1:
   - `runtime/vllm_strict.py` (attention forward, `ops.qknorm_scaled_rope(q_heads, k_heads, cos,
     sin, eps=..., qk_scale=...)`, line ~943): add `local=bool(self.use_rope),`
   - `runtime/megatron_strict.py::_layer` (line ~556/559): sliding branch `local=True`, global
     branch `local=False`
   - `runtime/strict_ops.py::probe_rope` (line ~236/238): `local=True` / `local=False`
   - `runtime/vllm_tp_sim.py` (line ~158): `local=True`
   - `ablation/ablation_strict.py` (line ~134): `local=layer.is_local`
   - then the E2E harness exports `MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1` to both runtimes
     (rollout workers and trainer ranks): any remaining implicit call raises instead of
     silently running NoPE on a sliding layer. `qknorm_rope.runtime_provenance()
     ["layer_kind_calls"]["implicit_local"/"implicit_global"]` must be 0 in a run log.
3. **Table identity at runtime** (kernels_rope:R2/R4):
   - `runtime/vllm_strict.py`: where `state.cfg["rope_table_positions"]` is set (worker init,
     line ~708, i.e. before warm-up and CUDA-graph capture), add
     ```python
     from rl_engine.models.muse_glimmer.ops import qknorm_rope as _o4
     if state.ops.is_strict("rope_table"):
         dev = torch.device("cuda", torch.cuda.current_device())
         _o4.canonical_rope_table(dev, verify=True)   # builds once, raises on a digest mismatch
         state.cfg["rope_table_sha256"] = _o4.rope_table_sha256(dev)
     ```
     `_rope_for` may keep `state.ops.rope_table(arange(max_pos))`: it now returns copies of the
     canonical rows (same bits).
   - `runtime/megatron_strict.py` (trainer init, before the first forward): the same two calls
     with `self.device`, and record `rope_table_sha256` in the provenance.
   - E2E orchestrator: compare the two runtimes' `rope_table_sha256` (or
     `runtime_provenance()["rope_table_sha256_built"]`) and fail the run if they differ.

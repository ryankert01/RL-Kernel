from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_attention_backend.py")
s = p.read_text()
old = '''  ``(blocks, 16, 2, 128)`` K and V caches, so this backend reinterprets the layer's (contiguous)
  buffer as ``(2, blocks, block_size, kv_heads, head_dim)`` and owns the KV write
  (``do_kv_cache_update`` -> ``reshape_and_cache_flash`` on those views; a pure copy).  Nothing
  else reads the buffer in strict mode (no prefix caching, no KV connector, no cascade).'''
new = '''  ``(blocks, 16, H_kv, 128)`` K and V caches, so this backend reinterprets the layer's
  (contiguous) buffer as ``(2, blocks, block_size, kv_heads, head_dim)`` and owns the KV write
  (``do_kv_cache_update`` -> ``reshape_and_cache_flash`` on those views; a pure copy).
  The strict page ``b`` is therefore NOT the native page ``b``: native page ``b`` is the bytes
  ``[b*N*H*2D, (b+1)*N*H*2D)`` of the layer region, strict K page ``b`` is ``[b*N*H*D,
  (b+1)*N*H*D)`` and strict V page ``b`` sits in the second half.  This is safe only because
  nothing but this backend touches the region: the binding rejects prefix caching (its partial-hit
  copy-on-write copies native page ranges), KV connectors / transfer / offloading, speculative
  decoding, context parallelism, KV sharing and KV-block zeroing at model init
  (``vllm_strict.check_strict_engine_config``) and makes the native page writers raise
  (``copy_kv_cache_blocks_inplace``, ``KVBlockZeroer.zero_block_ids``); cascade attention is off
  in the metadata builder and rejected in :meth:`forward`.  Layers of different KV-cache groups
  that share one tensor use the same strict view and disjoint block ids, so they cannot overlap.'''
assert old in s
s = s.replace(old, new, 1)
old = '''* The strict kernel pins the full head set (32 query heads, 2 KV heads).  At TP > 1 the installer
  replicates KV (every rank holds both KV heads and runs the full attention redundantly after an
  exact all-gather of q/k/v/gate), so this backend always sees 32/2.'''
new = '''* Heads per rank: vLLM builds each rank's ``Attention`` with ``32/t`` query heads and
  ``max(1, 2/t)`` KV heads (TP2 16/1, TP4 8/1 with the KV head replicated on two ranks); this
  backend passes exactly those counts to the strict op (``num_q_heads``/``num_kv_heads``).  Heads
  never interact in O5, so the rank's output is the matching slice of the TP1 output bit for bit
  (``ops/REPORT_gated_attention_fwd.md``, wave A: TP2/4/8 shard parity).'''
assert old in s
s = s.replace(old, new, 1)
old = '''    first half, V in the second.  Only the strict backend reads or writes the cache in strict mode
    (prefix caching, KV connectors and cascade attention are off), and page ``b`` keeps the same
    byte range in both views' coordinates, so sharing a tensor between layers of different KV
    groups (disjoint block ids) stays safe.
    """'''
new = '''    first half, V in the second.  The strict page ``b`` is NOT the native page ``b`` (different
    byte ranges), so this view is only valid while nothing else addresses the region by block id:
    prefix caching, zeroing and KV connectors are rejected at init and the native page writers
    raise (see the module docstring).  Layers of different KV groups that share a tensor use this
    same view with disjoint block ids, so they cannot overlap.
    """'''
assert old in s
s = s.replace(old, new, 1)
old = '''                window=window,
                max_seqlen_q=int(attn_metadata.max_query_len),
            )'''
new = '''                window=window,
                max_seqlen_q=int(attn_metadata.max_query_len),
                num_q_heads=int(self.num_heads),
                num_kv_heads=int(self.num_kv_heads),
            )'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")

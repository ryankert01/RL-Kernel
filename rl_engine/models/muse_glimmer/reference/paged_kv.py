# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Minimal paged KV cache for the Muse Glimmer strict reference (contract O5, rollout form).

Page = 16 tokens, so one 64-key attention block is 4 pages. Each layer owns its own page pool
and block table (local layers evict, global layers do not). Page 0 of every pool is a reserved
all-zero page: block-table entries that were never allocated or were evicted read it, which the
attention masks by position anyway, so eviction timing cannot change bits (contract O5).
Writes are exact copies; nothing here does arithmetic.
"""

from __future__ import annotations

import torch

from rl_engine.models.muse_glimmer.reference.golden_ops import BLOCK_N, KV_PAGE


class PagedKVCache:
    """Per-layer paged K/V storage with block tables for a few sequences."""

    def __init__(
        self,
        *,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        max_seqs: int = 1,
        page_size: int = KV_PAGE,
    ):
        self.num_layers = int(num_layers)
        self.page_size = int(page_size)
        self.max_seqs = int(max_seqs)
        self.max_pages_per_seq = -(-int(max_seq_len) // self.page_size)
        # +1 for the reserved zero page; pool shared by all sequences of this layer.
        self.num_pages = self.max_pages_per_seq * self.max_seqs + 1
        self.device = torch.device(device)
        shape = (self.num_layers, self.num_pages, self.page_size, int(n_kv_heads), int(head_dim))
        self.k = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=dtype, device=self.device)
        self.block_table = torch.full(
            (self.num_layers, self.max_seqs, self.max_pages_per_seq),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.seq_lens = torch.zeros((self.max_seqs,), dtype=torch.long, device=self.device)
        self._free: list[list[int]] = [
            list(range(self.num_pages - 1, 0, -1)) for _ in range(self.num_layers)
        ]
        self._bt_host: list[list[list[int]]] = [
            [[-1] * self.max_pages_per_seq for _ in range(self.max_seqs)]
            for _ in range(self.num_layers)
        ]
        self.evictions = 0

    # -- writes ---------------------------------------------------------------------------
    def append(
        self, layer: int, seq: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> None:
        """Write ``k_new``/``v_new`` ``(n, Hkv, D)`` at absolute positions ``start_pos..``."""
        n = int(k_new.shape[0])
        if n == 0:
            return
        if start_pos + n > self.max_pages_per_seq * self.page_size:
            raise RuntimeError("paged KV overflow: sequence longer than the cache was sized for")
        positions = torch.arange(start_pos, start_pos + n, device=self.device)
        logical = positions // self.page_size
        slots = positions % self.page_size
        first, last = start_pos // self.page_size, (start_pos + n - 1) // self.page_size
        row = self._bt_host[layer][seq]
        changed = False
        for lp in range(first, last + 1):
            if row[lp] < 0:
                if not self._free[layer]:
                    raise RuntimeError(f"paged KV: layer {layer} has no free pages")
                row[lp] = self._free[layer].pop()
                changed = True
        if changed:
            self.block_table[layer, seq] = torch.tensor(row, dtype=torch.int32, device=self.device)
        phys = self.block_table[layer, seq].to(torch.long).index_select(0, logical)
        self.k[layer, phys, slots] = k_new.to(self.k.dtype)
        self.v[layer, phys, slots] = v_new.to(self.v.dtype)

    def advance(self, seq: int, n: int) -> None:
        """Commit ``n`` new positions for ``seq`` (after every layer appended them)."""
        self.seq_lens[seq] += int(n)

    def seq_len(self, seq: int) -> int:
        return int(self.seq_lens[seq].item())

    # -- eviction -------------------------------------------------------------------------
    def evict_window(self, layer: int, seq: int, window: int) -> int:
        """Free pages no future query of ``seq`` can see on a local layer.

        The next query position is ``p = seq_len`` and sees keys ``> p − W``; a page whose
        every position is ``≤ seq_len − W`` is dead. Blocks straddling the boundary stay.
        """
        seq_len = self.seq_len(seq)
        dead_max_pos = seq_len - int(window)  # every position ≤ this is invisible forever
        # page lp covers positions [lp*ps, (lp+1)*ps − 1]; dead iff (lp+1)*ps − 1 ≤ dead_max_pos
        last_dead_page = (dead_max_pos + 1) // self.page_size - 1
        row = self._bt_host[layer][seq]
        freed = 0
        for lp in range(0, min(last_dead_page + 1, len(row))):
            if row[lp] >= 0:
                self._free[layer].append(row[lp])
                row[lp] = -1
                freed += 1
        if freed:
            self.block_table[layer, seq] = torch.tensor(row, dtype=torch.int32, device=self.device)
            self.evictions += freed
        return freed

    # -- reads ----------------------------------------------------------------------------
    def layer_views(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k[layer], self.v[layer], self.block_table[layer]

    def pages_in_use(self, layer: int) -> int:
        return self.num_pages - 1 - len(self._free[layer])

    def identity(self) -> dict[str, object]:
        return {
            "kind": "muse_glimmer_paged_kv",
            "page_size": self.page_size,
            "block_keys": BLOCK_N,
            "layout": "[layer, page, slot, kv_head, head_dim]; page 0 reserved zero page",
            "dtype": str(self.k.dtype).replace("torch.", ""),
        }


__all__ = ["PagedKVCache"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Minimal paged KV cache for the Muse-Glimmer strict POC tests (contract O5, rollout row).

Layout follows the vLLM-style paged cache the repo's FlashInfer path consumed: physical pages
of ``PAGE_SIZE = 16`` tokens, ``k_cache / v_cache`` of shape ``(num_pages, 16, 2, 128)`` bf16,
a per-sequence ``block_table (B, max_pages)`` int32 of physical page ids (``-1`` = not
allocated / evicted) and ``seq_lens (B,)`` int32.

Sliding-window eviction (contract O5): once the current query position of a sequence is
``p``, every page whose keys all sit at positions ``<= p - W`` may be freed. The strict kernel
masks those keys before it reads the block table, so eviction timing cannot change bits; the
tests prove this by poisoning freed pages with NaN and reusing them for other sequences.

This class is a test utility, not a production allocator: it is Python-driven and does one
small device write per append.
"""

from __future__ import annotations

import torch

PAGE_SIZE = 16
NUM_KV_HEADS = 2
HEAD_DIM = 128


class PagedKVCache:
    """Pages of 16 tokens, block tables, append and window eviction."""

    def __init__(
        self,
        *,
        num_pages: int,
        num_seqs: int,
        max_pages_per_seq: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        num_kv_heads: int = NUM_KV_HEADS,
        head_dim: int = HEAD_DIM,
        page_size: int = PAGE_SIZE,
        poison_freed: bool = False,
    ) -> None:
        if min(num_pages, num_seqs, max_pages_per_seq, num_kv_heads, head_dim, page_size) <= 0:
            raise ValueError("PagedKVCache dimensions must be positive")
        self.page_size = int(page_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_seqs = int(num_seqs)
        self.max_pages_per_seq = int(max_pages_per_seq)
        self.device = torch.device(device)
        self.dtype = dtype
        self.poison_freed = bool(poison_freed)
        shape = (int(num_pages), self.page_size, self.num_kv_heads, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.block_table = torch.full(
            (self.num_seqs, self.max_pages_per_seq), -1, dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.zeros((self.num_seqs,), dtype=torch.int32, device=self.device)
        # Host mirrors (the kernel reads the device tensors; Python drives allocation).
        self._lens = [0] * self.num_seqs
        self._table: list[list[int]] = [[-1] * self.max_pages_per_seq for _ in range(num_seqs)]
        self._first_live = [0] * self.num_seqs  # lowest logical page not yet evicted
        self._free: list[int] = list(range(int(num_pages) - 1, -1, -1))  # pop() -> page 0 first

    # ------------------------------------------------------------------ bookkeeping
    @property
    def num_free_pages(self) -> int:
        return len(self._free)

    def seq_len(self, seq: int) -> int:
        return self._lens[seq]

    def allocated_pages(self, seq: int) -> list[int]:
        return [p for p in self._table[seq] if p >= 0]

    def _alloc(self, seq: int, logical_page: int) -> int:
        if not self._free:
            raise RuntimeError("PagedKVCache: out of pages")
        if logical_page >= self.max_pages_per_seq:
            raise RuntimeError(
                f"PagedKVCache: sequence {seq} needs logical page {logical_page} >= "
                f"max_pages_per_seq={self.max_pages_per_seq}"
            )
        phys = self._free.pop()
        self._table[seq][logical_page] = phys
        self.block_table[seq, logical_page] = phys
        return phys

    # ------------------------------------------------------------------ writes
    def append(self, seq: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append ``n`` tokens ``(n, Hkv, D)`` to sequence ``seq`` at positions ``len..len+n-1``."""
        if k_new.shape != v_new.shape or k_new.dim() != 3:
            raise ValueError("k_new/v_new must both be (n, Hkv, D)")
        if tuple(k_new.shape[1:]) != (self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"k_new must be (n, {self.num_kv_heads}, {self.head_dim}), got {tuple(k_new.shape)}"
            )
        n = int(k_new.shape[0])
        if n == 0:
            return
        start = self._lens[seq]
        positions = torch.arange(start, start + n, device=self.device)
        first_page = start // self.page_size
        first_new_page = -(-start // self.page_size)  # ceil(start / page)
        last_page = (start + n - 1) // self.page_size
        for lp in range(first_new_page, last_page + 1):
            self._alloc(seq, lp)
        pages = torch.tensor(
            self._table[seq][first_page : last_page + 1], dtype=torch.int64, device=self.device
        )
        phys = pages[positions // self.page_size - first_page]
        slot = positions % self.page_size
        self.k_cache[phys, slot] = k_new.to(self.dtype)
        self.v_cache[phys, slot] = v_new.to(self.dtype)
        self._lens[seq] = start + n
        self.seq_lens[seq] = start + n

    def append_batch(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append the same number of tokens ``(B, n, Hkv, D)`` to every sequence."""
        if k_new.shape[0] != self.num_seqs:
            raise ValueError("append_batch expects one row per sequence")
        for seq in range(self.num_seqs):
            self.append(seq, k_new[seq], v_new[seq])

    # ------------------------------------------------------------------ eviction
    def evict_window(self, seq: int, window: int, query_pos: int | None = None) -> list[int]:
        """Free the pages of ``seq`` whose every key position is ``<= query_pos - window``.

        ``query_pos`` defaults to the last written position (``seq_len - 1``), i.e. the
        position of the decode query that is about to run. Returns the freed physical pages.
        Freed pages are filled with NaN when ``poison_freed`` is set so that any unmasked read
        of an evicted key is visible in the output.
        """
        p = self._lens[seq] - 1 if query_pos is None else int(query_pos)
        limit = p - int(window)  # keys at positions <= limit are invisible to every query >= p
        freed: list[int] = []
        lp = self._first_live[seq]
        while lp < self.max_pages_per_seq and lp * self.page_size + self.page_size - 1 <= limit:
            phys = self._table[seq][lp]
            if phys >= 0:
                self._table[seq][lp] = -1
                self._free.append(phys)
                freed.append(phys)
                if self.poison_freed:
                    self.k_cache[phys].fill_(float("nan"))
                    self.v_cache[phys].fill_(float("nan"))
            lp += 1
        if lp > self._first_live[seq]:
            self.block_table[seq, self._first_live[seq] : lp] = -1
            self._first_live[seq] = lp
        return freed

    def evict_window_all(self, window: int) -> int:
        return sum(len(self.evict_window(seq, window)) for seq in range(self.num_seqs))

    # ------------------------------------------------------------------ reads (tests)
    def gather(self, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live ``(k, v)`` of ``seq`` as ``(len, Hkv, D)``; evicted positions are NaN."""
        length = self._lens[seq]
        out_k = torch.full(
            (length, self.num_kv_heads, self.head_dim), float("nan"), dtype=self.dtype,
            device=self.device,
        )
        out_v = out_k.clone()
        for lp, phys in enumerate(self._table[seq]):
            if phys < 0:
                continue
            lo = lp * self.page_size
            hi = min(lo + self.page_size, length)
            if lo >= length:
                break
            out_k[lo:hi] = self.k_cache[phys, : hi - lo]
            out_v[lo:hi] = self.v_cache[phys, : hi - lo]
        return out_k, out_v


__all__ = ["HEAD_DIM", "NUM_KV_HEADS", "PAGE_SIZE", "PagedKVCache"]

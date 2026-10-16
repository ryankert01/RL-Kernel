import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/reference/strict_model.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
''', '''import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence
''')

rep('''    ``commit=False`` writes the new K/V into the cache slots but neither advances the sequence
    lengths nor evicts (a scratch step; the next call rewrites the same slots).
    """

    cache: PagedKVCache
    seqs: list[int]
    starts: list[int]
    lens: list[int]
    commit: bool = True
    _seq_index: torch.Tensor | None = field(default=None, repr=False)

    @property
    def total(self) -> int:
        return int(sum(self.lens))
''', '''    ``commit=False`` writes the new K/V into the cache slots but neither advances the sequence
    lengths nor evicts (a scratch step; the next call rewrites the same slots).

    :meth:`validate` (called by every forward that uses a plan) fails closed on duplicate or
    out-of-range sequence ids, ragged lists, empty entries and starts that are not the
    sequences' current lengths.
    """

    cache: PagedKVCache
    seqs: list[int]
    starts: list[int]
    lens: list[int]
    commit: bool = True

    @property
    def total(self) -> int:
        return int(sum(self.lens))

    def validate(self, seq_lens_host: Sequence[int] | None = None) -> None:
        """Raise ``ValueError`` unless the plan is well formed for its cache.

        Each sequence may appear once (two entries for one sequence would write the same
        cache slots, and the second write would be read by the first entry's attention), every
        id must index a cache row (``0 <= s < cache.max_seqs``; a negative id would alias the
        last row through Python indexing and then trip a device-side assert), ``seqs``,
        ``starts`` and ``lens`` must have equal lengths, every entry must add >= 1 token, and
        ``starts[j]`` must equal the committed length of ``seqs[j]`` (the K/V of the new tokens
        are written right after the existing ones)."""
        n = len(self.seqs)
        if not (len(self.starts) == len(self.lens) == n) or n == 0:
            raise ValueError(
                f"KVPlan needs equal, non-empty seqs/starts/lens; got {len(self.seqs)}/"
                f"{len(self.starts)}/{len(self.lens)}"
            )
        for s in self.seqs:
            if isinstance(s, bool) or int(s) != s:
                raise ValueError(f"KVPlan sequence ids must be ints; got {s!r}")
        bad = [s for s in self.seqs if not 0 <= int(s) < self.cache.max_seqs]
        if bad:
            raise ValueError(
                f"KVPlan sequence ids {bad} out of range [0, {self.cache.max_seqs}) of the cache"
            )
        if len(set(int(s) for s in self.seqs)) != n:
            raise ValueError(f"KVPlan sequence ids must be unique; got {list(self.seqs)}")
        if any(int(c) < 1 for c in self.lens):
            raise ValueError(f"KVPlan lens must be >= 1; got {list(self.lens)}")
        if seq_lens_host is None:
            seq_lens_host = [int(x) for x in self.cache.seq_lens.tolist()]
        wrong = [
            (int(s), int(st), int(seq_lens_host[int(s)]))
            for s, st in zip(self.seqs, self.starts)
            if int(st) != int(seq_lens_host[int(s)])
        ]
        if wrong:
            raise ValueError(
                "KVPlan starts must equal the sequences' committed lengths; (seq, start, "
                f"length): {wrong}"
            )
''')

rep('''        cfg, ops, w = self.cfg, self.ops, self.weights
        n = int(token_ids.shape[0])
        if isinstance(kv, Session):
            start = int(positions[0].item())
            kv = KVPlan(kv.cache, [kv.seq], [start], [n])
        if kv is None:
            if not torch.equal(positions, torch.arange(n, device=positions.device)):
                raise ValueError("training forward requires positions 0..T-1")
            cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        else:
            if kv.total != n:
                raise ValueError(f"KV plan covers {kv.total} tokens, got {n}")
''', '''        cfg, ops, w = self.cfg, self.ops, self.weights
        n = int(token_ids.shape[0])
        if isinstance(kv, Session):
            start = int(positions[0].item())
            kv = KVPlan(kv.cache, [kv.seq], [start], [n])
        if kv is None:
            if not torch.equal(positions, torch.arange(n, device=positions.device)):
                raise ValueError("training forward requires positions 0..T-1")
            cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        else:
            kv.validate()
            if kv.total != n:
                raise ValueError(f"KV plan covers {kv.total} tokens, got {n}")
            if not torch.equal(positions, kv.positions(positions.device)):
                raise ValueError("positions must be the plan's absolute positions starts[j]..")
''')

rep('''        window = cfg.sliding_window

        with self._region("embedding", "model"):''', '''        window = cfg.sliding_window
        # O4: ONE set of cos/sin rows per forward (rows of the canonical table, gathered once and
        # shared by every local layer); global layers get none and say so (local=False).
        rope_rows = None
        if any(layer.is_local for layer in w.layers):
            with self._region("qknorm_rope", "model"):
                rope_rows = ops.rope_table(
                    positions, theta=cfg.rope_theta, dim=cfg.head_dim, device=self.device
                )

        with self._region("embedding", "model"):''')

rep('''            with self._region("qknorm_rope", kind, i):
                if layer.is_local:
                    cos, sin = ops.rope_table(
                        positions, theta=cfg.rope_theta, dim=cfg.head_dim, device=self.device
                    )
                else:
                    cos = sin = None
                q_hat, k_hat = ops.qknorm_scaled_rope(
                    q, k, cos, sin, eps=cfg.rms_norm_eps, qk_scale=cfg.qk_scale_factor
                )''', '''            with self._region("qknorm_rope", kind, i):
                cos, sin = rope_rows if layer.is_local else (None, None)
                q_hat, k_hat = ops.qknorm_scaled_rope(
                    q,
                    k,
                    cos,
                    sin,
                    eps=cfg.rms_norm_eps,
                    qk_scale=cfg.qk_scale_factor,
                    local=layer.is_local,
                )''')

rep('''        """One rollout step for ``len(seqs)`` sequences of ``cache`` packed into ONE call
        (batched decode: every GEMM has M = batch, attention gets one query per sequence).

        ``tokens[b]`` is fed at the next position of ``seqs[b]``; returns the fp32 logp of
        ``targets[b]`` (NaN where −1). ``commit=False`` leaves the cache lengths unchanged
        (the K/V slots are rewritten by the next call), which lets the same step be replayed
        at another batch size for the batch-invariance gate (contract G2)."""
        seqs = [int(s) for s in seqs]
        lens_host = [int(x) for x in cache.seq_lens.tolist()]
        plan = KVPlan(cache, seqs, [lens_host[s] for s in seqs], [1] * len(seqs), commit=commit)
''', '''        """One rollout step for ``len(seqs)`` sequences of ``cache`` packed into ONE call
        (batched decode: every GEMM has M = batch, attention gets one query per sequence).

        ``tokens[b]`` is fed at the next position of ``seqs[b]``; returns the fp32 logp of
        ``targets[b]`` (NaN where −1). ``commit=False`` leaves the cache lengths unchanged
        (the K/V slots are rewritten by the next call), which lets the same step be replayed
        at another batch size for the batch-invariance gate (contract G2). Sequence ids must
        be unique and inside ``[0, cache.max_seqs)`` (``ValueError`` otherwise)."""
        seqs = list(seqs)
        lens_host = [int(x) for x in cache.seq_lens.tolist()]
        probe = KVPlan(cache, seqs, [0] * len(seqs), [1] * len(seqs), commit=commit)
        # ids first (a negative id must not reach lens_host[s]), then the real starts
        probe.validate([0] * cache.max_seqs)
        seqs = [int(s) for s in seqs]
        plan = KVPlan(cache, seqs, [lens_host[s] for s in seqs], [1] * len(seqs), commit=commit)
''')

rep('''        mode: str = "train",
        chunk_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Selected-token contract: fp32 logp of every completion token given the prefix.

        ``mode``: ``train`` (one differentiable training forward over prompt+completion),
        ``prefill`` (one-shot paged prefill of prompt+completion[:-1]), ``chunked`` (the same
        in ``chunk_sizes`` chunks), ``decode`` (one-shot prompt prefill, then one decode step
        per completion token), ``per_token`` (every token, prompt included, as its own step).
        """''', '''        mode: str = "train",
        chunk_sizes: Sequence[int] | None = None,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Selected-token contract: fp32 logp of every completion token given the prefix.

        ``mode``: ``train`` (one differentiable training forward over prompt+completion),
        ``prefill`` (one-shot paged prefill of prompt+completion[:-1]), ``chunked`` (the same
        in ``chunk_sizes`` chunks), ``decode`` (one-shot prompt prefill, then one decode step
        per completion token), ``per_token`` (every token, prompt included, as its own step).
        ``timings`` (decode only): filled with ``prompt_prefill_seconds`` and
        ``decode_step_seconds`` (one entry per decode step, device-synchronised wall time);
        timing never changes the arithmetic.
        """''')

rep('''        # decode: prompt one shot, then token by token through the completion
        prompt_targets = torch.cat([prompt[1:], completion[:1]])
        out = [self.prefill(session, prompt, targets=prompt_targets)[-1:]]
        for j in range(lc - 1):
            out.append(self.decode_step(session, completion[j], int(completion[j + 1])))
        return torch.cat(out, dim=0)''', '''        # decode: prompt one shot, then token by token through the completion
        prompt_targets = torch.cat([prompt[1:], completion[:1]])
        clock = _Clock(self.device, timings is not None)
        out = [self.prefill(session, prompt, targets=prompt_targets)[-1:]]
        prefill_s = clock.lap()
        steps: list[float] = []
        for j in range(lc - 1):
            out.append(self.decode_step(session, completion[j], int(completion[j + 1])))
            steps.append(clock.lap())
        if timings is not None:
            timings["prompt_prefill_seconds"] = prefill_s
            timings["prompt_len"] = lp
            timings["decode_step_seconds"] = steps
            timings["decode_context_start"] = lp  # step j attends to lp + j + 1 keys
        return torch.cat(out, dim=0)''')

rep('''def _row_parallel(ops: OpSet, a: torch.Tensor, lin) -> torch.Tensor:''', '''class _Clock:
    """Device-synchronised lap timer (a no-op unless ``enabled``)."""

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self._t = self._now()

    def _now(self) -> float:
        if self.enabled:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def lap(self) -> float:
        if not self.enabled:
            return float("nan")
        now = self._now()
        dt, self._t = now - self._t, now
        return dt


def _row_parallel(ops: OpSet, a: torch.Tensor, lin) -> torch.Tensor:''')

p.write_text(s)
print("ok")

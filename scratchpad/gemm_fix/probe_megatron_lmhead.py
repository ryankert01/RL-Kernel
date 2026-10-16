"""Check the hypothesis for test_lm_head_tiles_tp2_equals_tp1_and_reference: the record pass of
the RecordReplayComm returns the rank's OWN stats for every peer, so the backward partials it
records for the peers are computed with a wrong logZ.  A second record pass that already replays
the (correct) forward gathers fixes the emulation."""

import torch

import tests.models.muse_glimmer.test_megatron_strict_binding as T

torch.manual_seed(0)


class Comm(T.RecordReplayComm):
    def all_gather(self, x):
        idx = self.counter
        self.counter += 1
        if self.mode == "record":
            self.book.setdefault(idx, {})[self.rank] = x.detach().clone()
            return [x.detach().clone() for _ in range(self.world)]
        if self.mode == "rerecord":  # replay peers from pass 1, record own fresh value
            self.new_book.setdefault(idx, {})[self.rank] = x.detach().clone()
            return [x.detach().clone() if r == self.rank else self.book[idx][r]
                    for r in range(self.world)]
        return [self.book[idx][r] for r in range(self.world)]


def simulate_fixed(world, fn):
    book = {}
    comms = [Comm(world, r, book) for r in range(world)]
    for c in comms:
        fn(c)
    new_book = {}
    for c in comms:
        c.mode, c.counter, c.new_book = "rerecord", 0, new_book
        fn(c)
    book.clear()
    book.update(new_book)
    outs = []
    for c in comms:
        c.mode, c.counter = "replay", 0
        outs.append(fn(c))
    return outs


vocab, padded, hidden, m = 2000, 2048, 256, 24
tile_rows = padded // 8
mult, cap = 0.19611613513818404, 20.0
h = torch.randn(m, hidden, device="cuda").to(torch.bfloat16)
w = (torch.randn(vocab, hidden, device="cuda") * 0.05).to(torch.bfloat16)
w_pad = torch.nn.functional.pad(w, (0, 0, 0, padded - vocab))
targets = torch.randint(0, vocab, (m,), device="cuda")


def run(comm):
    ids = T.sk.local_chunk_ids(comm.rank, comm.world)
    tiles = w_pad[ids[0] * tile_rows:(ids[-1] + 1) * tile_rows].contiguous()
    hg = h.clone().requires_grad_(True)
    lp = T.StrictLMHeadLogprob.apply(hg, targets, tiles, tuple(ids), comm, vocab, tile_rows,
                                     mult, cap)
    (dh,) = torch.autograd.grad(lp.sum(), (hg,))
    return lp, dh


lp1, dh1 = T._simulate(1, run)[0]
print("original helper:", [(torch.equal(a, lp1), torch.equal(b, dh1)) for a, b in T._simulate(2, run)])
print("fixed helper   :", [(torch.equal(a, lp1), torch.equal(b, dh1)) for a, b in simulate_fixed(2, run)])
for world in (4, 8):
    print(f"fixed helper TP{world}:", [(torch.equal(a, lp1), torch.equal(b, dh1))
                                      for a, b in simulate_fixed(world, run)])

from rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob import (  # noqa: E402
    lm_head_softcap_logprob,
)

z = (h.float() @ w.float().t()) * mult
ref = torch.log_softmax(cap * torch.tanh(z / cap), dim=-1).gather(1, targets[:, None]).squeeze(1)
print("ref allclose:", torch.allclose(lp1, ref, atol=2e-2, rtol=1e-2))
sib = lm_head_softcap_logprob(h, w_pad, targets, multiplier=mult, softcap=cap, vocab_size=vocab)
print("sibling op == binding TP1:", torch.equal(sib, lp1))

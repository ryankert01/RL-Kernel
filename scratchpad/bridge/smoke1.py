import sys
import time

import torch

from rl_engine.models.muse_glimmer.e2e import bridge, grpo, mocks

work = sys.argv[1]
launcher = sys.argv[2] if len(sys.argv) > 2 else "process"
tp = 2 if launcher == "torchrun" else 1
t0 = time.time()
rt = bridge.RemoteTrainer(
    "mock", tensor_parallel_size=tp, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7,
    factory=bridge.MOCK_TRAINER_FACTORY, work_dir=work, launcher=launcher,
    call_timeout_s=120, startup_timeout_s=120,
)
print("ready", round(time.time() - t0, 2), rt._hello["versions"]["torch"], rt._hello["devices"])
local = mocks.MockTrainer(
    "mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7
)
seqs = [[256, 3, 5, 7, 9, 11], [256, 4, 6, 8, 10]]
starts = [3, 2]
for step in range(3):
    lr_ = rt.token_logprobs(seqs, starts)
    ll = local.token_logprobs(seqs, starts)
    eq = all(
        torch.equal(a.detach().view(torch.int32), b.detach().view(torch.int32))
        for a, b in zip(lr_, ll)
    )
    adv = torch.tensor([1.0, -1.0])
    lossr = grpo.grpo_token_loss(lr_, [t.detach() for t in ll], adv).loss
    lossl = grpo.grpo_token_loss(ll, [t.detach() for t in ll], adv).loss
    ir = rt.step(lossr)
    il = local.step(lossl)
    sdr, shr = rt.export_lora()
    sdl, shl = local.export_lora()
    print(step, "lp equal", eq, "gn", ir["grad_norm"], il["grad_norm"], "sha equal", shr == shl,
          rt.last_export["rank_sha256"])
print("calls", [(c["op"], round(c["rtt_s"] * 1e3, 2), c["server"]["ranks"][-1]["op_counts"])
                for c in rt.calls[-3:]])
rt.close()
print("closed alive=", rt.alive())

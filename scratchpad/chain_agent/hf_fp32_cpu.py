"""HF MuseGlimmer in fp32 on CPU for the chain-gate tokens: the anchor-of-the-anchor."""

import sys
import time

import torch

from rl_engine.models.muse_glimmer.reference.run_chain import make_tokens

ckpt, out_path = sys.argv[1], sys.argv[2]
prompt_len, completion_len = int(sys.argv[3]), int(sys.argv[4])
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 64
torch.set_num_threads(threads)
from transformers import MuseGlimmerForConditionalGeneration  # noqa: E402

t0 = time.perf_counter()
model = MuseGlimmerForConditionalGeneration.from_pretrained(
    ckpt, dtype=torch.float32, attn_implementation="eager"
)
model.eval()
print(f"loaded fp32 on cpu in {time.perf_counter() - t0:.1f}s", flush=True)
ids = make_tokens(prompt_len, completion_len, 20261015, 202048)
t1 = time.perf_counter()
with torch.no_grad():
    out = model(input_ids=ids[None], output_hidden_states=True, use_cache=False)
print(f"forward {ids.numel()} tokens in {time.perf_counter() - t1:.1f}s", flush=True)
torch.save(
    {
        "logits": out.logits[0].float(),
        "hidden_states": [h[0] for h in out.hidden_states],
        "dtype": "float32",
        "token_ids": ids,
    },
    out_path,
)
print("saved", out_path, flush=True)

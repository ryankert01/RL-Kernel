import os, sys, time, json
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
from megatron.bridge import AutoBridge
path = os.path.expanduser("~/models/Muse-Glimmer-30B")
bridge = AutoBridge.from_hf_pretrained(path)
cfg = bridge.get_model_config()
print("config type", type(cfg).__name__, "tp", cfg.transformer.tensor_model_parallel_size, flush=True)
tc = cfg.transformer
print("window_size", tc.window_size, "skip_freq[:8]", tc.window_attn_skip_freq[:8], "no_rope[:8]", tc.no_rope_freq[:8])
print("softmax_scale", tc.softmax_scale, "zero_centered", tc.layernorm_zero_centered_gamma, "impl", tc.transformer_impl)
print("eps", tc.layernorm_epsilon, tc.post_norm_epsilon, "mult", tc.output_multiplier, "softcap", tc.final_logit_softcapping)
print("vocab", cfg.vocab_size, "pad", cfg.should_pad_vocab, "seq", cfg.seq_length, "rotary_base", cfg.rotary_base)
tc.tensor_model_parallel_size = 1
t1 = time.time()
try:
    models = bridge.get_model(cfg, wrap_with_ddp=False, mixed_precision_wrapper=None)
except TypeError as e:
    print("get_model kwargs rejected:", e); models = bridge.get_model(cfg, wrap_with_ddp=False)
print("load s", time.time() - t1, "alloc GiB", torch.cuda.memory_allocated() / 2**30, flush=True)
m = models[0]
print("model type", type(m).__name__)
inner = m
while hasattr(inner, "module"):
    inner = inner.module
print("inner", type(inner).__name__)
n_text = n_vis = 0
for n, p in inner.named_parameters():
    if "vision" in n:
        n_vis += p.numel(); continue
    n_text += p.numel()
    if "layers." in n and ".layers.0." not in n and ".layers.3." not in n:
        continue
    print(f"  {n} {tuple(p.shape)} {p.dtype} grad={p.requires_grad}")
print("text params", n_text, "vision params", n_vis)
for n, b in inner.named_buffers():
    if "vision" in n: continue
    if "layers." in n and ".layers.0." not in n: continue
    print("  buf", n, tuple(b.shape), b.dtype)
l0 = inner.decoder.layers[0]
print("layer0 type", type(l0).__name__)
for n, mod in l0.named_modules():
    print("   mod", n, type(mod).__name__, {k: v for k, v in vars(mod).items() if k in ("eps", "zero_centered_gamma", "layer_number", "window_size", "softmax_scale", "input_is_parallel", "gather_output", "sequence_parallel")})
att = l0.self_attention
print("attn attrs", att.num_attention_heads_per_partition, att.num_query_groups_per_partition, att.hidden_size_per_attention_head, att.layer_number, type(att.core_attention).__name__, type(att.q_layernorm).__name__, getattr(att.q_layernorm, 'eps', None))
print("core attn", {k: v for k, v in vars(att.core_attention).items() if not k.startswith('_') and not isinstance(v, torch.nn.Module)})
print("final_norm", type(inner.decoder.final_norm).__name__, getattr(inner.decoder.final_norm, 'eps', None), getattr(inner.decoder.final_norm, 'zero_centered_gamma', None))
print("output_layer", type(inner.output_layer).__name__, tuple(inner.output_layer.weight.shape), inner.output_layer.gather_output)
print("embedding", type(inner.embedding.word_embeddings).__name__, tuple(inner.embedding.word_embeddings.weight.shape), getattr(inner.embedding.word_embeddings, 'vocab_start_index', None), getattr(inner.embedding.word_embeddings, 'vocab_end_index', None))
print("rotary", type(inner.rotary_pos_emb).__name__, inner.rotary_pos_emb.inv_freq[:4], inner.rotary_pos_emb.inv_freq.dtype, inner.rotary_pos_emb.inv_freq.shape)

from safetensors import safe_open
idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
def hf(name):
    with safe_open(os.path.join(path, idx[name]), "pt", device="cpu") as f:
        return f.get_tensor(name)
pre = "model.language_model."
qkv = l0.self_attention.linear_qkv.weight.detach().cpu()
print("qkv shape", qkv.shape)
G, hq, hkv, D, H = 2, 32, 2, 128, 6656
per_group = (2 * (hq // hkv) + 2) * D
blk = qkv.view(G, per_group, H)
q = blk[:, :16 * D].reshape(-1, H); g = blk[:, 16 * D:32 * D].reshape(-1, H)
k = blk[:, 32 * D:33 * D].reshape(-1, H); v = blk[:, 33 * D:34 * D].reshape(-1, H)
for nm, t in (("q_proj", q), ("gate_proj", g), ("k_proj", k), ("v_proj", v)):
    print(nm, torch.equal(t, hf(f"{pre}layers.0.self_attn.{nm}.weight")))
fc1 = l0.mlp.linear_fc1.weight.detach().cpu()
print("fc1", fc1.shape, torch.equal(fc1[:19968], hf(f"{pre}layers.0.mlp.gate_proj.weight")), torch.equal(fc1[19968:], hf(f"{pre}layers.0.mlp.up_proj.weight")))
print("fc2", torch.equal(l0.mlp.linear_fc2.weight.detach().cpu(), hf(f"{pre}layers.0.mlp.down_proj.weight")))
print("proj", torch.equal(l0.self_attention.linear_proj.weight.detach().cpu(), hf(f"{pre}layers.0.self_attn.o_proj.weight")))
print("in_ln", torch.equal(l0.input_layernorm.weight.detach().cpu(), hf(f"{pre}layers.0.input_layernorm.weight")))
print("post_attn_ln", torch.equal(l0.self_attention.post_layernorm.weight.detach().cpu(), hf(f"{pre}layers.0.post_attention_layernorm.weight")))
fn = inner.decoder.final_norm.weight.detach().cpu(); fh = hf(f"{pre}norm.weight")
print("final_norm raw equal", torch.equal(fn, fh), "plus1 equal", torch.equal((fn.float() + 1).to(torch.bfloat16), fh), "plus1 bf16-add equal", torch.equal(fn + 1, fh), "n diff", ((fn.float() + 1).to(torch.bfloat16) != fh).sum().item())
print("embed", torch.equal(inner.embedding.word_embeddings.weight.detach().cpu(), hf(f"{pre}embed_tokens.weight")))
print("lm_head", torch.equal(inner.output_layer.weight.detach().cpu(), hf("lm_head.weight")))

inner.eval()
ids = torch.tensor([[200000, 1000, 2000, 3000, 4000, 5000, 6000, 7000]], device="cuda")
pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
S = ids.shape[1]
mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device="cuda"), diagonal=1)[None, None]
with torch.no_grad():
    t2 = time.time()
    out = inner(input_ids=ids, position_ids=pos, attention_mask=mask)
    torch.cuda.synchronize()
print("native out", out.shape, out.dtype, "t", time.time() - t2, "max alloc GiB", torch.cuda.max_memory_allocated() / 2**30)
lp = torch.log_softmax(out.float(), dim=-1)
tgt = ids[:, 1:]
print("native logp", lp[0, :-1].gather(-1, tgt[0].unsqueeze(-1)).squeeze(-1).tolist())
print("total s", time.time() - t0)

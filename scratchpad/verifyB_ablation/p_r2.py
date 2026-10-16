import json
import sys

d = json.load(open(sys.argv[1]))
by = {}
for r in d['records']:
    if r['engine'] == 'hf' and not r.get('derived_from_stored_values') and '@' not in r['setting']:
        by.setdefault(r['setting'], {})[r['sequence']] = r
seqs = list(by['prefill_L_minus_1'])
print('sequences', seqs)


def cmp(a, b):
    per = []
    for s in seqs:
        x, y = by[a][s]['logp_hex'], by[b][s]['logp_hex']
        assert len(x) == len(y)
        per.append(sum(p != q for p, q in zip(x, y)))
    return sum(per), per


for c in (64, 256, 1024):
    print(f'chunk {c} vs L-1', cmp(f'chunked_prefill_{c}', 'prefill_L_minus_1'))
print('logits_fp32 vs softcap_fp32', cmp('logits_fp32', 'softcap_fp32'))
print('rope_el_tf32 vs rope_el', cmp('rope_elementwise_tf32_on', 'rope_elementwise'))
print('rollout sdpa vs rollout', cmp('rollout_like_batched_decode_sdpa', 'rollout_like_batched_decode'))
# baseline vs rope_inv_freq_cuda per seq
base = {p['sequence']: p for p in d['baselines']['hf']['per_sequence']}
per = [sum(p != q for p, q in zip(base[s]['logp_hex'], by['rope_inv_freq_cuda'][s]['logp_hex']))
       for s in seqs]
print('rope_inv_freq_cuda vs base', sum(per), per)
# derived summaries
for s in d['settings']:
    if s.get('derived_from_stored_values'):
        print(s['setting'], s['n_bits_differ'], s['n_tokens'])
print('inv_freq cpu', d['rope_inv_freq_hex'][:6])
diff = [(i, a, b) for i, (a, b) in enumerate(zip(d['rope_inv_freq_hex'], d['rope_inv_freq_cuda_hex'])) if a != b]
print('inv_freq entries differ', diff)
print('previous_runs', json.dumps([{k: v for k, v in p.items() if k != 'per_sequence'} for p in d['previous_runs']], indent=0)[:2000])

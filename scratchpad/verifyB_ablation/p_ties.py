import collections
import json
import sys

d = json.load(open(sys.argv[1]))
base = {p['sequence']: p for p in d['baselines']['hf']['per_sequence']}
print('baseline ties', sum(sum(p['top1_tie']) for p in base.values()),
      d['analysis']['hf_baseline_top1_ties']['n'])
meas = ['padded_batch9_right', 'padded_batch9_left', 'kv_cache_decode', 'prefill_L_minus_1', 'sdpa',
        'softcap_fp32', 'rope_inv_freq_cuda', 'bf16_reduced_reduction_off']
tot = collections.Counter()
for r in d['records']:
    if r['engine'] != 'hf' or r['setting'] not in meas:
        continue
    b = base[r['sequence']]
    for i, f in enumerate(r['argmax_flip']):
        if not f:
            continue
        bt = b['top1_tie'][i]
        vt = r['top1_tie'][i]
        kind = 'base_tie' if bt else ('var_tie_only' if vt else 'unique')
        tot[kind] += 1
        if kind != 'base_tie':
            print(r['setting'], r['sequence'], i, kind, 'bgap', b['top2_gap'][i], 'vgap',
                  r['var_top2_gap'][i], 'bam', b['argmax'][i], 'vam', r['var_argmax'][i])
print(tot)
alt = collections.defaultdict(set)
for r in d['records']:
    if r['engine'] != 'hf' or r.get('derived_from_stored_values') or r.get('argmax_flip') is None:
        continue
    b = base.get(r['sequence'])
    if b is None:
        continue
    for i, f in enumerate(r['argmax_flip']):
        if f and b['top1_tie'][i]:
            alt[(r['sequence'], i, b['argmax'][i])].add(r['var_argmax'][i])
for k, v in alt.items():
    print(k, sorted(v))

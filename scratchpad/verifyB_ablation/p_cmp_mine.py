import json
import sys

mine = json.load(open(sys.argv[1]))
d = json.load(open(sys.argv[2]))
print('my part commit', mine['commit'][:7], 'dirty', mine['worktree_dirty'], 'gpu', mine['env'].get('gpu_lease'),
      'status', mine['status'])
cb = {p['sequence']: p['logp_hex'] for p in d['baselines']['hf']['per_sequence']}
mb = {p['sequence']: p['logp_hex'] for p in mine['baseline']['per_sequence']}
print('baseline tokens differ vs committed', sum(sum(a != b for a, b in zip(mb[s], cb[s])) for s in cb))
crec = {(r['setting'], r['sequence']): r for r in d['records'] if r['engine'] == 'hf'}
mrec = {}
for r in mine['records']:
    mrec[(r['setting'], r['sequence'])] = r
    c = crec[(r['setting'], r['sequence'])]
    assert c['logp_hex'] == r['logp_hex'], (r['setting'], r['sequence'])
    assert c['var_argmax'] == r['var_argmax']
print('variant records bitwise equal to committed:', len(mine['records']))
seqs = list(cb)
per = [sum(a != b for a, b in zip(mrec[('chunked_prefill_256', s)]['logp_hex'], mrec[('prefill_L_minus_1', s)]['logp_hex'])) for s in seqs]
print('my chunk256 vs L-1', sum(per), per)
per = [sum(mrec[('rope_inv_freq_cuda', s)]['bits_differ']) for s in seqs]
print('my rope_inv_freq_cuda vs base', sum(per), per)

import json
import sys

d = json.load(open(sys.argv[1]))
print('tf32_probe', d['tf32_probe'])
print('layer_points', d['layer_points'])
for r in d['layer_records']:
    det = r['detail_points']
    firsts = [(p['point'], p['frac_elems_differ'], p.get('first_row_with_diff'), p.get('rows_differ', p.get('n_rows_differ')))
              for p in det if p['frac_elems_differ'] > 0][:3]
    print(f"{r['sequence']:18s} {r['setting']:28s} first={r['first_point_with_diff']} "
          f"first_res={r['first_residual_point_with_diff']} detail_first3={firsts}")
print(sorted(d['layer_records'][0]['detail_points'][0].keys()))
lp = d['analysis']['low_p_gold_tokens']
print(lp['status'], json.dumps(lp.get('by_bucket')))
print('n low', len(lp.get('low_p_tokens', [])), 'in calc', sum(t['in_calc'] for t in lp.get('low_p_tokens', [])),
      'first tok', [round(t['logp'], 1) for t in lp.get('low_p_tokens', []) if t['index'] == 0])
bb = d['analysis']['abs_dlogp_by_baseline_prob'].get('hf/padded_batch9_right')
print('bucket padded right', json.dumps(bb))
print('pairwise names', d['analysis']['hf_pairwise_identical_tokens']['settings'])

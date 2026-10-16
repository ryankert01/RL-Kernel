import json

d = json.load(open('/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-12/reports/muse_glimmer_poc/data/megatron_runtime.json'))
sets = [
    ('tp1_followup', d['tp1_bench_followup']['checks']['bench']['rows']),
    ('tp1_run', d['runs']['tp1']['checks'].get('bench', {}).get('rows', [])),
    ('tp2', d['runs']['tp2']['checks']['bench']['rows']),
]
for name, rows in sets:
    print(name)
    for r in rows:
        nfb = r.get('native_fwd_bwd_forward_ms', 0) + r.get('native_fwd_bwd_backward_ms', 0)
        st = r.get('strict_step_total_ms')
        print('  ', r['tokens'], 'sfwd %.0f' % r['strict_forward_ms'], 'nfwd %.0f' % r.get('native_forward_ms', -1),
              'sstep', None if st is None else round(st), 'nfb %.0f' % nfb,
              'ratio', None if st is None or not nfb else round(st / nfb, 2),
              'speak %.1f' % r.get('strict_train_peak_gib', -1),
              'recomp', r.get('strict_recompute_fwd_bwd_ms'),
              r.get('strict_train_error', '')[:60], r.get('native_fwd_bwd_error', '')[:60])
print(json.dumps({k: v for k, v in d['tp1_bench_followup'].items() if k != 'checks'}, indent=1)[:1500])
print(json.dumps(d['runs']['tp1']['load'])[:800])
print(json.dumps(d['runs']['tp2']['load'])[:800])
print('tp1 run keys with native', d['runs']['tp1']['checks']['native'].keys())
print('tp1 identity', {k: v for k, v in d['runs']['tp1']['checks']['identity'].items() if k not in ('per_tensor',)})
print('tp2 identity', {k: v for k, v in d['runs']['tp2']['checks']['identity'].items() if k not in ('per_tensor',)})
for t in ('tp1', 'tp2'):
    c = d['runs'][t]['checks']
    print(t, 'repeat', c['repeat']['total_mismatches'], 'noop', c['lora_noop']['total_mismatches'],
          'native max/mean', c['native']['max_abs_vs_strict'], c['native']['mean_abs_vs_strict'],
          'xrt s', c['xrt_strict']['seconds'],
          'grad', c['grad_repeat']['repeat_mismatching_elements'], c['grad_repeat']['elements'], c['grad_repeat']['zero_tensors'],
          [p.get('replicated_differing') for p in c['grad_repeat']['passes']], [p.get('replicated_checked') for p in c['grad_repeat']['passes']],
          'step', c['step']['info']['grad_norm'], c['step']['lora_sha256_after'][:12])

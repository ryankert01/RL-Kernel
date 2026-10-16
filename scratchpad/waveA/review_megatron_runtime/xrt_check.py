import json, struct, os, hashlib, itertools
W = '/home/ubuntu/RL-Kernel/.claude/worktrees/'
files = {
    'meg_s1': W + 'wf_7b064961-36f-12/reports/muse_glimmer_poc/data/xrt_megatron_strict_tp1.json',
    'meg_s2': W + 'wf_7b064961-36f-12/reports/muse_glimmer_poc/data/xrt_megatron_strict_tp2.json',
    'meg_n1': W + 'wf_7b064961-36f-12/reports/muse_glimmer_poc/data/xrt_megatron_native_tp1.json',
    'meg_n2': W + 'wf_7b064961-36f-12/reports/muse_glimmer_poc/data/xrt_megatron_native_tp2.json',
    'vllm_s1': W + 'wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json',
    'vllm_s2': W + 'wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_strict_tp2.json',
    'vllm_n1': W + 'wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_native_tp1.json',
    'ref': W + 'wf_7b064961-36f-8/reports/muse_glimmer_poc/data/xrt_reference_triton.json',
}
data = {}
for k, f in files.items():
    if not os.path.exists(f):
        print('MISSING', k, f)
        continue
    d = json.load(open(f))
    seqs = {s['id']: s for s in d['sequences']}
    data[k] = seqs
    print(k, d.get('commit'), d.get('tp'), d.get('mode'), 'nseq', len(seqs),
          'sha', hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12])
    bad = 0; neg = 0; tot = 0
    for s in d['sequences']:
        hx = s['logprobs_hex']
        fl = s.get('logprobs')
        for i, h in enumerate(hx):
            tot += 1
            if h.startswith('-'):
                neg += 1
            if fl is not None and struct.pack('>f', fl[i]) != bytes.fromhex(h):
                bad += 1
    print('   tokens', tot, 'neg-sign-hex', neg, 'hex!=float', bad, 'has floats', 'logprobs' in d['sequences'][0])
keys = list(data)
for a, b in itertools.combinations(keys, 2):
    mm = 0; n = 0; mx = 0.0
    for sid in data[a]:
        ha = data[a][sid]['logprobs_hex']; hb = data[b][sid]['logprobs_hex']
        assert len(ha) == len(hb), (a, b, sid)
        for x, y in zip(ha, hb):
            n += 1
            if x != y:
                mm += 1
                fx = struct.unpack('>f', bytes.fromhex(x))[0]; fy = struct.unpack('>f', bytes.fromhex(y))[0]
                mx = max(mx, abs(fx - fy))
    print(f'{a:8s} vs {b:8s}: {mm}/{n} max {mx:.4f}')

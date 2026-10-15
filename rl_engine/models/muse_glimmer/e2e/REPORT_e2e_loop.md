# REPORT — `final_e2e_lora`: GRPO LoRA loop for the Muse-Glimmer-30B strict POC

STATUS: partial — orchestrator, mocks and tests are complete and passing on CPU (37 tests). The
real run (vLLM TP2 + Megatron Bridge TP4) was NOT executed: the sibling engines
`rl_engine/models/muse_glimmer/runtime/{strict_rollout_engine,strict_trainer}.py` do not exist in
this checkout (imported lazily; `--mock` is the fallback). Resume with the commands in
"Commands for the integrator".

## Summary

* `rl_engine/models/muse_glimmer/e2e/` — single-process, dependency-light GRPO loop (no VIME, no
  Ray, no `datasets`): rollout at one physical TP, LoRA trainer at another, strict fp32
  selected-token logprobs on both sides, hash-verified LoRA sync, per-step drift metrics with
  sidecars, GSM8K exact-match reward. Built against the FIXED engine interfaces (`engines.py`
  holds the protocols); CPU mocks implement the same surface.
* Mock results (8 steps x 8 prompts x 4 samples x 64 tokens, seed 1234): strict/strict
  0 / 13951 bitwise mismatches, max |d| 0.0, ratio exactly 1 on every token, clip never active.
  strict/native, native/strict, native/native: >= 13948 / 13950 mismatches (max |d| 1.4e-2 to
  4.0e-2, first drift at step 0 sample 0 token 0). Perturbed rollout weights: 120/120 tokens
  drift. Dense-reward learning check (40 steps, lr 1e-2): reward 0.522 -> 0.794, 0/10076
  mismatches, 41/41 weight-sync checks. The metric detects drift and is exactly zero when the
  engines agree; the GRPO update has the right sign.
* Data: `~/data/gsm8k/{train,test}.jsonl` (7473 / 1319 rows, curl into a fresh dir, parsed as
  data only); sha256 17f347dc...3465 / 3730d312...9d14.

## Architecture

```
run_loop: export_lora_verified(trainer) -> transfer_lora(rollout)        # sync BEFORE rollout 0
for step:
  prompts = task.sample(P)                      # GSM8K chat-template prompts (system + user)
  groups  = rollout.generate(ids, n=S, max_tokens, T)   # RolloutSample(token_ids, fp32 logprobs)
  rewards = task.reward(completion_ids)         # exact match of the last "#### <number>"
  lp_tr   = trainer.token_logprobs(prompt+completion, starts)   # fp32 with grad, same tokens
  drift   = compare_logprobs(lp_tr, lp_rollout) ; sidecar step{n:04d}.pt
  (opt.)  rollout.logprobs(same tokens) vs generate-time lp      # decode-vs-prefill self check
  loss    = grpo_token_loss(lp_tr, lp_rollout, group_advantages) # fp32, clip 0.2, token mean
  trainer.step(loss); pkg = export_lora_verified(trainer)        # hash recomputed, keys checked
  transfer_lora(pkg, rollout, via_file?) ; rollout.provenance()["lora_sha256"] == pkg.sha256
  steps.jsonl + rounds.csv ; end: summary.json, README.md, provenance.json, config.json
```
`--mode <train>/<rollout>` (RFC "training-rollout" order). GPU placement is passed verbatim
(`gpu_ids=[0,1]` TP2 rollout, `[4,5,6,7]` TP4 trainer); the orchestrator never touches CUDA, so
both engines live at once. `--sync-via-file` moves LoRA through `run_dir/lora/step{n:04d}.pt`
(hash re-verified on load) for a separate-process trainer.

Files: engines.py, data.py, grpo.py, sync.py, drift.py, loop.py, mocks.py, run_poc_grpo.py,
tests/models/muse_glimmer/test_e2e_loop_mock.py.

## lora_sha256 — exact definition (schema muse_glimmer_lora_sha256_v1)

```python
h = hashlib.sha256()
for name in sorted(state_dict):                        # Python str order (code points)
    t = state_dict[name].detach().cpu().contiguous()
    shape = ",".join(str(int(d)) for d in t.shape)     # "16,6656"; "" for a 0-d tensor
    h.update(f"{name}|{t.dtype}|{shape}\n".encode("utf-8"))  # "layers.0.q_proj.lora_A|torch.bfloat16|16,6656\n"
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())  # raw little-endian bytes
sha256 = h.hexdigest()
```
Only name, dtype string, shape and raw bytes are hashed (no device, stride, grad). Keys:
`layers.{i}.{q_proj|k_proj|v_proj|gate_proj|o_proj|mlp.gate_proj|mlp.up_proj|mlp.down_proj}.lora_{A,B}`,
A `(r, K)`, B `(N, r)`, bf16 (`sync.validate_lora_state_dict`). Known answer:
`{"layers.0.q_proj.lora_A": zeros(1, 2, bf16)}` ->
`sha256(b"layers.0.q_proj.lora_A|torch.bfloat16|1,2\n" + 4 zero bytes)`. The trainer's
`export_lora()` must return this hash (the loop recomputes it and fails closed); the rollout's
`load_lora` must verify it and `provenance()["lora_sha256"]` must report it
(`weight_sync_identity`).

## Drift metric (drift.compare_logprobs)

On the SAME completion tokens under the SAME LoRA hash: trainer fp32 logprobs vs rollout fp32
logprobs (rollout gives fp32-exact floats, converted back losslessly).
`bitwise_mismatch_count` (4 fp32 bytes differ, `view(torch.uint8)`), `value_mismatch_count`
(`torch.ne`), `max_abs_diff` / `mean_abs_diff` (fp64), `mismatch_kl` = mean(rollout - train),
`mismatch_k3_kl` = mean(exp(d) - d - 1), `first_mismatch` = (sample, token) of the first
bitwise mismatch in batch order (token 0 = first generated token), `torch_equal`,
`bitwise_equal` (strict/strict pass criterion). Sidecar `mismatch-sidecars/step{n:04d}.pt`
holds both logprob lists, masks, token ids, completion starts, rewards and metrics;
`metrics_from_sidecar` recomputes them (tested equal). `summary.json["first_drift"]` is the
first (step, sample, token, |d|). Mirrors the Qwen3 E2E `compare_train_rollout_logps`
(byte view, max_abs_diff, k3) without the VIME dependency.

## GRPO (grpo.py)

A = (r - mean_g) / max(std_g, 1e-6), population std, identical rewards -> exactly 0.
rho = exp(lp_train - lp_rollout) (rollout detached); loss = -mean over completion tokens of
min(rho*A, clip(rho, 0.8, 1.2)*A); optional k3 KL to a reference (off). fp32 throughout. In
strict/strict rho == 1 bitwise, clip_frac == 0 and the gradient is group-normalized REINFORCE
(tested); clipping blocks the gradient on the clipped side (tested).

## Mocks (mocks.py)

Tiny 2-block byte LM (vocab 259, d 32), LoRA on all eight projection names, identical base
weights on both engines (seed 20261015). Strict path: fp32 with only exactly-rounded elementwise
primitives (+ - * / sqrt max floor frexp, integer bit ops), polynomial exp/log, GEMM as a
sequential K loop, causal mean as a sequential prefix sum -> batch-, padding-, split- and
decode-vs-prefill-invariant by construction (tested bitwise). Native path: BLAS linear, torch
transcendentals, engine-private bf16 cast points (rollout: after every linear + bf16 running
state with Welford decode; trainer: fp32 activations, one cast per block, cumsum prefill).
LoRA: fp32 master -> bf16 at use and export; rollout holds the bf16 tensors.

## Mock results (seed 1234, 8 x 8 x 4 x 64 tokens, r 16, lr 1e-5, 8 CPU threads)

| mode (train/rollout) | bitwise mismatches / tokens | max abs diff | mean abs diff | k3 KL | rollout decode-vs-prefill mismatches | first drift (step, sample, token) | ratio==1 frac / clip frac | reward mean | weight-sync checks | s/step (rollout / train logp / train step / sync) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| strict/strict | 0 / 13951 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 / 13951 | none | 1.000 / 0.000 | 0.0312 | 9/9 | 3.21 (1.85 / 0.13 / 0.39 / 0.002) |
| strict/native | 13948 / 13950 | 3.91e-02 | 4.41e-03 | 1.66e-05 | 13681 / 13950 | (0, 0, 0) abs 2.23e-03 | 0.000 / 0.000 | 0.0312 | 9/9 | 1.00 (0.26 / 0.18 / 0.45 / 0.002) |
| native/strict | 13948 / 13951 | 1.39e-02 | 1.93e-03 | 3.06e-06 | 0 / 13951 | (0, 0, 0) abs 5.61e-04 | 0.000 / 0.000 | 0.0312 | 9/9 | 2.84 (1.87 / 0.02 / 0.09 / 0.002) |
| native/native | 13950 / 13950 | 3.97e-02 | 4.82e-03 | 1.98e-05 | 13683 / 13950 | (0, 0, 0) abs 1.67e-03 | 0.000 / 0.000 | 0.0312 | 9/9 | 0.42 (0.24 / 0.01 / 0.08 / 0.002) |

Also: `--mock-perturb 1e-3` -> 120/120 tokens drift (max 1.4e-2); CLI 3-step strict/strict
0/1079; 30-step strict/strict 0/14840 (sync 31/31); GSM8K jsonl via the byte tokenizer
(`--task gsm8k`, 2 steps) 0/991 with reward 0 -> every group uniform -> advantages exactly 0 ->
LoRA hash constant (an invariant the real run can check); dense-reward learning check
(first-token parity, 40 steps, lr 1e-2): reward 0.522 -> 0.794, 0/10076 mismatches, 41/41.

## Test commands and results

```
PYTHONPATH=. ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_e2e_loop_mock.py -q
37 passed in 15.35s   (CPU only; includes a checkpoint-tokenizer test, skipped if absent)
PYTHONPATH=. ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo --mock \
  --mode strict/strict --steps 3 --prompts-per-step 4 --samples-per-prompt 4 --max-tokens 24
done: 3 steps, bitwise mismatches 0/1079, max |d| 0.000e+00, first drift None, weight sync 4/4
```

## Commands for the integrator (real engines; one 8-GPU node; needs a GPU lease and GPUs 6-7)

Run once per MODE in native/native, strict/native, native/strict, strict/strict:
```
PYTHONPATH=. ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \
  --rollout-tp 2 --rollout-gpus 0,1 --train-tp 4 --train-gpus 4,5,6,7 \
  --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \
  --lora-rank 16 --lr 1e-5 --seed 1234 --mode MODE --run-dir runs/poc_grpo/MODE-with-dash
```
Add `--sync-via-file` for a separate-process trainer. Compare `summary.json` (`drift_totals`,
`first_drift`, `reward`, `timings_mean_s`) and `rounds.csv` across the four run dirs.

## Interface requirements for the siblings (beyond the signatures)

1. `generate` logprobs: raw (temperature 1) fp32 selected-token logprobs of completion tokens,
   as fp32-exact Python floats (`tensor.float().tolist()`), one per generated token incl. EOS.
2. EOS = both ids in `generation_config.json` (200001, 200008 `<|eot|>`); `<|eom|>` (200007)
   must not stop generation.
3. `token_logprobs`: fp32 tensors with grad, length `len(ids) - completion_start`.
4. `export_lora()` hash == `lora_sha256`; `load_lora` verifies; `provenance()["lora_sha256"]`.
5. Prompts come from `data.HFChatTokenizer` (checkpoint template, system + user); with an
   explicit system message the template injects no date, so prompt ids are day-stable.

## Open issues

* Real run not executed (engines absent). Trainer multi-process design (4 Megatron ranks) is
  the sibling's; the loop supports in-memory or file transport.
* Strict mock is slow (Python K loops, ~3 s/step); irrelevant to the real engines.
* Mock sampled tokens coincide across modes (same seed), so per-mode rewards are identical.

## RFC feedback

* `final_e2e_lora`: say "rollout TP2 and trainer TP4 on one node; LoRA synced before the first
  rollout and after every step, hash-verified; report the four-mode matrix train/rollout in
  {native, strict}^2 with bitwise mismatch count, max |d|, first drift, reward, per-phase time".
  Define "first drift" operationally as the first (step, sample, token) with a bitwise mismatch
  between trainer and rollout logprobs on identical tokens under an identical LoRA hash, and pair
  it with the rollout-internal decode-vs-prefill check, which tells whether drift is inside the
  rollout engine or between engines.
* `museglimmer_ablation`: add "LoRA bf16 storage vs fp32 master" next to merged/unmerged LoRA.
* Experiments: the strict/strict criterion is `bitwise_equal`; ratio exactly 1 and clip fraction
  exactly 0 are free consistency checks, so report `ratio_exactly_one_frac`.

## Time spent

About 4 h.

# REPORT — `final_e2e_lora`: GRPO LoRA loop and cross-environment trainer bridge

Row `final_e2e_lora` of RFC #501 (Muse-Glimmer-30B strict train/rollout POC). Owner files:
`rl_engine/models/muse_glimmer/e2e/*`, `tests/models/muse_glimmer/test_e2e_loop_mock.py`,
`tests/models/muse_glimmer/test_e2e_bridge*.py`, `reports/muse_glimmer_poc/data/e2e_bridge.json`.

STATUS: the orchestrator, trainer bridge, timeline and tests are complete. The review wave B
fixes are in (13 issues; verdicts below). The integrator ran two one-step smokes with the REAL
30B engines through the bridge, on the merged code (commit c3650ed):

* strict/strict: 0/128 bitwise drift, weight sync 2/2.
* native/native: 57/128 drift, weight sync 2/2.

The four-mode 8-step GRPO runs are NOT RUN; their commands are in "Commands for the
integrator".

## Summary

* **Loop** (`loop.py`): one orchestrator process in the vLLM env (no VIME, no Ray, no
  `datasets`). It syncs the LoRA before rollout 0, then every step does:
  1. rollout `generate` (fp32 logprobs at sampling time);
  2. GSM8K exact-match reward;
  3. trainer `token_logprobs` on the same token ids;
  4. drift metrics and sidecar;
  5. rollout prefill re-score (decode-vs-prefill check);
  6. GRPO loss, then trainer `step`;
  7. `export_lora` (hash recomputed), `load_lora` and the `weight_sync_identity` check.

  Each step is logged to JSON and CSV with a per-phase timeline. The engine protocols are in
  `engines.py`; the CPU mocks implement the same surface.
* **Bridge** (`bridge.py` + `trainer_server.py`): vLLM (torch cu130) and Megatron Bridge (torch
  cu129) cannot share a process. The trainer therefore runs in a server started with
  `~/envs/megatron/bin/python`, and `RemoteTrainer` is a drop-in `TrainerLike`. The bridge
  touches the numerics in one place only: the gradient hand-off `S = sum_i (lp_i * g_i).sum()`.
  That hand-off is bit-exact (tested on CPU and CUDA, subnormals and signed zeros included).
* **Real 30B smokes** (integrator). Setup: rollout TP1 on GPU 4, trainer TP2 on GPUs 5,6,
  process launcher, 1 step x 2 prompts x 2 samples x 32 tokens.
  * strict/strict: 0/128 bitwise mismatches, and 0/128 on the rollout's decode-vs-prefill
    check. The ratio is exactly 1 on every token, weight sync passed 2/2, and a step took
    44.6 s.
  * native/native: 57/128 mismatches (max |d| 4.3e-4), with first drift at step 0, sample 0,
    token 3. The rollout's own decode-vs-prefill check already differs on 45/128 tokens. Weight
    sync passed 2/2, and a step took 17.8 s.
  * Every reward was 0, so the gradient was 0 and the LoRA hash did not change. Both sync
    checks therefore compared the same hash.
* **Wave B fixes**:
  * `rollout_lora_sha256` accepts only the top-level `provenance()["lora_sha256"]`.
  * Under torchrun, every rank's LoRA hash and logprob digest is gathered, and any difference
    raises.
  * The trainer builds before the client connects. Mock with a 6 s build and 9 s of caller
    work: `wait_ready` blocks 0.26 s now, against 9.30 s before.
  * A failed stateful op retires the bridge.
  * `requires_grad` is mirrored from the server.
  * Sync records name both hops (`bridge+memory`).
  * The Nagle on/off comparison is raw measured rows.
  * GPU placement defaults to the lease.
* **Tests**: 66 CPU tests pass (`test_e2e_loop_mock.py` + `test_e2e_bridge.py`). 12 of them
  were added or changed in wave B, and those failed on the pre-fix code. 3 GPU tests pass on
  one A100 (`test_e2e_bridge_gpu.py`).
* **Data**: `~/data/gsm8k/{train,test}.jsonl` (7473 / 1319 rows; downloaded with curl into a
  fresh directory and parsed as data only).

## Review wave B — issue verdicts

| id | sev | verdict | change (test: failed before -> passes after) |
|---|---|---|---|
| R1 | should_fix | fixed | This REPORT was rewritten to match the code (bridge, server, timeline, tests, smokes, commands). |
| R2 | should_fix | fixed with the integrator's real runs | `bench_bridge --import-e2e-run` stores both real 30B smokes through the bridge under `real_e2e_smoke`: raw per-step series, server-log timestamps and the trainer's StrictTrainer TP2 numbers. The `NOT RUN` stubs are removed. This row ran no 30B job itself; the wave B brief excludes that. |
| R3 | should_fix | fixed | `trainer_server.main` builds the trainer before accepting the client (the accept runs in a background thread). `hello["startup"]` records launch / build / connect / ready. Test `test_trainer_builds_while_the_caller_starts_the_rollout` (before: `wait_ready blocked 5.39s`). Bench `startup_overlap.{before_fix,after_fix}`. |
| R5 | should_fix | fixed | The server gathers every rank's LoRA hash and logprob digest and fails the reply on any difference. The client re-checks with `check_rank_hashes` and raises `RemoteWeightSyncError`. Tests `test_torchrun_rank_divergence_fails_closed[diverge_lora-…, diverge_logprobs-…]` (before: DID NOT RAISE) and `test_rank_hash_check_fails_closed`. |
| R6 | nit | fixed | `token_logprobs` applies the server's `requires_grad` flags, and `step` refuses logprobs without autograd history. Test `test_forward_only_trainer_is_rejected_at_scoring` (before: `assert not True`). |
| R7 | nit | fixed | The GPU test now asserts bit equality for the megatron-env variant as well. Rerun on GPU 7: 3 passed. |
| R8 | nit | fixed | The row is renamed `megatron_env_torchrun_2rank_replicated_control_plane` and its `note` states that there is no TP sharding. |
| R9 | nit | fixed | A remote failure of `token_logprobs`, `step` or `export_lora` kills the server group and refuses further calls. Stateless ops (`ping`, `provenance`, unknown op) stay retryable. Test `test_remote_exception_in_a_stateful_op_retires_the_bridge` (before: DID NOT RAISE). |
| R10 | nit | documented; engine hook requested | Profile entries record `rollout_engine_profiler` and the scope "orchestrator process only". The `StrictRolloutEngine.start_profile/stop_profile` hook belongs to the vLLM owner's file (see requests). |
| R11 | nit | not fixed: needs files outside this row | The fix touches `runtime/megatron_lora.py`, `vllm_strict.py`, `run_vllm_strict.py` and `strict_rollout_engine.py`, plus a new shared `muse_glimmer/lora_hash.py`; it is a request to the integrator. In the E2E path the bf16 guard holds: `export_lora_verified` and `StrictRolloutEngine.load_lora` call `validate_lora_state_dict(require_bf16=True)`. |
| R12 | nit | fixed | New `RemoteTrainer(tcp_nodelay=False)`. `transport` now holds raw interleaved rows for every repeat with Nagle on and off, and the hand-copied `transport_before_tcp_nodelay` is deleted. Sync records carry `trainer_to_orchestrator` and `orchestrator_to_rollout`, and `transport` is `bridge+memory` when bridged. Tests: the CLI bridge test (before: `'memory' == 'bridge+memory'`), `test_nagle_variant_is_selectable_and_bit_exact` and `test_sync_record_names_both_hops`. |
| R13 | nit | fixed | `--rollout-gpus` and `--train-gpus` default to the lease, in order (`bridge.default_gpu_ids`), so `gpu-lease -n 3` needs no ids. Test `test_gpu_ids_default_to_the_lease`. |

The issue list has no R4. Task item outside the list: `e2e/sync.rollout_lora_sha256` accepts
ONLY the rollout's top-level `provenance()["lora_sha256"]`, and the per-worker fallback is gone.
The vLLM engine recomputes that value on every TP worker and reports `None` when the workers
disagree. Test `test_rollout_hash_only_from_the_engine_level_provenance_key` (before: a
per-worker-only provenance was accepted, `'39f8…' is None`).

## Architecture

```
orchestrator process (~/envs/vllm; CUDA_VISIBLE_DEVICES = rollout GPUs)
  run_poc_grpo.build_engines
    1. RemoteTrainer(wait=False): trainer server (~/envs/megatron; CVD = train GPUs) starts and
       builds StrictTrainer at once (TP>1: StrictTrainer spawns its own NCCL worker ranks)
    2. StrictRolloutEngine (vLLM 0.31; EngineCore/worker subprocesses on the rollout GPUs)
    3. trainer.wait_ready(): connect (HMAC both ways) -> hello {build_s, startup, versions}
  loop.run_loop
    export_lora_verified(trainer) -> transfer_lora(rollout)          # sync BEFORE rollout 0
    for step:
      prompts = task.sample(P)                                       # GSM8K chat template
      groups  = rollout.generate(ids, n=S, max_tokens, T)            # tokens + fp32 logprobs
      rewards = task.reward(completion)                              # exact "#### <n>"
      lp_tr   = trainer.token_logprobs(prompt+completion, starts)    # RPC -> fp32 CPU leaves
      drift   = compare_logprobs(lp_tr, lp_rollout); sidecar step{n:04d}.pt
      recheck = rollout.logprobs(same ids) vs generate-time lp       # decode-vs-prefill
      loss    = grpo_token_loss(lp_tr, lp_rollout, advantages)       # fp32, clip 0.2
      trainer.step(loss)     # client loss.backward() -> g_i; server step(sum_i lp_i * g_i)
      pkg = export_lora_verified(trainer)  # trainer == server == client == every rank's hash
      transfer_lora(pkg, rollout)          # load_lora; provenance()["lora_sha256"] == pkg
      steps.jsonl, rounds.csv, timeline
    summary.json, README.md, provenance.json, config.json
```

`--mode <train>/<rollout>` sets each side to `strict` (RFC #501 strict ops) or `native` (vLLM /
Megatron Bridge kernels). `--sync-via-file` moves the LoRA through `run_dir/lora/step{n:04d}.pt`
and re-verifies the hash on load.

Files: `engines.py` (protocols), `data.py` (GSM8K, chat tokenizer), `grpo.py`, `drift.py`,
`sync.py`, `loop.py`, `timeline.py`, `bridge.py`, `trainer_server.py`, `mocks.py`,
`run_poc_grpo.py`, `bench_bridge.py`.

## Trainer bridge (`bridge.py`, `trainer_server.py`)

* **Transport**: `multiprocessing.connection` over 127.0.0.1 with a random 32-byte authkey.
  * The HMAC challenge runs both ways. The key is passed in the child's environment, never on
    the command line.
  * Each message is one `torch.save` blob, loaded with `weights_only=True`, and tensors travel
    as raw bytes.
  * `TCP_NODELAY` is set on both ends: Nagle plus delayed ACK stalls 16 KiB messages by about
    80 ms (see transport).
* **Gradient hand-off**: `token_logprobs` returns fresh CPU leaf tensors with the trainer's fp32
  values, and `requires_grad` is mirrored from the server. `step(loss)` runs `loss.backward()`
  in the orchestrator and ships `g_i = lp_i.grad`. The server then calls `trainer.step(S)` with
  `S = sum_i (lp_i_server * g_i).sum()`. The gradient of S with respect to `lp_i_server` is
  exactly `g_i`. S is not the loss, so `step` reports the orchestrator's loss.
* **Launchers**:
  * `process` (default): today's `StrictTrainer` spawns its own TP workers, and rank agreement
    inside it is the trainer's responsibility (its LoRA export gathers to every rank).
  * `torchrun`: one server per rank; rank 0 broadcasts every request on a gloo group and
    gathers per-rank status.
* **Start-up overlap (R3)**: rank 0 publishes the handshake file and accepts the client in a
  background thread, while every rank builds its trainer immediately. `run_poc_grpo` starts the
  server with `wait=False` and builds vLLM in the meantime. `wait_ready` then connects and
  receives the hello. `provenance.trainer_start.bridge.hello.startup` records launch -> build
  start -> build end -> client connected -> `wait_ready` called -> ready.
* **Rank agreement (R5)**: on `export_lora` every rank recomputes the canonical hash of the LoRA
  it exported, and on `token_logprobs` every rank hashes its logprob bytes with sha256.
  * Rank 0 fails the reply on any difference (`error_type` is `WeightSyncError` or
    `RankDivergenceError`).
  * The client requires `rank_sha256` to hold one hash per rank, each equal to its own
    recomputation (`check_rank_hashes`).
  * The trainer's reported hash, the server's recomputation and the client's recomputation
    must also agree.
* **Failure handling**: every call has a deadline, and the client polls both the socket and the
  server process.
  * A dead server raises `RemoteTrainerError` with the exit status and the log tail.
  * A missed deadline raises `RemoteTrainerTimeout` and kills the server's process group (the
    server runs in its own session).
  * A remote exception in a stateful op kills the group and retires the bridge (R9).
  * The server kills its own group when the client process disappears.
* **Placement (R13)**: GPU ids are physical.
  * If the flags are omitted, the rollout gets the first `--rollout-tp` leased ids and the
    trainer the next `--train-tp`.
  * `plan_gpu_placement` checks one id per rank, no overlap, and that every id is inside the
    lease.
  * The trainer server's `CUDA_VISIBLE_DEVICES` is the train ids (`""` for a CPU server). The
    orchestrator sets its own `CUDA_VISIBLE_DEVICES` to the rollout ids before vLLM starts.

## Timeline and profiling (`timeline.py`)

* **Per-step timing**: each step records the wall seconds of `rollout_generate`, `reward`,
  `trainer_logprobs`, `drift_metrics`, `rollout_recheck`, `sidecar`, `grpo_loss`,
  `backward_step`, `lora_export`, `lora_transfer_load` and `other`. It also records the bridge
  call stats: bytes, RTT and server compute.
* **Per-run series**: `summary.json["timeline"]` holds the per-phase series, means, totals and
  fractions of step time, plus tokens per second.
* **Profiling (`--profile-step k`)**: writes `torch.profiler` top-ops tables for trainer scoring
  and step (on the trainer server when bridged). It also writes the rollout call as seen by the
  orchestrator process. That rollout table contains no vLLM GPU kernels, because EngineCore and
  the workers are subprocesses. The `rollout_engine_profiler` field records whether the engine
  profiled itself (R10).
* **Code provenance**: `summary.json["code"]` records the commit and the dirty flag of the
  checkout the run used.

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
* **Hashed fields**: only the name, the dtype string, the shape and the raw bytes; no device,
  stride or grad.
* **Keys**: `layers.{i}.{q_proj|k_proj|v_proj|gate_proj|o_proj|mlp.gate_proj|mlp.up_proj|mlp.down_proj}.lora_{A,B}`,
  with A of shape `(r, K)` and B of shape `(N, r)`, bf16 (`sync.validate_lora_state_dict`).
  For the 30B at rank 16 that is 832 tensors and 209,584,128 bytes.
* **Known answer**: `{"layers.0.q_proj.lora_A": zeros(1, 2, bf16)}` hashes to
  `sha256(b"layers.0.q_proj.lora_A|torch.bfloat16|1,2\n" + 4 zero bytes)`.
* **Who checks it**: the trainer's reported hash, the bridge server's recomputation, the
  client's recomputation and (under torchrun) every rank's hash must all be equal. Next, the
  rollout's `load_lora` verifies the hash. Then its top-level `provenance()["lora_sha256"]` must
  equal it (`weight_sync_identity`); that key is the only one read.
* **Sync records**: each record names both hops, `trainer_to_orchestrator` (`in_process` or
  `bridge`) and `orchestrator_to_rollout` (`memory` or `file`). `transport` joins them, e.g.
  `bridge+memory`.
* **Runtime agreement**: `runtime/megatron_lora.lora_sha256` now delegates to this function.
  `hash_schema_check.all_equal` is true at 33a86ad (it was false at 0738241: F1).

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
`MockTrainer(build_delay_s)` simulates a slow trainer build (start-up overlap).
`FaultyMockTrainer` injects `exit`, `hang`, `raise`, `build`, `bad_hash`, `diverge_lora`,
`diverge_logprobs` or `forward_only`, optionally on one torchrun rank only.

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

## Tests

```
W=<worktree>; cd $W
PYTHONPATH=$W ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_e2e_bridge.py \
    tests/models/muse_glimmer/test_e2e_loop_mock.py -q -p no:cacheprovider
TBD-TESTS-TAIL
~/bin/gpu-lease -n 1 --pool 3,7 -- env PYTHONPATH=$W ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_e2e_bridge_gpu.py -q -p no:cacheprovider
[gpu-lease] using GPUs [7]
3 passed in 15.41s
```

The CPU tests start real trainer-server subprocesses in `~/envs/megatron`; the server runs on CPU
with `CUDA_VISIBLE_DEVICES=""`, and the client never initialises CUDA.

Before the wave B fixes, the same CPU command gave 12 failed / 54 passed. That run used the new
tests and the new mock fixtures against the old bridge, server, sync and loop (commit 87a4cd2;
log in the session scratchpad).

| test | before-fix failure |
|---|---|
| `test_trainer_builds_while_the_caller_starts_the_rollout` | `wait_ready blocked 5.39s: the build did not overlap` |
| `test_remote_exception_in_a_stateful_op_retires_the_bridge` | `DID NOT RAISE RemoteTrainerError` |
| `test_torchrun_rank_divergence_fails_closed[diverge_lora…]` | `DID NOT RAISE WeightSyncError` |
| `test_torchrun_rank_divergence_fails_closed[diverge_logprobs…]` | `DID NOT RAISE RemoteTrainerError` |
| `test_forward_only_trainer_is_rejected_at_scoring` | `assert not True` (requires_grad forced on) |
| `test_cli_mock_through_bridge_equals_in_process_and_writes_timeline` | `'memory' == 'bridge+memory'` |
| `test_rollout_hash_only_from_the_engine_level_provenance_key` | the per-worker hash was accepted (`'39f8…' is None`) |
| `test_sync_record_names_both_hops`, `test_sync_via_file_roundtrip_and_tamper` | `KeyError: 'trainer_to_orchestrator'` / `'orchestrator_to_rollout'` |
| `test_nagle_variant_is_selectable_and_bit_exact` | `unexpected keyword argument 'tcp_nodelay'` |
| `test_gpu_ids_default_to_the_lease`, `test_rank_hash_check_fails_closed` | new API (`default_gpu_ids`, `check_rank_hashes`) missing |

## Commands for the integrator — four modes, two concurrent runs on one node

Each run leases exactly its four GPUs: rollout TP2 + trainer TP2, process launcher. The four
modes run in two waves of two concurrent runs:

| wave | run 1: GPUs 0-3 (rollout 0,1 + trainer 2,3) | run 2: GPUs 4-7 (rollout 4,5 + trainer 6,7) |
|---|---|---|
| 1 | strict/strict | native/native |
| 2 | strict/native | native/strict |

Every run uses the same seed, so all four sample the same prompts. Run from `$W`: with
`python -m`, the current directory comes first on `sys.path`.

```
W=<checkout with this branch merged>; cd $W; mkdir -p runs/poc_grpo
COMMON="--trainer-python $HOME/envs/megatron/bin/python --trainer-launcher process \
  --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \
  --lora-rank 16 --lr 1e-5 --seed 1234"

# wave 1: start both, then wait for both
~/bin/gpu-lease -n 4 --pool 0-3 -- env PYTHONPATH=$W ~/envs/vllm/bin/python \
  -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \
  --rollout-tp 2 --rollout-gpus 0,1 --train-tp 2 --train-gpus 2,3 $COMMON \
  --mode strict/strict --run-dir runs/poc_grpo/strict-strict > runs/poc_grpo/strict-strict.log 2>&1 &
~/bin/gpu-lease -n 4 --pool 4-7 -- env PYTHONPATH=$W ~/envs/vllm/bin/python \
  -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \
  --rollout-tp 2 --rollout-gpus 4,5 --train-tp 2 --train-gpus 6,7 $COMMON \
  --mode native/native --run-dir runs/poc_grpo/native-native > runs/poc_grpo/native-native.log 2>&1 &
wait

# wave 2: the same two commands with
#   GPUs 0-3: --mode strict/native --run-dir runs/poc_grpo/strict-native  (log strict-native.log)
#   GPUs 4-7: --mode native/strict --run-dir runs/poc_grpo/native-strict  (log native-strict.log)

python3 reports/muse_glimmer_poc/tools/collect_e2e.py \
  runs/poc_grpo/strict-strict runs/poc_grpo/native-native \
  runs/poc_grpo/strict-native runs/poc_grpo/native-strict
PYTHONPATH=$W ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.bench_bridge \
  --import-e2e-run runs/poc_grpo/strict-strict runs/poc_grpo/native-native \
  runs/poc_grpo/strict-native runs/poc_grpo/native-strict
```

* **Placement flags**: optional inside a 4-GPU lease. `--rollout-tp 2 --train-tp 2` alone gives
  the first two leased ids to the rollout and the next two to the trainer.
* **Pass checks**:
  * strict/strict: `drift_totals.bitwise_mismatch_count == 0` and rollout decode-vs-prefill
    0.
  * Every mode: `weight_sync.passed == checks` (9 per run), `trainer_step.recompute_logp_mismatches == 0`
    in `steps.jsonl`, and the LoRA hash changes after any step with a nonzero reward.
* **Not yet exercised in any E2E run**: rollout TP2. The smokes used TP1; strict LoRA at TP2 is
  covered by the vLLM row's xrt dumps.
* **Run time**: NOT measured for 8 x 8 x 4 x 256. Each step generates up to 8,192 tokens,
  against 128 in the smokes.

## Interface requirements for the siblings (beyond the signatures)

1. **`generate` logprobs**: the raw (temperature 1) fp32 selected-token logprobs of the
   completion tokens. They are fp32-exact Python floats (`tensor.float().tolist()`), one per
   generated token, EOS included.
2. **EOS**: both ids in `generation_config.json` stop generation (200001, and 200008
   `<|eot|>`). `<|eom|>` (200007) must not stop generation.
3. **`token_logprobs`**: fp32 tensors of length `len(ids) - completion_start` with autograd
   history (the bridge mirrors `requires_grad`). Under torchrun, every rank must return the same
   bytes.
4. **LoRA hash**:
   * `export_lora()` returns the `lora_sha256` hash, and under torchrun every rank must export
     the same LoRA.
   * `load_lora` verifies the hash.
   * `provenance()["lora_sha256"]` must hold on every TP rank (`None` otherwise); it is the only
     value the loop reads.
5. **Prompts**: they come from `data.HFChatTokenizer` (checkpoint template, system + user). With
   an explicit system message the template injects no date, so prompt ids are day-stable.

## Open issues and requests

* **Integrator**: run the four modes (commands above).
* **R11 (integrator)**: create `rl_engine/models/muse_glimmer/lora_hash.py` holding
  `muse_glimmer_lora_sha256_v1`, and have `e2e/sync.py` re-export it. Then
  `runtime/megatron_lora.py`, `vllm_strict.py`, `run_vllm_strict.py` and
  `strict_rollout_engine.py` import it from there instead of from `e2e`. Keep the bf16 checks at
  export and import.
* **R10 (vLLM owner)**: add `start_profile()`/`stop_profile()` to `StrictRolloutEngine`
  (`LLM.start_profile` with `VLLM_TORCH_PROFILER_DIR`), so that `--profile-step` captures the
  EngineCore and worker GPU kernels.
* **F7 (Megatron owner)**: `StrictTrainer.close()` takes about 127 s at TP2, because the TP
  worker join times out after 120 s and the worker is then terminated. Fix: make the worker exit
  after the `close` broadcast and barrier.
* **F6**: the torchrun launcher is tested only with the mock trainer. `StrictTrainer` runs under
  the process launcher.
* **Mocks**: the strict mock is slow (Python K loops, a few s/step); this does not affect the
  real engines. Mock sampled tokens coincide across modes (same seed), so per-mode mock rewards
  are identical.

## RFC feedback

* **`final_e2e_lora`**: say that rollout and trainer may live in different Python environments
  (vLLM and Megatron here need different torch builds).
  * Specify a trainer bridge whose only numerical contract is an exact gradient hand-off: the
    surrogate `sum_i lp_i * g_i`, whose gradient with respect to `lp_i` is `g_i` bit for bit.
  * Give weight sync one canonical LoRA hash, checked on every trainer rank, on both ends of the
    socket and on every rollout rank.
  * Report the transport per hop.
* **Four-mode matrix**: report train/rollout in {native, strict}^2 with bitwise mismatch count,
  max |d|, first drift, reward and per-phase time.
  * Define first drift as the first (step, sample, token) with a bitwise mismatch between
    trainer and rollout logprobs, on identical tokens under an identical LoRA hash.
  * Pair it with the rollout's own decode-vs-prefill check. In the native/native smoke, the
    rollout's decode and prefill disagree on 45/128 tokens, so most of the native drift sits
    inside vLLM rather than between the engines.
* **Smokes**: include a nonzero-reward step (or log `grad_norm > 0` as its own gate). With all
  rewards 0, the weight-sync check only ever sees an unchanged LoRA.
* **`museglimmer_ablation`**: add "LoRA bf16 storage vs fp32 master" next to merged/unmerged
  LoRA.
* **Experiments**: the strict/strict criterion is `bitwise_equal`. Ratio exactly 1 and clip
  fraction exactly 0 come for free as consistency checks, so report `ratio_exactly_one_frac`.

## Time spent

About 4 h for the original loop, about 4 h for the bridge and timeline (wave A), and about 3 h
for wave B.

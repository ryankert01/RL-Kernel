# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU tests for the Muse-Glimmer-30B strict POC GRPO loop (row ``final_e2e_lora``) in mock mode.

Run with ``PYTHONPATH=<worktree> ~/envs/vllm/bin/python -m pytest
tests/models/muse_glimmer/test_e2e_loop_mock.py -q``. No GPU, no network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence

import pytest
import torch

from rl_engine.models.muse_glimmer.e2e import data, drift, grpo, loop, mocks, run_poc_grpo, sync
from rl_engine.models.muse_glimmer.e2e.engines import RolloutSample, coerce_samples

pytestmark = pytest.mark.unit

GSM8K_TRAIN = Path("~/data/gsm8k/train.jsonl").expanduser()
CHECKPOINT = Path("~/models/Muse-Glimmer-30B").expanduser()


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(4, old))
    yield
    torch.set_num_threads(old)


class FirstTokenParityTask:
    """Deterministic mock task whose reward splits groups: 1.0 iff the first token is even."""

    name = "first-token-parity"

    def __init__(self) -> None:
        self.inner = data.SyntheticArithmeticTask(mocks.ByteTokenizer())

    def sample(self, rng: random.Random, count: int) -> list[data.Prompt]:
        return self.inner.sample(rng, count)

    def reward(self, completion_ids: Sequence[int], prompt: data.Prompt) -> float:
        return 1.0 if int(completion_ids[0]) % 2 == 0 else 0.0

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.inner.decode(token_ids)

    def describe(self) -> dict:
        return {"task": self.name}


def _engines(
    mode: str, *, perturb: float = 0.0, lora_rank: int = 4, lr: float = 1e-3, seed: int = 7
):
    train_strict, rollout_strict = loop.parse_mode(mode)
    trainer = mocks.MockTrainer(
        "mock", tensor_parallel_size=4, gpu_ids=[4, 5, 6, 7], lora_rank=lora_rank, lr=lr,
        strict=train_strict, seed=seed,
    )
    rollout = mocks.MockRolloutEngine(
        "mock", tensor_parallel_size=2, gpu_ids=[0, 1], strict=rollout_strict, max_model_len=1024,
        seed=seed, lora_rank=lora_rank, perturb=perturb,
    )
    return rollout, trainer


def _config(tmp_path: Path, mode: str, **overrides) -> loop.LoopConfig:
    kwargs = dict(
        run_dir=tmp_path / mode.replace("/", "-"), steps=3, prompts_per_step=2,
        samples_per_prompt=4, max_tokens=10, lora_rank=4, lr=1e-3, mode=mode, mock=True,
        task="first-token-parity",
    )
    kwargs.update(overrides)
    return loop.LoopConfig(**kwargs)


def _run(tmp_path: Path, mode: str, *, perturb: float = 0.0, **overrides):
    cfg = _config(tmp_path, mode, **overrides)
    rollout, trainer = _engines(mode, perturb=perturb, lora_rank=cfg.lora_rank, lr=cfg.lr)
    summary = loop.run_loop(cfg, FirstTokenParityTask(), rollout, trainer, log=None)
    records = [json.loads(line) for line in (cfg.run_dir / "steps.jsonl").read_text().splitlines()]
    return cfg, summary, records


# ------------------------------------------------------------------------------------------
# The loop in mock mode
# ------------------------------------------------------------------------------------------


def test_loop_three_steps_strict_strict_zero_drift(tmp_path: Path) -> None:
    cfg, summary, records = _run(tmp_path, "strict/strict")
    assert summary["status"] == "ok"
    assert summary["steps_completed"] == 3
    assert summary["first_drift"] is None
    assert summary["drift_totals"]["all_steps_bitwise_equal"]
    assert summary["drift_totals"]["token_count"] > 0
    assert summary["weight_sync"]["all_passed"] and summary["weight_sync"]["checks"] == 4
    for record in records:
        assert record["drift"]["bitwise_mismatch_count"] == 0
        assert record["drift"]["max_abs_diff"] == 0.0
        assert record["grpo"]["ratio_exactly_one_frac"] == 1.0
        assert record["grpo"]["clip_frac"] == 0.0
        assert record["rollout_decode_vs_prefill"]["bitwise_mismatch_count"] == 0
        assert record["sync"]["identity"] is True
    # The parity reward splits at least one group, so the LoRA actually moves.
    assert any(r["trainer_step"]["grad_norm"] > 0 for r in records)
    assert summary["weight_sync"]["initial_sha256"] != summary["weight_sync"]["final_sha256"]
    expected_files = (
        "steps.jsonl", "rounds.csv", "summary.json", "README.md", "provenance.json", "config.json"
    )
    for name in expected_files:
        assert (cfg.run_dir / name).exists(), name
    assert sorted(p.name for p in (cfg.run_dir / "mismatch-sidecars").glob("*.pt")) == [
        "step0000.pt", "step0001.pt", "step0002.pt",
    ]
    with (cfg.run_dir / "rounds.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0", "1", "2"]
    assert all(row["bitwise_mismatch_count"] == "0" for row in rows)
    readme = (cfg.run_dir / "README.md").read_text()
    assert "strict/strict" in readme and "none (bitwise equal at every step)" in readme


def test_perturbed_rollout_gives_nonzero_drift(tmp_path: Path) -> None:
    _, summary, records = _run(tmp_path, "strict/strict", perturb=1e-2, steps=2)
    assert summary["first_drift"] is not None and summary["first_drift"]["step"] == 0
    assert records[0]["drift"]["bitwise_mismatch_count"] > 0
    assert records[0]["drift"]["max_abs_diff"] > 0.0
    assert records[0]["drift"]["first_mismatch"] is not None
    # The perturbation lives in the rollout weights, so decode and prefill still agree there.
    assert records[0]["rollout_decode_vs_prefill"]["bitwise_mismatch_count"] == 0


@pytest.mark.parametrize("mode", ["native/native", "strict/native", "native/strict"])
def test_mixed_and_native_modes_drift(tmp_path: Path, mode: str) -> None:
    _, summary, records = _run(tmp_path, mode, steps=1)
    assert summary["first_drift"] is not None
    assert records[0]["drift"]["bitwise_mismatch_count"] > 0
    assert records[0]["drift"]["value_mismatch_count"] > 0
    assert records[0]["grpo"]["ratio_exactly_one_frac"] < 1.0
    internal = records[0]["rollout_decode_vs_prefill"]["bitwise_mismatch_count"]
    if mode.endswith("/native"):
        assert internal > 0, "native decode must differ from native prefill in the mock"
    else:
        assert internal == 0


def test_sidecar_recompute_matches_logged(tmp_path: Path) -> None:
    cfg, _, records = _run(tmp_path, "strict/native", steps=2)
    for record in records:
        sidecar = cfg.run_dir / "mismatch-sidecars" / f"step{record['step']:04d}.pt"
        payload = drift.load_sidecar(sidecar)
        recomputed = drift.metrics_from_sidecar(payload).to_dict()
        assert recomputed == record["drift"]
        assert payload["response_lengths"] == [len(t) for t in payload["train_log_probs"]]
        assert payload["extra"]["rewards"] == record["rewards"]


def test_sync_via_file_roundtrip_and_tamper(tmp_path: Path) -> None:
    cfg, summary, _ = _run(tmp_path, "strict/strict", steps=1, sync_via_file=True)
    assert summary["weight_sync"]["transport"] == "file"
    files = sorted((cfg.run_dir / "lora").glob("*.pt"))
    assert [f.name for f in files] == ["step0000.pt", "step0001.pt"]
    package = sync.load_lora_package(files[1])
    assert package.sha256 == summary["weight_sync"]["final_sha256"]
    payload = torch.load(files[1], weights_only=True)
    key = sorted(payload["state_dict"])[0]
    payload["state_dict"][key] = payload["state_dict"][key] + 1.0
    torch.save(payload, files[1])
    with pytest.raises(sync.WeightSyncError):
        sync.load_lora_package(files[1])


def test_cli_main_mock(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli"
    code = run_poc_grpo.main(
        [
            "--mock", "--mode", "strict/strict", "--steps", "2", "--prompts-per-step", "2",
            "--samples-per-prompt", "2", "--max-tokens", "8", "--lora-rank", "4", "--run-dir",
            str(run_dir), "--threads", "2",
        ]
    )
    assert code == 0
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["steps_completed"] == 2
    assert summary["first_drift"] is None
    assert summary["task"]["task"] == "synthetic-arith"


# ------------------------------------------------------------------------------------------
# Mock numerics
# ------------------------------------------------------------------------------------------


def test_strict_primitives_shape_invariant_and_accurate() -> None:
    x = torch.linspace(-30.0, 30.0, 1001)
    batched = mocks.strict_exp(x)
    single = torch.stack([mocks.strict_exp(x[i : i + 1])[0] for i in range(x.numel())])
    assert torch.equal(batched, single)
    assert torch.allclose(batched, torch.exp(x), rtol=1e-6, atol=0.0)
    y = torch.linspace(1e-3, 1e3, 1001)
    single_log = torch.stack([mocks.strict_log(y[i : i + 1])[0] for i in range(1001)])
    assert torch.equal(mocks.strict_log(y), single_log)
    assert torch.allclose(mocks.strict_log(y), torch.log(y), rtol=0.0, atol=1e-6)
    weight = torch.randn(7, 5)
    xs = torch.randn(3, 4, 5)
    full = mocks.strict_linear(xs, weight)
    rows = torch.stack([mocks.strict_linear(xs[i], weight) for i in range(3)])
    assert torch.equal(full, rows)
    assert torch.allclose(full, xs @ weight.T, rtol=1e-5, atol=1e-6)


def test_strict_engines_batch_and_phase_invariance() -> None:
    rollout, trainer = _engines("strict/strict")
    package = sync.export_lora_verified(trainer, step=0)
    sync.transfer_lora(package, rollout)
    tok = mocks.ByteTokenizer()
    prompts = [tok.encode_chat([{"role": "user", "content": f"{i} + {i}"}]) for i in range(3)]
    groups = rollout.generate(prompts, n=2, max_tokens=6, temperature=1.0)
    seqs = [p + s.token_ids for p, g in zip(prompts, groups) for s in g]
    starts = [len(p) for p, g in zip(prompts, groups) for s in g]
    sampled = [s.logprobs for g in groups for s in g]
    batched = [t.detach() for t in trainer.token_logprobs(seqs, starts)]
    single = [trainer.token_logprobs([s], [c])[0].detach() for s, c in zip(seqs, starts)]
    assert drift.compare_logprobs(batched, single).bitwise_equal
    assert drift.compare_logprobs(batched, sampled).bitwise_equal
    assert drift.compare_logprobs(rollout.logprobs(seqs, starts), sampled).bitwise_equal
    assert all(len(s.logprobs) == len(s.token_ids) for g in groups for s in g)
    with_grad = trainer.token_logprobs(seqs, starts)
    assert all(t.requires_grad and t.dtype == torch.float32 for t in with_grad)


def test_byte_tokenizer_roundtrip() -> None:
    tok = mocks.ByteTokenizer()
    ids = tok.encode_chat([{"role": "user", "content": "héllo"}])
    assert ids[0] == mocks.ByteTokenizer.BOS and tok.decode(ids) == "<user>héllo\n<assistant>"
    assert tok.decode([mocks.ByteTokenizer.EOS, 65]) == "A"


def test_coerce_samples_accepts_duck_types() -> None:
    class Sample:
        token_ids = (1, 2)
        logprobs = (-0.5, -0.25)
        finish_reason = "stop"

    groups = coerce_samples([[Sample(), {"token_ids": [3], "logprobs": [-1.0]}]])
    assert groups[0][0] == RolloutSample([1, 2], [-0.5, -0.25], "stop")
    assert groups[0][1].finish_reason == "unknown"
    with pytest.raises(ValueError):
        coerce_samples([[{"token_ids": [1, 2], "logprobs": [-1.0]}]])


# ------------------------------------------------------------------------------------------
# lora_sha256 / weight sync
# ------------------------------------------------------------------------------------------


def _reference_lora_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Second, independent transcription of the hash definition from sync.py's docstring."""

    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name].detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{','.join(str(d) for d in t.shape)}\n".encode("utf-8"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _lora_dict() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    out = {}
    for i in range(2):
        for proj in sync.LORA_PROJECTIONS:
            out[f"layers.{i}.{proj}.lora_A"] = torch.randn(4, 6, generator=gen).to(torch.bfloat16)
            out[f"layers.{i}.{proj}.lora_B"] = torch.randn(5, 4, generator=gen).to(torch.bfloat16)
    return out


def test_lora_sha256_known_answer_stability_and_sensitivity() -> None:
    sd = _lora_dict()
    digest = sync.lora_sha256(sd)
    assert digest == _reference_lora_sha256(sd)
    reordered = dict(reversed(list(sd.items())))
    assert digest == sync.lora_sha256(reordered)  # insertion order irrelevant
    assert digest == sync.lora_sha256({k: v.clone() for k, v in sd.items()})
    transposed_twice = {k: v.T.contiguous().T for k, v in sd.items()}
    assert digest == sync.lora_sha256(transposed_twice)  # strides irrelevant
    tiny = {"layers.0.q_proj.lora_A": torch.zeros(1, 2, dtype=torch.bfloat16)}
    expected = hashlib.sha256(b"layers.0.q_proj.lora_A|torch.bfloat16|1,2\n" + bytes(4)).hexdigest()
    assert sync.lora_sha256(tiny) == expected
    flipped = {k: v.clone() for k, v in sd.items()}
    key = "layers.1.mlp.down_proj.lora_B"
    raw = flipped[key].view(torch.int16)
    raw[0, 0] ^= 1  # flip the lowest mantissa bit of one element
    assert sync.lora_sha256(flipped) != digest
    assert sync.lora_sha256({**sd, key: sd[key].reshape(4, 5)}) != digest  # shape in the stream
    assert sync.lora_sha256({**sd, key: sd[key].to(torch.float32)}) != digest  # dtype in the stream
    renamed = dict(sd)
    renamed["layers.1.mlp.down_proj.lora_a"] = renamed.pop(key)
    assert sync.lora_sha256(renamed) != digest
    assert sync.validate_lora_state_dict(sd)["ranks"] == [4]


def test_validate_lora_state_dict_schema() -> None:
    sd = _lora_dict()
    q_a = sd["layers.0.q_proj.lora_A"]
    with pytest.raises(sync.WeightSyncError):
        sync.validate_lora_state_dict({**sd, "layers.0.lm_head.lora_A": q_a})
    with pytest.raises(sync.WeightSyncError):
        sync.validate_lora_state_dict({**sd, "layers.0.q_proj.lora_A": q_a.float()})
    bad = dict(sd)
    bad["layers.0.q_proj.lora_B"] = torch.zeros(5, 3, dtype=torch.bfloat16)
    with pytest.raises(sync.WeightSyncError):
        sync.validate_lora_state_dict(bad)
    with pytest.raises(sync.WeightSyncError):
        sync.validate_lora_state_dict({})


def test_load_lora_rejects_wrong_hash_and_lying_rollout() -> None:
    rollout, trainer = _engines("strict/strict")
    sd, sha = trainer.export_lora()
    with pytest.raises(sync.WeightSyncError):
        rollout.load_lora(sd, "0" * 64)
    assert rollout.provenance()["lora_sha256"] is None

    class LyingRollout:
        def load_lora(self, state_dict, sha256):
            return None

        def provenance(self):
            return {"lora_sha256": "f" * 64}

    package = sync.export_lora_verified(trainer, step=0)
    with pytest.raises(sync.WeightSyncError):
        sync.transfer_lora(package, LyingRollout())

    class LyingTrainer:
        def export_lora(self):
            return sd, "e" * 64

    with pytest.raises(sync.WeightSyncError):
        sync.export_lora_verified(LyingTrainer(), step=0)
    record = sync.transfer_lora(package, rollout)
    assert record["identity"] and rollout.provenance()["lora_sha256"] == sha
    assert sync.weight_sync_identity(sha, sha) and not sync.weight_sync_identity(sha, None)


# ------------------------------------------------------------------------------------------
# GRPO
# ------------------------------------------------------------------------------------------


def test_group_advantages_zero_mean_and_uniform_groups() -> None:
    adv = grpo.group_normalized_advantages([1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 4)
    assert adv.dtype == torch.float32 and adv.shape == (16,)
    groups = adv.view(4, 4)
    assert torch.allclose(groups.sum(dim=1), torch.zeros(4), atol=1e-6)
    assert torch.allclose(groups[0], torch.tensor([1.0, -1.0, -1.0, 1.0]))
    assert torch.equal(groups[1], torch.zeros(4)) and torch.equal(groups[2], torch.zeros(4))
    assert torch.allclose(groups[3].pow(2).mean(), torch.tensor(1.0), atol=1e-5)
    with pytest.raises(ValueError):
        grpo.group_normalized_advantages([1.0, 0.0, 1.0], 2)


def test_grpo_zero_drift_ratio_one_and_no_clipping() -> None:
    train = [
        torch.tensor([-1.0, -2.0, -0.5], requires_grad=True),
        torch.tensor([-0.1, -3.0], requires_grad=True),
    ]
    rollout = [t.detach().clone() for t in train]
    adv = torch.tensor([1.0, -1.0])
    out = grpo.grpo_token_loss(train, rollout, adv, clip_eps=0.2)
    expected = -(1.0 * 3 + (-1.0) * 2) / 5
    assert math.isclose(out.loss.item(), expected, rel_tol=1e-6)
    assert out.stats["ratio_exactly_one_frac"] == 1.0 and out.stats["clip_frac"] == 0.0
    assert out.stats["active_tokens"] == 5 and out.stats["kl"] == 0.0
    out.loss.backward()
    # d(-rho*A)/d logp_train = -A at rho == 1: REINFORCE with group-normalized advantages.
    assert torch.allclose(train[0].grad, torch.full((3,), -1.0 / 5))
    assert torch.allclose(train[1].grad, torch.full((2,), 1.0 / 5))


def test_grpo_clipping_active_blocks_gradient() -> None:
    log_ratio = math.log(1.5)
    train = [
        torch.tensor([log_ratio], requires_grad=True),
        torch.tensor([log_ratio], requires_grad=True),
    ]
    rollout = [torch.zeros(1), torch.zeros(1)]
    out = grpo.grpo_token_loss(train, rollout, torch.tensor([1.0, -1.0]), clip_eps=0.2)
    assert out.stats["clip_frac"] == 1.0
    assert out.stats["clipped_selected_frac"] == 0.5
    out.loss.backward()
    assert train[0].grad.item() == 0.0  # A > 0 and rho > 1 + eps: clipped branch, zero gradient
    assert train[1].grad.item() != 0.0  # A < 0 and rho > 1 + eps: unclipped branch, gradient flows
    assert math.isclose(out.loss.item(), -(1.2 * 1.0 + 1.5 * -1.0) / 2, rel_tol=1e-6)


def test_grpo_optional_kl_term() -> None:
    train = [torch.tensor([-1.0, -1.5], requires_grad=True)]
    rollout = [torch.tensor([-1.0, -1.5])]
    adv = torch.tensor([0.0])
    ref_same = [torch.tensor([-1.0, -1.5])]
    same = grpo.grpo_token_loss(train, rollout, adv, kl_coef=0.1, logp_ref=ref_same)
    assert same.stats["kl"] == 0.0 and same.loss.item() == 0.0
    ref_moved = [torch.tensor([-0.5, -2.5])]
    moved = grpo.grpo_token_loss(train, rollout, adv, kl_coef=0.1, logp_ref=ref_moved)
    assert moved.stats["kl"] > 0.0 and moved.loss.item() > 0.0
    with pytest.raises(ValueError):
        grpo.grpo_token_loss(train, rollout, adv, kl_coef=0.1)
    with pytest.raises(ValueError):
        grpo.grpo_token_loss(train, [torch.tensor([-1.0])], adv)


# ------------------------------------------------------------------------------------------
# Drift metric on synthetic inputs
# ------------------------------------------------------------------------------------------


def test_compare_logprobs_bitwise_vs_value_and_first_index() -> None:
    a = [torch.tensor([-1.0, -2.0, 0.0]), torch.tensor([-0.5, -0.25])]
    b = [torch.tensor([-1.0, -2.0, -0.0]), torch.tensor([-0.5, -0.25 + 2**-10])]
    metrics = drift.compare_logprobs(a, b)
    assert metrics.value_mismatch_count == 1  # only the second sample's token differs in value
    assert metrics.bitwise_mismatch_count == 2  # -0.0 vs 0.0 differs bytewise
    assert metrics.first_mismatch == (0, 2) and metrics.first_mismatch_abs_diff == 0.0
    assert metrics.max_abs_diff == pytest.approx(2**-10)
    assert not metrics.torch_equal and not metrics.bitwise_equal
    masked = drift.compare_logprobs(a, b, masks=[[True, True, False], [True, False]])
    assert masked.bitwise_equal and masked.token_count == 3
    with pytest.raises(ValueError):
        drift.compare_logprobs([torch.tensor([-1.0])], [torch.tensor([-1.0, -2.0])])
    with pytest.raises(TypeError):
        drift.compare_logprobs([torch.tensor([-1.0], dtype=torch.float64)], [[-1.0]])
    assert drift.compare_logprobs([[-1.0, -2.0]], [torch.tensor([-1.0, -2.0])]).bitwise_equal


def test_parse_mode() -> None:
    assert loop.parse_mode("strict/native") == (True, False)
    assert loop.parse_mode("native/strict") == (False, True)
    assert loop.parse_mode("NATIVE/native") == (False, False)
    with pytest.raises(ValueError):
        loop.parse_mode("strict")


# ------------------------------------------------------------------------------------------
# Data / reward
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion, gold, expected",
    [
        ("so the answer is 72.\n#### 72", "72", 1.0),
        ("#### 1,000", "1000", 1.0),
        ("#### $5.50", "5.5", 1.0),
        ("#### 72.0", "72", 1.0),
        ("#### 71", "72", 0.0),
        ("the answer is 72", "72", 0.0),
        ("#### 10\nwait\n#### 12", "12", 1.0),
        ("#### 10\nwait\n#### 12", "10", 0.0),
        ("#### -3", "-3", 1.0),
        ("<|start|>assistant<|message|>#### 18 <|eot|>", "18", 1.0),
        ("#### 18 dollars", "18", 1.0),
        ("####", "18", 0.0),
    ],
)
def test_reward_exact_match(completion: str, gold: str, expected: float) -> None:
    assert data.reward(completion, gold) == expected


def test_normalize_and_gold_extraction() -> None:
    assert data.normalize_number(" 1,234.50 ") == "1234.5"
    assert data.normalize_number("$0.0") == "0"
    assert data.normalize_number("abc") is None
    assert data.extract_final_number("x\n#### 7\n") == "7"
    assert data.gsm8k_gold("... = <<48+24=72>>72 clips.\n#### 72") == "72"
    with pytest.raises(ValueError):
        data.gsm8k_gold("no final line")


def test_gsm8k_jsonl_parsing_and_task(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"question": "1 + 1?", "answer": "1+1=2\n#### 2"},
        {"question": "Price?", "answer": "<<3*4=12>>\n#### 1,200"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    examples = data.load_gsm8k(path)
    assert [e.gold for e in examples] == ["2", "1200"]
    assert data.load_gsm8k(path, limit=1)[0].question == "1 + 1?"
    task = data.GSM8KTask(examples, mocks.ByteTokenizer())
    prompts = task.sample(random.Random(0), 2)
    assert {p.gold for p in prompts} == {"2", "1200"}
    assert prompts[0].token_ids[0] == mocks.ByteTokenizer.BOS
    assert task.reward(list(b"#### 2"), next(p for p in prompts if p.gold == "2")) == 1.0
    assert task.reward(list(b"#### 3"), next(p for p in prompts if p.gold == "2")) == 0.0
    (tmp_path / "bad.jsonl").write_text('{"question": 1, "answer": "#### 2"}\n')
    with pytest.raises(ValueError):
        data.load_gsm8k(tmp_path / "bad.jsonl")
    (tmp_path / "notjson.jsonl").write_text("not json\n")
    with pytest.raises(json.JSONDecodeError):
        data.load_gsm8k(tmp_path / "notjson.jsonl")


@pytest.mark.skipif(not GSM8K_TRAIN.exists(), reason="~/data/gsm8k/train.jsonl not downloaded")
def test_gsm8k_real_file_parses() -> None:
    examples = data.load_gsm8k(GSM8K_TRAIN, limit=200)
    assert len(examples) == 200 and examples[0].gold == "72"
    assert all(data.normalize_number(e.gold) == e.gold for e in examples)


@pytest.mark.skipif(not (CHECKPOINT / "REVISION").exists(), reason="checkpoint not complete")
def test_hf_chat_tokenizer_prompt_shape() -> None:
    tokenizer = data.HFChatTokenizer(CHECKPOINT)
    assert tokenizer.eos_token_ids == (200001, 200008)
    ids = tokenizer.encode_chat(data.build_messages("What is 2 + 2?"))
    assert ids[0] == 200000  # <|begin_of_text|>
    assert ids[-2:] == [200022, 140680]  # <|start|>assistant (generation prompt)
    rendered = tokenizer.render_chat(data.build_messages("What is 2 + 2?"))
    assert "#### <number>" in rendered
    # With an explicit system message the template injects no date, so prompts are day-stable.
    assert "Current date" not in rendered
    assert tokenizer.encode_chat(data.build_messages("What is 2 + 2?")) == ids  # stable
    assert tokenizer.decode(ids[-2:]) == "<|start|>assistant"

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""GSM8K data, chat-template prompts and the verifiable exact-match reward.

* Data: ``{train,test}.jsonl`` from the openai/grade-school-math repository, fetched with
  ``curl`` into ``~/data/gsm8k/`` (a directory that holds nothing else) and parsed line by
  line with ``json.loads``. The files are untrusted input: they are read as data only.
* Prompt: system + user messages rendered with the checkpoint's chat template
  (``~/models/Muse-Glimmer-30B``, ``add_generation_prompt=True``). The template injects
  ``Current date`` from ``strftime_now`` unless ``current_date`` is passed, so the prompt
  builder pins ``current_date`` to keep token ids reproducible across days.
* Reward: ``reward(completion_text, gold) == 1.0`` iff the number on the last
  ``#### <number>`` line of the completion equals the gold number after stripping commas,
  ``$`` and whitespace and normalizing the decimal representation (``72.0 == 72``).
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, Sequence

GSM8K_BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
)
GSM8K_SPLITS = ("train", "test")
DEFAULT_DATA_DIR = Path("~/data/gsm8k").expanduser()
DEFAULT_MODEL_PATH = Path("~/models/Muse-Glimmer-30B").expanduser()
PROMPT_CURRENT_DATE = "2026-10-15"
SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your answer with a final line of the form '#### <number>' that contains only the "
    "final numeric answer, with no units, commas or currency symbols."
)
FINAL_ANSWER_RE = re.compile(r"####\s*([^\n]*)")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def download_gsm8k(
    data_dir: Path = DEFAULT_DATA_DIR,
    splits: Sequence[str] = GSM8K_SPLITS,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Fetch the jsonl splits with curl (only the missing ones unless ``force``)."""

    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in splits:
        if split not in GSM8K_SPLITS:
            raise ValueError(f"unknown GSM8K split {split!r}")
        path = data_dir / f"{split}.jsonl"
        if force or not path.exists():
            url = f"{GSM8K_BASE_URL}/{split}.jsonl"
            subprocess.run(
                ["curl", "-sSL", "--fail", "--retry", "3", "-o", str(path), url], check=True
            )
        paths[split] = path
    return paths


@dataclass(frozen=True)
class GSM8KExample:
    index: int
    question: str
    answer: str
    gold: str


def normalize_number(text: str | None) -> str | None:
    """Canonical decimal string of the first number in ``text`` (commas/$/spaces stripped)."""

    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("$", "").replace(" ", "")
    cleaned = cleaned.rstrip(".")
    match = NUMBER_RE.search(cleaned)
    if match is None:
        return None
    try:
        value = Decimal(match.group(0)).normalize()
    except InvalidOperation:
        return None
    if value == 0:
        return "0"
    return format(value, "f")


def extract_final_number(text: str) -> str | None:
    """Number on the LAST ``#### <number>`` line, normalized; ``None`` if absent."""

    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def gsm8k_gold(answer: str) -> str:
    gold = extract_final_number(answer)
    if gold is None:
        raise ValueError("GSM8K answer has no '#### <number>' line")
    return gold


def reward(completion_text: str, gold: str) -> float:
    """Exact match of the final number: 1.0 or 0.0."""

    predicted = extract_final_number(completion_text)
    expected = normalize_number(gold)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def load_gsm8k(path: Path, *, limit: int | None = None) -> list[GSM8KExample]:
    """Parse a GSM8K jsonl file as data; malformed lines raise."""

    examples: list[GSM8KExample] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"{path}:{line_no}: question/answer must be strings")
            examples.append(GSM8KExample(len(examples), question, answer, gsm8k_gold(answer)))
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def build_messages(question: str, *, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


class ChatTokenizer(Protocol):
    """What the tasks need from a tokenizer."""

    @property
    def eos_token_ids(self) -> tuple[int, ...]: ...

    @property
    def vocab_size(self) -> int: ...

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class HFChatTokenizer:
    """The checkpoint tokenizer + chat template (CPU only, lazy transformers import)."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        *,
        current_date: str = PROMPT_CURRENT_DATE,
    ) -> None:
        from transformers import AutoTokenizer  # lazy: heavy import

        self.model_path = Path(model_path).expanduser()
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._current_date = current_date
        self._eos = self._read_eos_ids()

    def _read_eos_ids(self) -> tuple[int, ...]:
        config = self.model_path / "generation_config.json"
        ids: list[int] = []
        if config.exists():
            payload = json.loads(config.read_text(encoding="utf-8"))
            raw = payload.get("eos_token_id")
            if isinstance(raw, int):
                ids = [raw]
            elif isinstance(raw, list):
                ids = [int(v) for v in raw]
        if not ids and self._tokenizer.eos_token_id is not None:
            ids = [int(self._tokenizer.eos_token_id)]
        return tuple(ids)

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos

    @property
    def vocab_size(self) -> int:
        return int(len(self._tokenizer))

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            current_date=self._current_date,
        )

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        text = self.render_chat(messages)
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return [int(t) for t in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode([int(t) for t in token_ids], skip_special_tokens=False)


@dataclass
class Prompt:
    index: int
    token_ids: list[int]
    gold: str
    text: str


class Task(Protocol):
    name: str

    def sample(self, rng: random.Random, count: int) -> list[Prompt]: ...

    def reward(self, completion_ids: Sequence[int], prompt: Prompt) -> float: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def describe(self) -> dict[str, Any]: ...


class GSM8KTask:
    """Sample GSM8K prompts and score completions with the exact-match reward."""

    name = "gsm8k"

    def __init__(
        self,
        examples: Sequence[GSM8KExample],
        tokenizer: ChatTokenizer,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        split: str = "train",
    ) -> None:
        if not examples:
            raise ValueError("no GSM8K examples")
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.split = split
        self._cache: dict[int, list[int]] = {}

    def encode(self, example: GSM8KExample) -> list[int]:
        ids = self._cache.get(example.index)
        if ids is None:
            messages = build_messages(example.question, system_prompt=self.system_prompt)
            ids = self.tokenizer.encode_chat(messages)
            self._cache[example.index] = ids
        return list(ids)

    def sample(self, rng: random.Random, count: int) -> list[Prompt]:
        chosen = rng.sample(range(len(self.examples)), min(count, len(self.examples)))
        prompts = []
        for idx in chosen:
            example = self.examples[idx]
            prompts.append(Prompt(example.index, self.encode(example), example.gold, example.question))
        return prompts

    def reward(self, completion_ids: Sequence[int], prompt: Prompt) -> float:
        return reward(self.tokenizer.decode(completion_ids), prompt.gold)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.name,
            "split": self.split,
            "num_examples": len(self.examples),
            "reward": "exact match of the last '#### <number>' line (commas/$ stripped)",
            "system_prompt": self.system_prompt,
            "tokenizer": type(self.tokenizer).__name__,
            "eos_token_ids": list(self.tokenizer.eos_token_ids),
        }


class SyntheticArithmeticTask:
    """MOCK-ONLY task: tiny addition prompts with a lenient 'gold digits appear' reward.

    An untrained mock model never emits ``#### <number>``; the lenient reward gives the GRPO
    groups non-degenerate advantages so the trainer step and LoRA sync are exercised.
    """

    name = "synthetic-arith"

    def __init__(self, tokenizer: ChatTokenizer, *, max_operand: int = 20) -> None:
        self.tokenizer = tokenizer
        self.max_operand = max_operand

    def sample(self, rng: random.Random, count: int) -> list[Prompt]:
        prompts = []
        for i in range(count):
            a = rng.randint(0, self.max_operand)
            b = rng.randint(0, self.max_operand)
            question = f"What is {a} + {b}?"
            messages = build_messages(question, system_prompt="Answer with the number only.")
            prompts.append(Prompt(i, self.tokenizer.encode_chat(messages), str(a + b), question))
        return prompts

    def reward(self, completion_ids: Sequence[int], prompt: Prompt) -> float:
        return 1.0 if prompt.gold in self.tokenizer.decode(completion_ids) else 0.0

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.name,
            "reward": "1.0 iff the gold number string appears in the decoded completion (mock only)",
            "max_operand": self.max_operand,
            "tokenizer": type(self.tokenizer).__name__,
        }

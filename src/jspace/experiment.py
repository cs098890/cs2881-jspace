"""Datasets, prompting, answer extraction and scoring for the three difficulty tiers."""

from __future__ import annotations

import re
from typing import Literal

from datasets import load_dataset

Condition = Literal["cot", "direct"]

# Exact dataset sources, stated for reproducibility.
SOURCES = {
    "gsm8k":     ("openai/gsm8k", "main", "test"),
    "math500":   ("HuggingFaceH4/MATH-500", None, "test"),
    # AIME 2025 (AIME I + II, 30 problems). Chosen over AIME 2024 because Qwen3's
    # pretraining cutoff makes 2024 a contamination risk.
    "aime2025":  ("yentinglin/aime_2025", None, "train"),
}

COT_INSTRUCTION = (
    "Solve the problem step by step, showing your reasoning. "
    "Then give the final answer on its own line in the form: The answer is <answer>."
)
DIRECT_INSTRUCTION = (
    "Give only the final answer. Do not show any reasoning or working. "
    "Respond with exactly one line in the form: The answer is <answer>."
)


def load_problems(name: str, n: int, seed: int = 0):
    """Return [{'problem': str, 'answer': str}] for the named dataset."""
    repo, config, split = SOURCES[name]
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out = []
    for row in ds:
        if name == "gsm8k":
            q, a = row["question"], row["answer"].split("####")[-1].strip()
        elif name == "math500":
            q, a = row["problem"], str(row["answer"]).strip()
        else:
            q = row.get("problem") or row.get("question")
            a = str(row.get("answer")).strip()
        out.append({"problem": q, "answer": a})
    return out


def build_prompt(tokenizer, problem: str, condition: Condition) -> str:
    instr = COT_INSTRUCTION if condition == "cot" else DIRECT_INSTRUCTION
    messages = [{"role": "user", "content": f"{problem}\n\n{instr}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,   # thinking traces are 10k+ tokens; out of budget
    )


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm(s: str) -> str:
    s = s.strip().rstrip(".").replace(",", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\d?frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = s.removeprefix("\\").strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()


def extract_answer(text: str) -> str | None:
    """Pull the predicted answer out of a completion."""
    m = list(re.finditer(r"(?:the answer is|answer:)\s*\**\s*([^\n]*)", text, re.I))
    if m:
        cand = m[-1].group(1).strip()
        cand = cand.split("</")[0].strip()
        if cand:
            boxed = re.search(r"\\boxed\{([^}]*)\}", cand)
            return _norm(boxed.group(1) if boxed else cand)
    boxed = list(re.finditer(r"\\boxed\{([^}]*)\}", text))
    if boxed:
        return _norm(boxed[-1].group(1))
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    try:                                   # numeric tolerance
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False

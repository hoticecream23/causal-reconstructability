"""Tasks. A task is prompts plus a scalar per-prompt metric read at the final token.

The metric is always a *signed* logit difference `logit[correct] - logit[incorrect]`, so
higher is always more correct and ablating a causally useful feature drives it down.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .config import Config


@dataclass
class Task:
    prompts: list[str]
    correct_ids: torch.Tensor  # [n]
    incorrect_ids: torch.Tensor  # [n]
    name: str

    def __len__(self) -> int:
        return len(self.prompts)


def metric_from_logits(logits: torch.Tensor, correct: torch.Tensor, incorrect: torch.Tensor):
    """[batch, seq, vocab] -> [batch] signed logit difference at the final position."""
    last = logits[:, -1, :].to(torch.float32)
    return last.gather(1, correct[:, None]).squeeze(1) - last.gather(1, incorrect[:, None]).squeeze(1)


def build(cfg: Config, tokenizer) -> Task:
    if cfg.task == "synthetic":
        return _synthetic(cfg, tokenizer)
    if cfg.task == "bias_in_bios":
        return _bias_in_bios(cfg, tokenizer)
    raise KeyError(f"unknown task {cfg.task!r}")


def _first_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{text!r} tokenized to nothing")
    return ids[0]


# --------------------------------------------------------------------------- synthetic

_CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Greece", "Athens"),
    ("Norway", "Oslo"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Russia", "Moscow"), ("China", "Beijing"), ("Kenya", "Nairobi"), ("Cuba", "Havana"),
    ("Ireland", "Dublin"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
    ("Hungary", "Budapest"), ("Peru", "Lima"), ("Iraq", "Baghdad"), ("Iran", "Tehran"),
    ("Sweden", "Stockholm"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"),
]

_TEMPLATES = [
    "The capital city of {c} is",
    "Q: What is the capital of {c}? A:",
    "When people visit {c}, they usually fly into the capital,",
    "According to the atlas, the capital of {c} is",
    "She had never been to {c} before, but she knew its capital was",
    "The embassy is located in the capital of {c}, namely",
    "Geography quiz, question {n}: name the capital of {c}. The answer is",
    "Travelling through {c} last spring, we spent three days in the capital,",
]

_PREAMBLES = [
    "", "The weather had turned cold. ", "It was a long afternoon. ",
    "Notes from the seminar: ", "He put down his coffee. ",
    "The train was running late again. ", "Chapter four. ",
    "Someone in the back row raised a hand. ", "After a short pause, ",
    "The lecture continued. ",
]


def _synthetic(cfg: Config, tokenizer) -> Task:
    """Offline smoke-test task. No downloads; exists so the pipeline can be exercised.

    Prompt variety matters: with few distinct prompts the train/eval split degenerates and
    every R² pins at 1.0. 26 countries x 8 templates x 10 preambles x quiz number keeps
    duplicates rare at the sample sizes used here.
    """
    rng = random.Random(cfg.seed)
    prompts, correct, incorrect = [], [], []
    for _ in range(cfg.n_prompts):
        country, city = rng.choice(_CAPITALS)
        other = rng.choice([c for _, c in _CAPITALS if c != city])
        body = rng.choice(_TEMPLATES).format(c=country, n=rng.randint(1, 40))
        prompts.append(rng.choice(_PREAMBLES) + body)
        correct.append(_first_token(tokenizer, " " + city))
        incorrect.append(_first_token(tokenizer, " " + other))
    return Task(prompts, torch.tensor(correct), torch.tensor(incorrect), "synthetic")


# ------------------------------------------------------------------------ bias in bios

_PROMPT = "{bio}\n\nThe profession of this person is"
_PAIR = ("professor", "nurse")

# LabHC/bias_in_bios stores `profession` as a bare int64 with no ClassLabel names, so the
# mapping has to come from somewhere. These are the 28 professions in alphabetical order,
# which is the order the ids follow. Spot-checked: id 21 returns academic bios, id 13
# returns nursing bios, and id 21 is the largest class as expected for this dataset.
_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher",
]


def _bias_in_bios(cfg: Config, tokenizer) -> Task:
    from datasets import load_dataset

    ds = load_dataset("LabHC/bias_in_bios", split="train")

    names = getattr(ds.features.get("profession"), "names", None) or _PROFESSIONS
    if len(names) != 28:
        raise RuntimeError(f"expected 28 professions, dataset has {len(names)}")
    label_of = {p: names.index(p) for p in _PAIR}

    keep = set(label_of.values())
    ds = ds.filter(lambda r: r["profession"] in keep).shuffle(seed=cfg.seed)

    # The raw classes are ~86/14 professor:nurse. Left unbalanced, a constant "professor"
    # answer scores 86% and the signed logit difference mostly measures the class prior
    # rather than anything the bio says. Take equal numbers of each.
    per_class = cfg.n_prompts // 2
    chosen = []
    for label in (label_of[_PAIR[0]], label_of[_PAIR[1]]):
        idx = [i for i, p in enumerate(ds["profession"]) if p == label][:per_class]
        if len(idx) < per_class:
            print(f"[data] warning: only {len(idx)} examples for label {label}, "
                  f"wanted {per_class}")
        chosen.extend(idx)
    ds = ds.select(sorted(chosen))

    pos_id = _first_token(tokenizer, " " + _PAIR[0])
    neg_id = _first_token(tokenizer, " " + _PAIR[1])
    if pos_id == neg_id:
        raise ValueError("answer pair shares a first token; pick different professions")

    # Truncate the bio itself so the answer-bearing suffix always survives.
    budget = cfg.max_len - 24
    prompts, correct, incorrect = [], [], []
    for row in ds:
        ids = tokenizer.encode(row["hard_text"], add_special_tokens=False)[:budget]
        prompts.append(_PROMPT.format(bio=tokenizer.decode(ids)))
        is_prof = row["profession"] == label_of[_PAIR[0]]
        correct.append(pos_id if is_prof else neg_id)
        incorrect.append(neg_id if is_prof else pos_id)

    return Task(prompts, torch.tensor(correct), torch.tensor(incorrect), "bias_in_bios")


def split_indices(task: Task, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Disjoint train/eval indices, grouped by prompt text.

    Splitting on row index leaks whenever a prompt repeats: the reconstructor sees the
    exact activation vector it is later scored on and R² pins at 1.0. Grouping by prompt
    guarantees a duplicated prompt lands wholly on one side.
    """
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(task.prompts):
        groups.setdefault(p, []).append(i)

    keys = sorted(groups)
    g = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(keys), generator=g).tolist()

    target = int(len(task) * cfg.train_frac)
    train: list[int] = []
    ev: list[int] = []
    for k in order:
        (train if len(train) < target else ev).extend(groups[keys[k]])

    if not ev:
        raise RuntimeError(
            f"eval split is empty: {len(keys)} distinct prompts for {len(task)} rows. "
            "The task does not have enough variety to hold anything out."
        )
    return torch.tensor(sorted(train)), torch.tensor(sorted(ev))


def distinct_ratio(task: Task) -> float:
    return len(set(task.prompts)) / max(len(task), 1)

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


AUTOINTERP_PROTOCOL = "saebench-autointerp-v0.3-compatible"


@dataclass(frozen=True)
class AutoInterpConfig:
    """Protocol settings used by the SAEBench AutoInterp evaluation."""

    context_size: int = 128
    buffer: int = 10
    no_overlap: bool = True
    activation_threshold_fraction: float = 0.01
    n_top_generation: int = 10
    n_importance_generation: int = 5
    n_top_scoring: int = 2
    n_random_scoring: int = 10
    n_importance_scoring: int = 2
    max_explanation_tokens: int = 30
    use_explanation_demos: bool = True
    seed: int = 42

    @property
    def n_top_total(self) -> int:
        return self.n_top_generation + self.n_top_scoring

    @property
    def n_importance_total(self) -> int:
        return self.n_importance_generation + self.n_importance_scoring

    @property
    def n_scoring(self) -> int:
        return (
            self.n_top_scoring
            + self.n_random_scoring
            + self.n_importance_scoring
        )

    @property
    def n_positive_scoring(self) -> int:
        return self.n_top_scoring + self.n_importance_scoring

    @property
    def scoring_baseline_accuracy(self) -> float:
        return (self.n_scoring - self.n_positive_scoring) / self.n_scoring

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenWindow:
    sequence_index: int
    center_index: int
    token_ids: list[int]
    activations: list[float]
    activation_threshold: float
    kind: str

    @property
    def center_offset(self) -> int:
        return len(self.token_ids) // 2

    @property
    def center_activation(self) -> float:
        if not self.activations:
            return 0.0
        return float(self.activations[self.center_offset])

    @property
    def active_mask(self) -> list[bool]:
        # The reference token AutoInterp labels and highlights the selected
        # center token. Context tokens are evidence for the explanation, not
        # additional activation units.
        result = [False] * len(self.activations)
        if result:
            value = self.center_activation
            result[self.center_offset] = (
                math.isfinite(value) and value > self.activation_threshold
            )
        return result

    @property
    def is_active(self) -> bool:
        return bool(self.active_mask[self.center_offset]) if self.active_mask else False

    @property
    def maximum_activation(self) -> float:
        value = self.center_activation
        return value if math.isfinite(value) else 0.0

    def to_record(self, tokenizer, *, mark_active: bool) -> dict[str, Any]:
        if not mark_active:
            text = tokenizer.decode(
                self.token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            center = self.center_offset
            decode = lambda ids: tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            text = (
                decode(self.token_ids[:center])
                + f"<<{decode(self.token_ids[center : center + 1])}>>"
                + decode(self.token_ids[center + 1 :])
            )
        text = text.replace("�", "").replace("\n", "↵")
        return {
            "kind": self.kind,
            "sequence_index": self.sequence_index,
            "center_index": self.center_index,
            "token_ids": self.token_ids,
            "activations": self.activations,
            "center_activation": self.center_activation,
            "maximum_activation": self.maximum_activation,
            "activation_threshold": self.activation_threshold,
            "is_active": self.is_active,
            "text": text,
        }


@dataclass(frozen=True)
class ChunkExample:
    """One complete variable-length chunk and its scalar SAE activation."""

    chunk_index: int
    token_ids: list[int]
    activation: float
    activation_threshold: float
    kind: str
    length: int

    @property
    def is_active(self) -> bool:
        return (
            math.isfinite(self.activation)
            and self.activation > self.activation_threshold
        )

    @property
    def maximum_activation(self) -> float:
        return self.activation if math.isfinite(self.activation) else 0.0

    def to_record(self, tokenizer, *, mark_active: bool) -> dict[str, Any]:
        text = tokenizer.decode(
            self.token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # The entire chunk is the activation unit. Delimit the whole chunk in
        # explanation prompts rather than inventing a token-level trigger.
        if mark_active and self.is_active:
            text = f"<<CHUNK>>{text}<</CHUNK>>"
        text = text.replace("�", "").replace("\n", "↵")
        return {
            "kind": self.kind,
            "chunk_index": self.chunk_index,
            "length": self.length,
            "token_ids": self.token_ids,
            "activation": self.activation,
            "maximum_activation": self.maximum_activation,
            "activation_threshold": self.activation_threshold,
            "is_active": self.is_active,
            "text": text,
        }


def _eligible_mask(
    activations: torch.Tensor,
    *,
    buffer: int,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError("activations must have shape [sequence, position]")
    if buffer < 0 or activations.shape[1] <= 2 * buffer:
        raise ValueError("buffer leaves no eligible token positions")
    eligible = torch.zeros_like(activations, dtype=torch.bool)
    eligible[:, buffer : activations.shape[1] - buffer] = True
    if valid_mask is not None:
        if valid_mask.shape != activations.shape:
            raise ValueError("valid_mask shape does not match activations")
        eligible &= valid_mask.to(device=activations.device, dtype=torch.bool)
    eligible &= torch.isfinite(activations)
    return eligible


def top_activation_indices(
    activations: torch.Tensor,
    *,
    k: int,
    buffer: int,
    no_overlap: bool,
    minimum_exclusive: float | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return active top centers with SAEBench's no-overlap semantics."""

    if k <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    eligible = _eligible_mask(
        activations,
        buffer=buffer,
        valid_mask=valid_mask,
    )
    if minimum_exclusive is not None:
        eligible &= activations > float(minimum_exclusive)
    candidate_flat = torch.nonzero(
        eligible.flatten(),
        as_tuple=False,
    ).flatten()
    if not len(candidate_flat):
        return torch.empty((0, 2), dtype=torch.long)
    flat_activations = activations.flatten()
    candidate_values = flat_activations.index_select(0, candidate_flat)
    width = activations.shape[1]
    candidate_count = min(
        candidate_values.numel(),
        max(1024, k * max(4, 2 * buffer + 1)),
    )
    selected: list[tuple[int, int]] = []
    while True:
        values, indices = torch.topk(
            candidate_values,
            candidate_count,
            sorted=True,
        )
        selected = []
        blocked: set[tuple[int, int]] = set()
        for value, flat_index in zip(
            values.tolist(), indices.tolist(), strict=True
        ):
            if not math.isfinite(float(value)):
                break
            original_flat = int(candidate_flat[flat_index])
            row, column = divmod(original_flat, width)
            if no_overlap and (row, column) in blocked:
                continue
            selected.append((row, column))
            if no_overlap:
                for offset in range(-buffer, buffer + 1):
                    blocked.add((row, column + offset))
            if len(selected) == k:
                break
        if len(selected) == k or candidate_count == candidate_values.numel():
            break
        candidate_count = min(candidate_values.numel(), candidate_count * 2)
    if not selected:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(selected, dtype=torch.long)

def importance_sample_indices(
    activations: torch.Tensor,
    *,
    k: int,
    buffer: int,
    generator: torch.Generator,
    minimum_exclusive: float = 0.0,
    excluded_indices: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Activation-weighted sample from active centers outside the top set.

    The copied reference protocol zeros the exact top centers and samples the
    remaining active centers in proportion to activation (not activation
    squared), without replacement.
    """

    if k <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    eligible = _eligible_mask(
        activations,
        buffer=buffer,
        valid_mask=valid_mask,
    )
    values = activations.float().clamp_min(0)
    eligible &= values > float(minimum_exclusive)
    if excluded_indices is not None and excluded_indices.numel():
        excluded = excluded_indices.to(
            device=eligible.device,
            dtype=torch.long,
        ).reshape(-1, 2)
        eligible[excluded[:, 0], excluded[:, 1]] = False
    candidate_flat = torch.nonzero(
        eligible.flatten(),
        as_tuple=False,
    ).flatten()
    available = int(len(candidate_flat))
    if available < k:
        raise ValueError(
            f"only {available} active importance-sampling candidates for k={k}"
        )
    weights = values.flatten().index_select(0, candidate_flat).cpu()
    sampled_local = torch.multinomial(
        weights,
        k,
        replacement=False,
        generator=generator,
    )
    sampled = candidate_flat.cpu().index_select(0, sampled_local)
    width = activations.shape[1]
    return torch.stack((sampled // width, sampled % width), dim=1)

def random_indices(
    activations: torch.Tensor,
    *,
    k: int,
    buffer: int,
    generator: torch.Generator,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Uniformly sample eligible token centers, matching SAEBench random negatives."""

    def draw_unique(
        population: int,
        is_valid,
    ) -> torch.Tensor:
        selected: list[int] = []
        seen: set[int] = set()
        maximum_draws = max(10_000, k * 200)
        draws_used = 0
        while len(selected) < k and draws_used < maximum_draws:
            draws = torch.randint(
                population,
                (max((k - len(selected)) * 4, 32),),
                generator=generator,
            ).tolist()
            draws_used += len(draws)
            for value in draws:
                value = int(value)
                if value in seen or not is_valid(value):
                    continue
                seen.add(value)
                selected.append(value)
                if len(selected) == k:
                    break
        return torch.tensor(selected, dtype=torch.long)

    if k <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    eligible = _eligible_mask(
        activations,
        buffer=buffer,
        valid_mask=valid_mask,
    )
    if valid_mask is None:
        rows, width = activations.shape
        eligible_width = width - 2 * buffer
        population = rows * eligible_width
        if population < k:
            raise ValueError(f"only {population} random candidates for k={k}")
        sampled = draw_unique(population, lambda _value: True)
        return torch.stack(
            (
                sampled // eligible_width,
                sampled % eligible_width + buffer,
            ),
            dim=1,
        )
    rows, width = activations.shape
    eligible_width = width - 2 * buffer
    population = rows * eligible_width
    sampled = draw_unique(
        population,
        lambda value: bool(
            eligible[
                value // eligible_width,
                value % eligible_width + buffer,
            ].item()
        ),
    )
    if len(sampled) < k:
        # This fallback is only reached for unusually sparse masks.
        candidates = torch.nonzero(
            eligible[:, buffer : width - buffer].flatten(),
            as_tuple=False,
        ).flatten().cpu()
        if len(candidates) < k:
            raise ValueError(f"only {len(candidates)} random candidates for k={k}")
        order = torch.randperm(len(candidates), generator=generator)[:k]
        sampled = candidates.index_select(0, order)
    return torch.stack(
        (
            sampled // eligible_width,
            sampled % eligible_width + buffer,
        ),
        dim=1,
    )


def windows_from_indices(
    token_ids: torch.Tensor,
    activations: torch.Tensor,
    indices: torch.Tensor,
    *,
    buffer: int,
    activation_threshold: float,
    kind: str,
) -> list[TokenWindow]:
    if token_ids.shape != activations.shape:
        raise ValueError("token_ids and activations must have the same shape")
    windows = []
    for row, column in indices.tolist():
        left, right = column - buffer, column + buffer + 1
        windows.append(
            TokenWindow(
                sequence_index=int(row),
                center_index=int(column),
                token_ids=token_ids[row, left:right].tolist(),
                activations=activations[row, left:right].float().tolist(),
                activation_threshold=float(activation_threshold),
                kind=kind,
            )
        )
    return windows


def split_examples(
    *,
    token_ids: torch.Tensor,
    activations: torch.Tensor,
    config: AutoInterpConfig,
    feature_seed: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[list[TokenWindow], list[TokenWindow]]:
    """Construct SAEBench token-centered generation/scoring examples."""

    if token_ids.shape != activations.shape:
        raise ValueError("token_ids and activations must have the same shape")
    generator = torch.Generator().manual_seed(int(feature_seed))
    eligible = _eligible_mask(
        activations,
        buffer=config.buffer,
        valid_mask=valid_mask,
    )
    maximum = float(
        activations.float().masked_fill(~eligible, float("-inf")).max().item()
    )
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("feature has no positive activation in held-out data")
    activation_threshold = config.activation_threshold_fraction * maximum
    top_indices = top_activation_indices(
        activations,
        k=config.n_top_total,
        buffer=config.buffer,
        no_overlap=config.no_overlap,
        minimum_exclusive=activation_threshold,
        valid_mask=valid_mask,
    )
    if len(top_indices) < config.n_top_total:
        raise ValueError(
            f"found only {len(top_indices)}/{config.n_top_total} active top examples"
        )
    importance_indices = importance_sample_indices(
        activations,
        k=config.n_importance_total,
        buffer=config.buffer,
        generator=generator,
        minimum_exclusive=activation_threshold,
        excluded_indices=top_indices,
        valid_mask=valid_mask,
    )
    random_centers = random_indices(
        activations,
        k=config.n_random_scoring,
        buffer=config.buffer,
        generator=generator,
        valid_mask=valid_mask,
    )

    top_generation = top_indices[: config.n_top_generation]
    top_scoring = top_indices[
        config.n_top_generation : config.n_top_total
    ]
    importance_generation = importance_indices[
        : config.n_importance_generation
    ]
    importance_scoring = importance_indices[
        config.n_importance_generation : config.n_importance_total
    ]

    generation = windows_from_indices(
        token_ids,
        activations,
        top_generation,
        buffer=config.buffer,
        activation_threshold=activation_threshold,
        kind="top",
    ) + windows_from_indices(
        token_ids,
        activations,
        importance_generation,
        buffer=config.buffer,
        activation_threshold=activation_threshold,
        kind="importance",
    )
    generation.sort(key=lambda window: window.maximum_activation, reverse=True)

    scoring = windows_from_indices(
        token_ids,
        activations,
        top_scoring,
        buffer=config.buffer,
        activation_threshold=activation_threshold,
        kind="top",
    ) + windows_from_indices(
        token_ids,
        activations,
        importance_scoring,
        buffer=config.buffer,
        activation_threshold=activation_threshold,
        kind="importance",
    ) + windows_from_indices(
        token_ids,
        activations,
        random_centers,
        buffer=config.buffer,
        activation_threshold=activation_threshold,
        kind="random",
    )
    random.Random(int(feature_seed) + 1_000_003).shuffle(scoring)
    return generation, scoring

def split_chunk_examples(
    *,
    token_ids: torch.Tensor,
    offsets: torch.Tensor,
    activations: torch.Tensor,
    config: AutoInterpConfig,
    feature_seed: int,
    eligible_mask: torch.Tensor | None = None,
) -> tuple[list[ChunkExample], list[ChunkExample]]:
    """Apply the same AutoInterp pipeline to complete chunk activation units."""

    if token_ids.ndim != 1 or offsets.ndim != 1 or activations.ndim != 1:
        raise ValueError("chunk inputs must be flat token IDs, offsets, and scores")
    if len(offsets) != len(activations) + 1:
        raise ValueError("chunk offsets do not match activation rows")
    scores = activations.float()
    finite = torch.isfinite(scores)
    if eligible_mask is not None:
        if eligible_mask.shape != scores.shape:
            raise ValueError("eligible_mask does not match chunk activations")
        finite &= eligible_mask.to(dtype=torch.bool, device=scores.device)
    maximum = float(scores.masked_fill(~finite, float("-inf")).max().item())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("feature has no positive chunk activation")
    activation_threshold = config.activation_threshold_fraction * maximum
    active = finite & (scores > activation_threshold)
    active_count = int(active.sum().item())
    if active_count < config.n_top_total + config.n_importance_total:
        raise ValueError(
            f"only {active_count} active chunks for "
            f"{config.n_top_total + config.n_importance_total} positive examples"
        )
    top_scores = scores.masked_fill(~active, float("-inf"))
    _values, top_indices = torch.topk(
        top_scores,
        config.n_top_total,
        sorted=True,
    )
    top_indices = top_indices.cpu()

    generator = torch.Generator().manual_seed(int(feature_seed))
    importance_weights = scores.clamp_min(0).masked_fill(~active, 0)
    importance_weights[top_indices.to(importance_weights.device)] = 0
    importance_weights = importance_weights.cpu()
    available = int((importance_weights > 0).sum().item())
    if available < config.n_importance_total:
        raise ValueError(
            f"only {available} active chunk importance candidates for "
            f"k={config.n_importance_total}"
        )
    importance_indices = torch.multinomial(
        importance_weights,
        config.n_importance_total,
        replacement=False,
        generator=generator,
    )
    random_candidates = torch.nonzero(finite, as_tuple=False).flatten().cpu()
    if len(random_candidates) < config.n_random_scoring:
        raise ValueError("not enough chunks for random scoring examples")
    random_chunk_indices = random_candidates.index_select(
        0,
        torch.randperm(len(random_candidates), generator=generator)[
            : config.n_random_scoring
        ],
    )

    top_generation = top_indices[: config.n_top_generation]
    top_scoring = top_indices[
        config.n_top_generation : config.n_top_total
    ]
    importance_generation = importance_indices[
        : config.n_importance_generation
    ]
    importance_scoring = importance_indices[
        config.n_importance_generation : config.n_importance_total
    ]

    def make(indices: torch.Tensor, kind: str) -> list[ChunkExample]:
        result = []
        for index in indices.tolist():
            start, stop = int(offsets[index]), int(offsets[index + 1])
            result.append(
                ChunkExample(
                    chunk_index=int(index),
                    token_ids=token_ids[start:stop].tolist(),
                    activation=float(scores[index].item()),
                    activation_threshold=activation_threshold,
                    kind=kind,
                    length=stop - start,
                )
            )
        return result

    generation = (
        make(top_generation, "top")
        + make(importance_generation, "importance")
    )
    generation.sort(
        key=lambda example: example.maximum_activation,
        reverse=True,
    )
    scoring = (
        make(top_scoring, "top")
        + make(importance_scoring, "importance")
        + make(random_chunk_indices, "random")
    )
    random.Random(int(feature_seed) + 1_000_003).shuffle(scoring)
    return generation, scoring

def generation_prompt(
    example_records: Sequence[dict[str, Any]],
    *,
    use_demos: bool,
) -> str:
    examples = "\n".join(
        f"{index}. {record['text']}"
        for index, record in enumerate(example_records, start=1)
    )
    system = (
        "We're studying neurons in a neural network. Each neuron activates on "
        "some particular word/words/substring/concept in a short document. The "
        "activating words in each document are indicated with << ... >>. We will "
        "give you a list of documents on which the neuron activates, in order "
        "from most strongly activating to least strongly activating. Look at the "
        "parts of the document the neuron activates for and summarize in a "
        "single sentence what the neuron is activating on. Try not to be overly "
        "specific in your explanation. Note that some neurons will activate only "
        "on specific words or substrings, but others will activate on most/all "
        "words in a sentence provided that sentence contains some particular "
        "concept. Your explanation should cover most or all activating words. "
        "Pay attention to capitalization and punctuation when relevant. Keep the "
        "explanation as short and simple as possible, limited to 20 words or "
        "less. Omit punctuation and formatting. Avoid long lists of words."
    )
    if use_demos:
        system += (
            ' Examples: "This neuron activates on the word knows in rhetorical '
            'questions", "This neuron activates on verbs related to '
            'decision-making and preferences", "This neuron activates on the '
            'substring Ent at the start of words", and "This neuron activates '
            'on text about government economic policy".'
        )
    return (
        f"SYSTEM:\n{system}\n\n"
        f"USER:\nThe activating documents are given below:\n\n{examples}"
    )


def chunk_generation_prompt(
    example_records: Sequence[dict[str, Any]],
    *,
    use_demos: bool,
) -> str:
    examples = "\n".join(
        f"{index}. {record['text']}"
        for index, record in enumerate(example_records, start=1)
    )
    system = (
        "We're studying sparse features defined over complete text chunks. "
        "Each example between <<CHUNK>> and <</CHUNK>> is one indivisible "
        "activation unit: every token in the chunk jointly determines the "
        "feature activation, and there is no privileged trigger token. The "
        "examples are ordered from strongest to weakest activation. Summarize "
        "in one concise sentence the common whole-chunk topic, domain, style, "
        "intent, discourse role, or document function that best explains why "
        "the feature activates. Use only a pattern shared across most examples; "
        "do not describe incidental words or merely say that the chunks are "
        "coherent text. If the examples are heterogeneous, say Insufficient "
        "evidence. Keep the explanation to 20 words or fewer and omit extra "
        "commentary."
    )
    if use_demos:
        system += (
            ' Examples: "appellate court opinions describing procedural '
            'history", "technical troubleshooting questions seeking code '
            'solutions", and "product pages advertising consumer electronics".'
        )
    return (
        f"SYSTEM:\n{system}\n\n"
        f"USER:\nThe activating chunks are given below:\n\n{examples}"
    )


def scoring_prompt(
    explanation: str,
    example_records: Sequence[dict[str, Any]],
    *,
    positive_count: int,
    feature_seed: int,
) -> str:
    examples = "\n".join(
        f"{index}. {record['text']}"
        for index, record in enumerate(example_records, start=1)
    )
    demo = sorted(
        random.Random(int(feature_seed) + 2_000_003).sample(
            range(1, len(example_records) + 1),
            k=positive_count,
        )
    )
    demo_text = ", ".join(map(str, demo))
    system = (
        "We're studying neurons in a neural network. Each neuron activates on "
        "some particular word/words/substring/concept in a short document. You "
        f"will be given a short explanation and then {len(example_records)} "
        "example sequences in random order. Return a comma-separated list of "
        "the examples where the neuron should activate at least once, on ANY "
        "word or substring in the document. For example, your response might "
        f'look like "{demo_text}". Try not to be overly specific. If none '
        'should activate, respond with "None". Include nothing except '
        "comma-separated numbers or None."
    )
    return (
        f"SYSTEM:\n{system}\n\n"
        f"USER:\nHere is the explanation: this neuron fires on {explanation}."
        f"\n\nHere are the examples:\n\n{examples}"
    )


def chunk_scoring_prompt(
    explanation: str,
    example_records: Sequence[dict[str, Any]],
    *,
    positive_count: int,
    feature_seed: int,
) -> str:
    examples = "\n".join(
        f"{index}. {record['text']}"
        for index, record in enumerate(example_records, start=1)
    )
    demo = sorted(
        random.Random(int(feature_seed) + 2_000_003).sample(
            range(1, len(example_records) + 1),
            k=positive_count,
        )
    )
    system = (
        "We're studying sparse features defined over complete text chunks. "
        f"You will receive one whole-chunk explanation and "
        f"{len(example_records)} complete chunks in random order. Decide which "
        "chunks should activate the feature as a whole. Do not search for a "
        "single trigger token. Return only a comma-separated list of 1-based "
        f"indices, for example {', '.join(map(str, demo))}, or None if no chunk "
        "matches. Include no other text."
    )
    return (
        f"SYSTEM:\n{system}\n\n"
        f"USER:\nHere is the explanation: this feature activates on "
        f"{explanation}.\n\nHere are the chunks:\n\n{examples}"
    )


def parse_explanation(response: str) -> str:
    text = response.strip()
    text = re.split(r"(?i)activates on", text)[-1].strip()
    return text.rstrip(".").strip()


def parse_predictions(response: str, n_examples: int) -> list[int] | None:
    text = response.strip().rstrip(".")
    if not text or text.lower() == "none":
        return []
    # Accept a JSON wrapper from models which insist on structured output.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("indices", parsed.get("predictions"))
        if isinstance(parsed, list):
            values = [int(value) for value in parsed]
            return (
                sorted(set(values))
                if all(1 <= value <= n_examples for value in values)
                else None
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    normalized = re.sub(r"(?i)\band\b", ",", text)
    values = [part.strip() for part in normalized.split(",") if part.strip()]
    if not values:
        return []
    if not all(value.isdigit() for value in values):
        return None
    predictions = sorted(set(map(int, values)))
    if not all(1 <= value <= n_examples for value in predictions):
        return None
    return predictions


def score_predictions(
    predictions: Sequence[int],
    scoring_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    predicted = set(map(int, predictions))
    gold = {
        index
        for index, record in enumerate(scoring_records, start=1)
        if bool(record["is_active"])
    }
    labels = []
    for index in range(1, len(scoring_records) + 1):
        labels.append(
            {
                "index": index,
                "gold": index in gold,
                "predicted": index in predicted,
                "correct": (index in gold) == (index in predicted),
            }
        )
    return {
        "n": len(scoring_records),
        "predicted_indices": sorted(predicted),
        "gold_indices": sorted(gold),
        "correct": sum(row["correct"] for row in labels),
        "score": sum(row["correct"] for row in labels)
        / max(1, len(scoring_records)),
        "labels": labels,
    }

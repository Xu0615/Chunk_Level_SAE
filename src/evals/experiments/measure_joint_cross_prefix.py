"""Measure the empirical dictionary support of the Cross task.

The Joint Chunk SAE prefix variant reconstructs the owning chunk from the full
dictionary and predicts the partner chunk from the leading ``h`` columns only.
``h`` is therefore not a free hyperparameter: it must be at least the number of
dictionary directions the Cross task actually uses, or the prefix becomes a
bottleneck that starves partner prediction.

This script reads the shared invariant audit artifact (full-dictionary Lorenz
curves over a fixed validation sample) and reports, per SAE, how much activation
mass the top-``f`` fraction of coordinates carries.  It prints a recommended
``h`` and the matching expected ``k_prefix`` so the training run can be checked
for prefix collapse from its first logged step.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT = (
    "qwen/qwen35-9b-base/pile_proportional_exact_occurrence/ablation/"
    "invariant_evaluations.json"
)
DICT_WIDTH = 65536
ACTIVATION_K = 128
# Candidate prefix sizes: powers of two so BatchTopK histogram bucketing stays aligned.
CANDIDATE_PREFIXES = (4096, 8192, 16384, 32768)


def lorenz_mass_of_top(
    population: list[float],
    activity: list[float],
    top_fraction: float,
) -> float:
    """Return the activation mass held by the top ``top_fraction`` of features.

    The stored Lorenz curve is an ascending cumulative distribution: the least
    active coordinates come first.  The mass held by the most active fraction
    ``f`` is therefore ``1 - L(1 - f)`` with ``L`` linearly interpolated.
    """

    target = 1.0 - top_fraction
    index = min(max(bisect.bisect_left(population, target), 1), len(population) - 1)
    x0, x1 = population[index - 1], population[index]
    y0, y1 = activity[index - 1], activity[index]
    span = x1 - x0
    interpolated = y0 + (y1 - y0) * ((target - x0) / span if span > 0 else 0.0)
    return 1.0 - interpolated


def smallest_prefix_covering(
    population: list[float],
    activity: list[float],
    coverage: float,
) -> tuple[int, int]:
    """Bracket the smallest dictionary prefix reaching ``coverage``.

    The stored curve has 256 buckets over the whole dictionary, so its knots sit
    ``DICT_WIDTH / 256`` coordinates apart.  Reporting a single integer would be
    interpolation noise below that grid, so return the enclosing bucket bracket
    instead.
    """

    bucket = DICT_WIDTH // (len(population) - 1)
    for width in range(bucket, DICT_WIDTH + 1, bucket):
        if lorenz_mass_of_top(population, activity, width / DICT_WIDTH) >= coverage:
            return width - bucket, width
    return DICT_WIDTH, DICT_WIDTH


def load_utilization(artifact: Path) -> dict[str, dict[str, Any]]:
    with artifact.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError(f"artifact lacks evaluations: {artifact}")
    selected: dict[str, dict[str, Any]] = {}
    for record in evaluations:
        if record.get("stage") != "full":
            continue
        label = f"{record['experiment']}_seed{record['seed']}"
        adjacent = record.get("adjacent") or {}
        utilization = adjacent.get("dictionary_utilization")
        if isinstance(utilization, dict) and utilization.get("lorenz_activity"):
            selected[label] = utilization
    if not selected:
        raise ValueError("no full-stage dictionary_utilization records found")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    utilization = load_utilization(Path(args.artifact).expanduser())

    print("Cross-task dictionary support from the shared invariant audit")
    print(f"artifact: {args.artifact}")
    print()

    header = (
        f"{'model':<12}{'sampled':>9}{'active':>8}{'gini':>7}"
        f"{'eff.feat':>10}{'top1%':>8}{'top25%':>8}"
    )
    print(header)
    print("-" * len(header))
    rows: dict[str, Any] = {}
    for label in sorted(utilization):
        u = utilization[label]
        population = u["lorenz_population"]
        activity = u["lorenz_activity"]
        mass_1pct = lorenz_mass_of_top(population, activity, 0.01)
        mass_25pct = lorenz_mass_of_top(population, activity, 0.25)
        print(
            f"{label:<12}{u['sampled_features']:>9}{u['active_features']:>8}"
            f"{u['gini']:>7.3f}{u['effective_features']:>10.1f}"
            f"{mass_1pct:>8.3f}{mass_25pct:>8.3f}"
        )
        rows[label] = {
            "sampled_features": u["sampled_features"],
            "active_features": u["active_features"],
            "gini": u["gini"],
            "effective_features": u["effective_features"],
            "mass_top_1pct": mass_1pct,
            "mass_top_25pct": mass_25pct,
        }

    print()
    print(f"Activation mass covered by each candidate prefix (coverage target {args.coverage:.0%})")
    print(
        "The curve stores 256 buckets over the dictionary, so every candidate below"
        " lands on an exact knot; h* is reported as the enclosing bucket bracket."
    )
    header2 = f"{'model':<12}" + "".join(f"{h:>10}" for h in CANDIDATE_PREFIXES) + f"{'h* bracket':>18}"
    print(header2)
    print("-" * len(header2))
    for label in sorted(utilization):
        u = utilization[label]
        population = u["lorenz_population"]
        activity = u["lorenz_activity"]
        cells = "".join(
            f"{lorenz_mass_of_top(population, activity, h / DICT_WIDTH):>10.3f}"
            for h in CANDIDATE_PREFIXES
        )
        low, high = smallest_prefix_covering(population, activity, args.coverage)
        print(f"{label:<12}{cells}{f'({low}, {high}]':>18}")
        rows[label]["prefix_mass"] = {
            str(h): lorenz_mass_of_top(population, activity, h / DICT_WIDTH)
            for h in CANDIDATE_PREFIXES
        }
        rows[label]["required_prefix_bracket"] = [low, high]

    print()
    print("Expected k_prefix if the code were spread uniformly over the dictionary")
    print("(collapse guard: k_prefix -> 0 starves Cross; k_prefix -> k degenerates to")
    print(" one output fitting two targets)")
    for h in CANDIDATE_PREFIXES:
        print(f"  h={h:>6}  uniform k_prefix ~ {ACTIVATION_K * h / DICT_WIDTH:>6.1f} of {ACTIVATION_K}")

    print()
    print("Caveats that bound how far these numbers can be pushed:")
    print(
        "  - The curve is built from each pair's top-"
        f"{next(iter(utilization.values())).get('common_activity_budget', '?')}"
        " codes, not the full top-128, so mass is biased high and the h needed to"
        " cover a true k=128 code is larger than shown."
    )
    print(
        "  - Mean-target rows (E0/E1/E2) sample 8192 of 65536 alive coordinates, so"
        " scaling by h/65536 is valid.  Sequence rows (E3/E4) census an already"
        " support-filtered pool and are NOT comparable on this axis."
    )
    print(
        "  - Same-config seed spread reaches ~0.04 in mass, wider than the 256-"
        "coordinate quantization error; treat differences below that as noise."
    )

    if args.json_out:
        out = Path(args.json_out).expanduser()
        payload = {
            "format": "chunk-saes-joint-prefix-support-v1",
            "artifact": str(args.artifact),
            "coverage_target": args.coverage,
            "dict_width": DICT_WIDTH,
            "activation_k": ACTIVATION_K,
            "candidate_prefixes": list(CANDIDATE_PREFIXES),
            "models": rows,
        }
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

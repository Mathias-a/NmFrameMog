from __future__ import annotations

from random import Random

from .baseline import combine_distributions, floor_and_normalize
from .contract import CLASS_COUNT, TerrainClass
from .models import SeedState


def aggregate_rollouts(
    state: SeedState,
    *,
    rollout_count: int,
    random_seed: int,
) -> list[list[list[float]]]:
    height = len(state.initial_grid)
    width = len(state.initial_grid[0])
    counts: list[list[list[int]]] = [
        [[0 for _ in range(CLASS_COUNT)] for _ in range(width)] for _ in range(height)
    ]
    rng = Random(random_seed)

    for _ in range(rollout_count):
        sampled_classes = _sample_classes(state.current_tensor, rng=rng)
        smoothed_distribution = _distribution_from_sampled_classes(sampled_classes)
        for y in range(height):
            for x in range(width):
                sampled_class = _sample_from_distribution(
                    smoothed_distribution[y][x], rng=rng
                )
                counts[y][x][sampled_class] += 1

    aggregated: list[list[list[float]]] = []
    for y in range(height):
        row = []
        for x in range(width):
            probabilities = [count / rollout_count for count in counts[y][x]]
            row.append(floor_and_normalize(probabilities))
        aggregated.append(row)
    return aggregated


def _sample_classes(tensor: list[list[list[float]]], *, rng: Random) -> list[list[int]]:
    return [
        [_sample_from_distribution(cell, rng=rng) for cell in row] for row in tensor
    ]


def _distribution_from_sampled_classes(
    sampled_classes: list[list[int]],
) -> list[list[list[float]]]:
    height = len(sampled_classes)
    width = len(sampled_classes[0])
    distributions: list[list[list[float]]] = []
    for y in range(height):
        row = []
        for x in range(width):
            base_distribution = [0.0] * CLASS_COUNT
            base_distribution[sampled_classes[y][x]] = 1.0
            neighbor_classes = _neighbor_classes(sampled_classes, x=x, y=y)
            if neighbor_classes:
                support_distribution = [0.0] * CLASS_COUNT
                for terrain_class in neighbor_classes:
                    support_distribution[terrain_class] += 1.0 / len(neighbor_classes)
                row.append(
                    combine_distributions(
                        floor_and_normalize(base_distribution),
                        floor_and_normalize(support_distribution),
                        prior_weight=0.7,
                        evidence_weight=0.3,
                    )
                )
            else:
                row.append(floor_and_normalize(base_distribution))
        distributions.append(row)
    return distributions


def _neighbor_classes(sampled_classes: list[list[int]], *, x: int, y: int) -> list[int]:
    height = len(sampled_classes)
    width = len(sampled_classes[0])
    classes: list[int] = []
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            if delta_x == 0 and delta_y == 0:
                continue
            next_x = x + delta_x
            next_y = y + delta_y
            if 0 <= next_x < width and 0 <= next_y < height:
                classes.append(sampled_classes[next_y][next_x])
    if (
        int(TerrainClass.PORT) in classes
        and int(TerrainClass.SETTLEMENT) not in classes
    ):
        classes.append(int(TerrainClass.SETTLEMENT))
    return classes


def _sample_from_distribution(distribution: list[float], *, rng: Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(distribution):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(distribution) - 1

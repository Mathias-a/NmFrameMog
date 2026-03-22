"""Main entry point for supervised learning prediction pipeline.

Usage:
  uv run python -m astar_island.solver train           # Train on all fixtures, save model
  uv run python -m astar_island.solver predict <grid>   # Predict from saved model (saves to disk)
  uv run python -m astar_island.solver submit           # Fetch active round, predict all seeds, submit
  uv run python -m astar_island.solver submit --dry-run # Same but save locally without submitting
  uv run python -m astar_island.solver observe          # Query + blend + re-submit (uses budget)
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from astar_island.calibration import calibrate_prediction
from astar_island.features import (
    extract_cell_features,
    extract_cell_targets,
    raw_grid_to_class_grid,
)
from astar_island.fixtures import Fixture, load_all_fixtures
from astar_island.models import GBTSoftClassifier
from astar_island.prediction import apply_probability_floor, validate_prediction
from astar_island.terrain import TERRAIN_CODE_TO_CLASS, NUM_PREDICTION_CLASSES

MODEL_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent.parent / "models"
DEFAULT_ARTIFACT: Path = MODEL_DIR / "gbt_model.joblib"
DEFAULT_TEMPERATURE: float = 1.12

_cached_artifact: dict[str, Any] | None = None
_cached_artifact_path: str | None = None
_artifact_lock = threading.Lock()


def _load_artifact(artifact_path: Path) -> dict[str, Any]:
    global _cached_artifact, _cached_artifact_path
    with _artifact_lock:
        resolved = str(artifact_path.resolve())
        if _cached_artifact is not None and _cached_artifact_path == resolved:
            return _cached_artifact

        import joblib

        artifact = joblib.load(artifact_path)
        _cached_artifact = artifact
        _cached_artifact_path = resolved
        return artifact


def train(
    fixtures: list[Fixture] | None = None, artifact_path: Path = DEFAULT_ARTIFACT
) -> None:
    import joblib

    all_fixtures = fixtures or load_all_fixtures()
    if not all_fixtures:
        msg = "No fixtures available for training."
        raise ValueError(msg)
    print(f"Training on {len(all_fixtures)} fixtures...")

    all_x = []
    all_y = []
    for f in all_fixtures:
        class_grid = raw_grid_to_class_grid(f.initial_grid)
        all_x.append(extract_cell_features(class_grid, f.initial_grid))
        all_y.append(extract_cell_targets(f.ground_truth))

    x_train = np.vstack(all_x)
    y_train = np.vstack(all_y)
    print(f"Feature matrix: {x_train.shape}, Target matrix: {y_train.shape}")

    model = GBTSoftClassifier()
    model.fit(x_train, y_train)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "temperature": DEFAULT_TEMPERATURE,
        "n_fixtures": len(all_fixtures),
        "n_cells": len(x_train),
    }
    joblib.dump(artifact, artifact_path)
    print(f"Model saved to {artifact_path}")


def predict_grid(
    raw_grid: list[list[int]],
    artifact_path: Path = DEFAULT_ARTIFACT,
) -> np.ndarray:
    artifact = _load_artifact(artifact_path)
    model: GBTSoftClassifier = artifact["model"]
    temperature: float = artifact["temperature"]

    raw_probs = model.predict_grid(raw_grid)
    return calibrate_prediction(raw_probs, raw_grid, temperature)


def evaluate() -> None:
    from astar_island.evaluation import run_lorocv

    fixtures = load_all_fixtures()
    print(f"Running LOROCV on {len(fixtures)} fixtures...\n")
    run_lorocv(fixtures, verbose=True)


def submit(*, dry_run: bool = False) -> None:
    """Fetch the active round, predict all seeds, and submit (or save locally).

    Steps for each seed:
      1. Extract initial_grid from round detail.
      2. Run predict_grid() to get calibrated probabilities.
      3. Apply probability floor and validate the tensor.
      4. Submit to API (or save to disk if dry_run=True).

    Args:
        dry_run: If True, save predictions to disk instead of submitting.
    """
    from astar_island.api import (
        get_active_round,
        get_round_detail,
        make_client,
        submit_prediction,
    )

    with make_client() as client:
        active = get_active_round(client)
        if active is None:
            print("No active round found.")
            return

        round_id = str(active["id"])
        round_number = active.get("round_number", "?")
        print(f"Active round: #{round_number} ({round_id})")

        detail = get_round_detail(client, round_id)
        seeds_count = int(str(detail["seeds_count"]))
        initial_states: object = detail.get("initial_states")
        if not isinstance(initial_states, list):
            print("ERROR: No initial_states in round detail.")
            return

        width = int(str(detail.get("map_width", 40)))
        height = int(str(detail.get("map_height", 40)))
        print(f"Map: {width}x{height}, seeds: {seeds_count}")

        results: list[tuple[int, str]] = []
        for seed_idx in range(seeds_count):
            if seed_idx >= len(initial_states):
                print(f"  Seed {seed_idx}: no initial_state available, skipping")
                continue

            state = initial_states[seed_idx]
            if not isinstance(state, dict):
                print(f"  Seed {seed_idx}: invalid initial_state format, skipping")
                continue

            raw_grid: object = state.get("grid")
            if not isinstance(raw_grid, list):
                print(f"  Seed {seed_idx}: no grid in initial_state, skipping")
                continue

            print(f"  Seed {seed_idx}: predicting...", end=" ", flush=True)

            # Predict and post-process
            raw_pred = predict_grid(raw_grid)
            safe_pred = apply_probability_floor(raw_pred, raw_grid)

            # Validate before submission
            errors = validate_prediction(safe_pred, width, height)
            if errors:
                print("VALIDATION FAILED:")
                for err in errors:
                    print(f"    - {err}")
                continue

            pred_list: list[list[list[float]]] = safe_pred.tolist()

            if dry_run:
                # Save to disk instead of submitting
                output_dir = MODEL_DIR.parent / "predictions"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{round_id}_seed{seed_idx}.json"
                output_path.write_text(json.dumps(pred_list))
                print(f"saved to {output_path}")
                results.append((seed_idx, "saved"))
            else:
                # Submit to API
                try:
                    resp = submit_prediction(client, round_id, seed_idx, pred_list)
                    status = resp.get("status", "unknown")
                    print(f"submitted — {status}")
                    results.append((seed_idx, str(status)))
                except (httpx.HTTPStatusError, RuntimeError, TypeError) as exc:
                    print(f"FAILED: {exc}")
                    results.append((seed_idx, f"error: {exc}"))

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Round #{round_number} — {'DRY RUN' if dry_run else 'SUBMITTED'}")
        for seed_idx, status in results:
            print(f"  Seed {seed_idx}: {status}")


# ---------------------------------------------------------------------------
# Viewport query planning
# ---------------------------------------------------------------------------

VIEWPORT_SIZE: int = 15


def _plan_viewports(width: int, height: int, n_queries: int) -> list[tuple[int, int]]:
    """Plan viewport positions with concentrated repeats for reliable blending.

    Instead of 9 unique viewports with 1 observation each (unreliable),
    uses fewer unique positions with multiple repeats so each observed cell
    gets 3+ observations.  This makes the empirical class distribution
    at each cell much more reliable for Dirichlet blending.

    Strategy:
      - With 10 queries: 3 unique positions × 3 repeats + 1 extra = 10
      - The 3 positions are chosen to cover the grid well:
        top-left, top-right, bottom-center (covers ~84% of 40×40)
      - With fewer queries: proportionally fewer unique positions

    Args:
        width: Grid width.
        height: Grid height.
        n_queries: Number of queries available for this seed.

    Returns:
        List of (vx, vy) top-left anchor positions (with repeats).
    """
    max_x = max(0, width - VIEWPORT_SIZE)
    max_y = max(0, height - VIEWPORT_SIZE)

    # Candidate positions for good coverage of 40×40 with 15×15 viewports:
    # These 4 positions cover the full grid with overlaps at the center.
    candidate_positions: list[tuple[int, int]] = [
        (0, 0),  # top-left
        (max_x, 0),  # top-right
        (0, max_y),  # bottom-left
        (max_x, max_y),  # bottom-right
    ]

    if n_queries <= 0:
        return []

    # Determine how many unique positions and repeats per position.
    # Target: at least 2 repeats per position for reliable observations.
    n_unique = max(1, min(len(candidate_positions), n_queries // 2))
    positions_to_use = candidate_positions[:n_unique]

    # Distribute queries evenly across unique positions
    result: list[tuple[int, int]] = []
    for i, pos in enumerate(positions_to_use):
        # Base repeats + distribute remainder
        base_repeats = n_queries // n_unique
        extra = 1 if i < (n_queries % n_unique) else 0
        for _ in range(base_repeats + extra):
            result.append(pos)

    return result[:n_queries]


# ---------------------------------------------------------------------------
# Observation accumulation + Dirichlet blending
# ---------------------------------------------------------------------------

# Lookup table: terrain code → prediction class index
_MAX_TERRAIN_CODE: int = max(TERRAIN_CODE_TO_CLASS) + 1
_CODE_LUT: np.ndarray = np.zeros(_MAX_TERRAIN_CODE, dtype=np.int32)
for _code, _cls in TERRAIN_CODE_TO_CLASS.items():
    _CODE_LUT[_code] = _cls


def _accumulate_observation(
    counts: np.ndarray,
    grid_rows: list[list[int]],
    vx: int,
    vy: int,
) -> None:
    """Accumulate one-hot counts from a viewport observation into the counts tensor.

    Args:
        counts: Shape (H, W, K) int32 array — modified in place.
        grid_rows: Viewport grid as list of rows (vy-indexed rows, vx-indexed cols).
        vx: Viewport top-left x (column offset).
        vy: Viewport top-left y (row offset).
    """
    obs = np.array(grid_rows, dtype=np.int32)
    if obs.ndim != 2:
        msg = f"Expected 2D viewport grid, got shape {obs.shape}"
        raise ValueError(msg)
    vh, vw = obs.shape
    ch, cw = counts.shape[0], counts.shape[1]
    if vy + vh > ch or vx + vw > cw:
        msg = f"Viewport ({vx},{vy}) size ({vw},{vh}) exceeds counts grid ({cw},{ch})"
        raise ValueError(msg)
    if np.any((obs < 0) | (obs >= _MAX_TERRAIN_CODE)):
        invalid = obs[(obs < 0) | (obs >= _MAX_TERRAIN_CODE)]
        msg = f"Unknown terrain codes in observation: {np.unique(invalid).tolist()}"
        raise ValueError(msg)
    classes = _CODE_LUT[obs]

    one_hot = np.zeros((vh, vw, NUM_PREDICTION_CLASSES), dtype=np.int32)
    rows_idx = np.arange(vh)[:, np.newaxis]
    cols_idx = np.arange(vw)[np.newaxis, :]
    one_hot[rows_idx, cols_idx, classes] = 1

    counts[vy : vy + vh, vx : vx + vw, :] += one_hot


def _dirichlet_blend(
    ml_pred: np.ndarray,
    counts: np.ndarray,
    ess: float = 5.0,
) -> np.ndarray:
    """Blend ML prediction (prior) with observation counts via Dirichlet update.

    prior_alpha = ml_pred * ess  (effective sample size of the ML prior)
    posterior_alpha = prior_alpha + counts
    posterior_pred = posterior_alpha / sum(posterior_alpha)

    With 10 observations per cell and ESS=5, the posterior is ~67% observations
    and ~33% ML prior. For cells with 0 observations, it's 100% ML prior.

    Args:
        ml_pred: Shape (H, W, K) float64 — ML predicted probabilities.
        counts: Shape (H, W, K) int32 — accumulated observation counts.
        ess: Effective sample size of the ML prior (controls blend weight).

    Returns:
        Blended prediction tensor of shape (H, W, K).
    """
    alpha_prior = ml_pred * ess
    alpha_post = alpha_prior + counts.astype(np.float64)
    denom = alpha_post.sum(axis=-1, keepdims=True)
    # Avoid division by zero (shouldn't happen with positive prior)
    denom = np.maximum(denom, 1e-12)
    return alpha_post / denom


# ---------------------------------------------------------------------------
# Adaptive blending gate
# ---------------------------------------------------------------------------

# Entropy threshold below which we skip blending (ML is confident enough).
# Calibrated from empirical data: Round #19 seeds had mean entropy ~0.6 and
# scored ~90 with ML-only but dropped to ~78 after blending.  Round #20 seeds
# had mean entropy ~1.2 and benefited from blending (+4.5 pts).
# Using 0.8 as a conservative split: below → keep ML, above → blend.
BLEND_ENTROPY_THRESHOLD: float = 0.8


def _mean_dynamic_entropy(
    pred: np.ndarray,
    raw_grid: list[list[int]],
) -> float:
    """Compute mean per-cell entropy of the ML prediction on dynamic cells only.

    Dynamic cells are those NOT in {ocean (10), mountain (5)} — i.e. cells whose
    outcome depends on hidden simulation parameters.  Static cells (ocean stays
    class 0, mountain stays class 5) are excluded because their entropy is
    artificially low and would drag the mean down.

    Args:
        pred: (H, W, K) calibrated ML prediction.
        raw_grid: Raw terrain grid (list of rows of terrain codes).

    Returns:
        Mean entropy across dynamic cells.  Returns inf if no dynamic cells.
    """
    grid_arr = np.array(raw_grid, dtype=np.int32)
    # Dynamic = not ocean (10) and not mountain (5)
    dynamic_mask = (grid_arr != 10) & (grid_arr != 5)
    if not dynamic_mask.any():
        return float("inf")

    # Per-cell entropy: -sum(p * log(p))
    safe_pred = np.clip(pred, 1e-15, 1.0)
    cell_entropy = -np.sum(safe_pred * np.log(safe_pred), axis=-1)  # (H, W)
    return float(cell_entropy[dynamic_mask].mean())


# ---------------------------------------------------------------------------
# Observe and submit — uses simulation queries to improve predictions
# ---------------------------------------------------------------------------


def observe_and_submit(*, queries_per_seed: int = 10) -> None:
    """Fire simulation queries, blend observations with ML prior, and re-submit.

    For each seed:
      1. Generate ML prediction via predict_grid().
      2. Plan viewport positions for near-full grid coverage.
      3. Fire simulation queries and accumulate terrain counts.
      4. Blend ML prior with observed counts via Dirichlet update.
      5. Apply probability floor, validate, and submit.

    Args:
        queries_per_seed: Number of simulation queries per seed (default 10).
    """
    from astar_island.api import (
        get_active_round,
        get_round_detail,
        make_client,
        simulate_query,
        submit_prediction,
    )

    with make_client() as client:
        active = get_active_round(client)
        if active is None:
            print("No active round found.")
            return

        round_id = str(active["id"])
        round_number = active.get("round_number", "?")
        print(f"Active round: #{round_number} ({round_id})")

        detail = get_round_detail(client, round_id)
        seeds_count = int(str(detail["seeds_count"]))
        initial_states: object = detail.get("initial_states")
        if not isinstance(initial_states, list):
            print("ERROR: No initial_states in round detail.")
            return

        width = int(str(detail.get("map_width", 40)))
        height = int(str(detail.get("map_height", 40)))
        print(f"Map: {width}x{height}, seeds: {seeds_count}")
        print(f"Queries per seed: {queries_per_seed}")

        results: list[tuple[int, str]] = []
        total_queries_used = 0

        for seed_idx in range(seeds_count):
            if seed_idx >= len(initial_states):
                print(f"\n  Seed {seed_idx}: no initial_state, skipping")
                continue

            state = initial_states[seed_idx]
            if not isinstance(state, dict):
                print(f"\n  Seed {seed_idx}: invalid state format, skipping")
                continue

            raw_grid: object = state.get("grid")
            if not isinstance(raw_grid, list):
                print(f"\n  Seed {seed_idx}: no grid, skipping")
                continue

            # Step 1: ML prediction
            print(f"\n  Seed {seed_idx}: ML predict...", end=" ", flush=True)
            ml_pred = predict_grid(raw_grid)
            print("done")

            # Step 2: Plan viewports
            viewports = _plan_viewports(width, height, queries_per_seed)

            # Step 3: Fire queries and accumulate counts
            counts = np.zeros((height, width, NUM_PREDICTION_CLASSES), dtype=np.int32)
            queries_fired = 0

            for qi, (vx, vy) in enumerate(viewports):
                print(
                    f"    Query {qi + 1}/{len(viewports)}: viewport ({vx},{vy})...",
                    end=" ",
                    flush=True,
                )
                try:
                    resp = simulate_query(client, round_id, seed_idx, vx, vy)
                    grid_rows: object = resp.get("grid")
                    if not isinstance(grid_rows, list):
                        print("bad response, skipping")
                        continue
                    _accumulate_observation(counts, grid_rows, vx, vy)
                    queries_fired += 1
                    budget_used = resp.get("queries_used", "?")
                    budget_max = resp.get("queries_max", "?")
                    print(f"ok (budget: {budget_used}/{budget_max})")
                except (httpx.HTTPStatusError, RuntimeError, TypeError) as exc:
                    print(f"FAILED: {exc}")

            total_queries_used += queries_fired
            print(f"    Observations: {queries_fired} queries fired")

            # Step 4: Adaptive blending gate
            # Only blend when ML prediction has high entropy (model is uncertain).
            # When model is confident, observations add noise (Round #19: −12 pts).
            seed_entropy = _mean_dynamic_entropy(ml_pred, raw_grid)
            if seed_entropy < BLEND_ENTROPY_THRESHOLD:
                print(
                    f"    Entropy {seed_entropy:.3f} < {BLEND_ENTROPY_THRESHOLD} "
                    f"→ SKIP blending (ML confident)"
                )
                blended = ml_pred
            else:
                print(
                    f"    Entropy {seed_entropy:.3f} ≥ {BLEND_ENTROPY_THRESHOLD} "
                    f"→ blending with observations"
                )
                blended = _dirichlet_blend(ml_pred, counts, ess=2.0)

            # Step 5: Floor, validate, submit
            safe_pred = apply_probability_floor(blended, raw_grid)
            errors = validate_prediction(safe_pred, width, height)
            if errors:
                print("    VALIDATION FAILED:")
                for err in errors:
                    print(f"      - {err}")
                results.append((seed_idx, "validation_failed"))
                continue

            pred_list: list[list[list[float]]] = safe_pred.tolist()
            try:
                resp_submit = submit_prediction(client, round_id, seed_idx, pred_list)
                status = resp_submit.get("status", "unknown")
                print(f"    Submitted — {status}")
                results.append((seed_idx, str(status)))
            except (httpx.HTTPStatusError, RuntimeError, TypeError) as exc:
                print(f"    Submit FAILED: {exc}")
                results.append((seed_idx, f"error: {exc}"))

        # Summary
        print(f"\n{'=' * 50}")
        print(f"Round #{round_number} — OBSERVE & SUBMIT")
        print(f"Total queries used: {total_queries_used}")
        for seed_idx, status in results:
            print(f"  Seed {seed_idx}: {status}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: solver.py [train|predict|evaluate|submit]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        train()
    elif command == "evaluate":
        evaluate()
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: solver.py predict <grid_json_path>")
            sys.exit(1)
        grid_path = Path(sys.argv[2])
        raw_grid = json.loads(grid_path.read_text())
        prediction = predict_grid(raw_grid)
        output_path = grid_path.with_suffix(".prediction.json")
        output_path.write_text(json.dumps(prediction.tolist()))
        print(f"Prediction saved to {output_path}")
    elif command == "submit":
        dry_run = "--dry-run" in sys.argv
        submit(dry_run=dry_run)
    elif command == "observe":
        observe_and_submit()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

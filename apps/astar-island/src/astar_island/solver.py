"""HybridSolver: full orchestrator for offline training and live round solving.

Implements the three-phase query loop:
  Phase 1: Broad coverage viewports for initial calibration
  Phase 2: EIG-guided targeted refinement
  Phase 3: Exploitation of remaining uncertainty

Offline training builds the prior model and historical archetype tables.
Live solving uses cross-seed empirical Bayes to calibrate the prior per round.
"""

from __future__ import annotations

import logging

import httpx
import numpy as np
from numpy.typing import NDArray

from astar_island.api import (
    API_BASE,
    RoundData,
    _make_client,
    download_all,
    get_active_round,
    load_all_rounds,
    save_prediction,
    simulate_query,
    submit_prediction,
)
from astar_island.archetypes import classify_archetypes
from astar_island.calibration import (
    RoundCalibrator,
    compute_archetype_frequencies,
)
from astar_island.config import Config
from astar_island.features import extract_features
from astar_island.planner import (
    allocate_budget,
    plan_next_query,
    plan_phase1,
)
from astar_island.posterior import (
    accumulate_counts,
    compute_ess,
    compute_posterior,
    init_alpha,
    observation_count_map,
    posterior_predictive,
)
from astar_island.prior import PriorModel, prepare_training_data
from astar_island.types import (
    N_SEEDS,
    STATIC_CODES,
    VIEWPORT,
    H,
    K,
    W,
)

logger = logging.getLogger(__name__)


def _cast_rounds_int32(
    rounds: RoundData,
) -> dict[int, list[tuple[NDArray[np.int32], NDArray[np.float64]]]]:
    """Cast RoundData grids from np.int_ to np.int32 for downstream use."""
    result: dict[int, list[tuple[NDArray[np.int32], NDArray[np.float64]]]] = {}
    for rnum, seeds in rounds.items():
        cast_seeds: list[tuple[NDArray[np.int32], NDArray[np.float64]]] = []
        for grid, gt in seeds:
            cast_seeds.append((np.asarray(grid, dtype=np.int32), gt))
        result[rnum] = cast_seeds
    return result


class HybridSolver:
    """Full hybrid solver: offline training + live round solving."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.prior = PriorModel(self.config)
        self._theta_hist: NDArray[np.float64] | None = None
        self._round_ids: list[int] | None = None
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Offline training
    # ------------------------------------------------------------------

    def train_offline(self, rounds: RoundData | None = None) -> None:
        """Train the prior model and build historical archetype tables.

        If rounds is None, loads from the local cache via api.load_all_rounds().
        """
        if rounds is None:
            rounds = load_all_rounds()

        if not rounds:
            msg = "No historical rounds available for training"
            raise RuntimeError(msg)

        # Cast np.int_ grids to np.int32 for prepare_training_data
        rounds_i32 = _cast_rounds_int32(rounds)

        # Prepare training data from round dict
        features, labels, round_ids, weights = prepare_training_data(
            rounds_i32,
            self.config,
        )

        logger.info(
            "Training prior on %d samples from %d rounds",
            len(labels),
            len(rounds),
        )
        self.prior.train(features, labels, round_ids, weights)

        # Build historical archetype frequency tables
        theta_hist, hist_round_ids = compute_archetype_frequencies(rounds)
        self._theta_hist = theta_hist
        self._round_ids = hist_round_ids

        logger.info(
            "Built archetype tables: shape %s, %d rounds",
            theta_hist.shape,
            len(hist_round_ids),
        )
        self._trained = True

    # ------------------------------------------------------------------
    # Static/dynamic mask helpers
    # ------------------------------------------------------------------

    def _static_mask(self, grid: NDArray[np.int32]) -> NDArray[np.bool_]:
        """Return (H, W) boolean mask of static cells."""
        result: NDArray[np.bool_] = np.zeros((H, W), dtype=np.bool_)
        for code in STATIC_CODES:
            result |= np.asarray(grid == code, dtype=np.bool_)
        return result

    # ------------------------------------------------------------------
    # Viewport observation → count tensor
    # ------------------------------------------------------------------

    def _parse_observation(
        self,
        response: dict[str, object],
        vy: int,
        vx: int,
    ) -> NDArray[np.int32]:
        """Parse a simulation response into a count tensor update.

        Extracts terrain grid codes via posterior.accumulate_counts and logs
        settlement-level statistics for diagnostics.
        """
        counts: NDArray[np.int32] = np.zeros((H, W, K), dtype=np.int32)
        raw_grid: object = response.get("grid")
        if not isinstance(raw_grid, list):
            logger.warning("No grid in simulation response")
            return counts

        # Convert nested list to numpy array
        grid_rows: list[list[int]] = []
        for row in raw_grid:
            if isinstance(row, list):
                grid_rows.append([int(v) for v in row if isinstance(v, (int, float))])

        if not grid_rows:
            return counts

        grid_obs: NDArray[np.int32] = np.array(
            grid_rows,
            dtype=np.int32,
        )
        vh, vw = grid_obs.shape

        # Parse settlement objects for diagnostic logging
        self._log_settlement_stats(response, vy, vx)

        return accumulate_counts(counts, grid_obs, vx, vy, vw, vh)

    @staticmethod
    def _log_settlement_stats(response: dict[str, object], vy: int, vx: int) -> None:
        """Log settlement stats from simulation response for diagnostics."""
        raw_settlements: object = response.get("settlements")
        if not isinstance(raw_settlements, list) or not raw_settlements:
            return

        alive_count = 0
        dead_count = 0
        port_count = 0
        total_pop = 0.0
        starved_count = 0

        for s in raw_settlements:
            if not isinstance(s, dict):
                continue
            alive = s.get("alive", True)
            if alive:
                alive_count += 1
            else:
                dead_count += 1
            if s.get("has_port", False):
                port_count += 1
            pop = s.get("population", 0)
            if isinstance(pop, (int, float)):
                total_pop += float(pop)
            food = s.get("food", 1)
            if isinstance(food, (int, float)) and food <= 0:
                starved_count += 1

        logger.debug(
            "Viewport (%d,%d) settlements: %d alive, %d dead, %d ports, "
            "total_pop=%.0f, %d starved",
            vx,
            vy,
            alive_count,
            dead_count,
            port_count,
            total_pop,
            starved_count,
        )

    # ------------------------------------------------------------------
    # Solve a single seed
    # ------------------------------------------------------------------

    def _solve_seed(
        self,
        client: httpx.Client,
        round_id: str,
        seed_index: int,
        initial_grid: NDArray[np.int32],
        calibrator: RoundCalibrator,
        p1_queries: int,
        p2_queries: int,
        p3_queries: int,
        phase1_plan: list[list[tuple[int, int]]],
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Run the 3-phase query loop for one seed.

        Returns: (prediction tensor, accumulated counts)
        """
        dynamic = ~self._static_mask(initial_grid)

        # Get learned prior prediction
        feats = extract_features(initial_grid, self.config)
        p_prior = self.prior.predict_grid(feats, initial_grid)

        # Initial round template from calibrator
        p_template = calibrator.get_template(seed_index)

        # Initialise count tensor and observation tracker
        counts: NDArray[np.int32] = np.zeros(
            (H, W, K),
            dtype=np.int32,
        )
        n_obs: NDArray[np.int32] = np.zeros((H, W), dtype=np.int32)
        queries_used = 0

        # Compute initial posterior and alpha for EIG
        pred = compute_posterior(
            p_prior,
            p_template,
            counts,
            initial_grid,
            self.config,
        )
        ess = compute_ess(pred, self.config)
        alpha = init_alpha(pred, ess) + counts.astype(np.float64)

        # Phase 1: Broad coverage
        for q_idx in range(p1_queries):
            vx, vy = plan_next_query(
                phase=1,
                seed_index=seed_index,
                pred=pred,
                alpha=alpha,
                n_obs=n_obs,
                dynamic=dynamic,
                phase1_plan=phase1_plan,
                phase1_idx=q_idx,
            )
            response = simulate_query(
                client,
                round_id,
                seed_index,
                vx,
                vy,
                VIEWPORT,
            )
            obs_counts = self._parse_observation(response, vy, vx)
            counts += obs_counts
            n_obs = observation_count_map(counts)
            alpha += obs_counts.astype(np.float64)
            queries_used += 1
            logger.debug(
                "Seed %d Phase1 query %d: vx=%d vy=%d",
                seed_index,
                queries_used,
                vx,
                vy,
            )

        # Recompute posterior after Phase 1
        p_template = calibrator.get_template(seed_index)
        pred = compute_posterior(
            p_prior,
            p_template,
            counts,
            initial_grid,
            self.config,
        )
        ess = compute_ess(pred, self.config)
        alpha = init_alpha(pred, ess) + counts.astype(np.float64)

        # Phase 2: EIG-guided refinement
        for _ in range(p2_queries):
            vx, vy = plan_next_query(
                phase=2,
                seed_index=seed_index,
                pred=pred,
                alpha=alpha,
                n_obs=n_obs,
                dynamic=dynamic,
            )
            response = simulate_query(
                client,
                round_id,
                seed_index,
                vx,
                vy,
                VIEWPORT,
            )
            obs_counts = self._parse_observation(response, vy, vx)
            counts += obs_counts
            n_obs = observation_count_map(counts)
            alpha += obs_counts.astype(np.float64)
            # Recompute pred from updated alpha so next EIG uses fresh entropy
            pred = posterior_predictive(alpha)
            queries_used += 1
            logger.debug(
                "Seed %d Phase2 query %d: vx=%d vy=%d",
                seed_index,
                queries_used,
                vx,
                vy,
            )

        # Phase 3: Exploitation
        for _ in range(p3_queries):
            vx, vy = plan_next_query(
                phase=3,
                seed_index=seed_index,
                pred=pred,
                alpha=alpha,
                n_obs=n_obs,
                dynamic=dynamic,
            )
            response = simulate_query(
                client,
                round_id,
                seed_index,
                vx,
                vy,
                VIEWPORT,
            )
            obs_counts = self._parse_observation(response, vy, vx)
            counts += obs_counts
            n_obs = observation_count_map(counts)
            alpha += obs_counts.astype(np.float64)
            # Recompute pred so next exploitation query sees fresh entropy
            pred = posterior_predictive(alpha)
            queries_used += 1
            logger.debug(
                "Seed %d Phase3 query %d: vx=%d vy=%d",
                seed_index,
                queries_used,
                vx,
                vy,
            )

        # Final prediction via full pipeline
        p_template = calibrator.get_template(seed_index)
        pred = compute_posterior(
            p_prior,
            p_template,
            counts,
            initial_grid,
            self.config,
        )

        logger.info(
            "Seed %d: %d queries used",
            seed_index,
            queries_used,
        )
        return pred, counts

    # ------------------------------------------------------------------
    # Solve a full round (all seeds)
    # ------------------------------------------------------------------

    def solve_round(
        self,
        client: httpx.Client,
        round_id: str,
        initial_grids: list[NDArray[np.int32]],
    ) -> list[NDArray[np.float64]]:
        """Solve all seeds of a round with cross-seed calibration.

        Args:
            client: Authenticated httpx.Client.
            round_id: The UUID of the active round.
            initial_grids: List of (H, W) initial grid arrays per seed.

        Returns: List of (H, W, K) prediction tensors, one per seed.
        """
        if not self._trained:
            msg = "Solver not trained — call train_offline() first"
            raise RuntimeError(msg)

        if self._theta_hist is None or self._round_ids is None:
            msg = "Historical tables not built"
            raise RuntimeError(msg)

        n_seeds = len(initial_grids)

        # Build archetype maps for all seeds
        archetype_maps: list[NDArray[np.int32]] = [
            classify_archetypes(g) for g in initial_grids
        ]

        # Build dynamic mask (shared structure for planning)
        dynamic = ~self._static_mask(initial_grids[0])

        # Initialize calibrator with historical data + seed archetypes
        calibrator = RoundCalibrator(
            self._theta_hist,
            self._round_ids,
            archetype_maps,
            self.config,
        )

        # Allocate budget across phases and seeds
        p1_per_seed, p2_per_seed, p3_per_seed = allocate_budget(
            self.config,
        )

        # Plan Phase 1 viewports for all seeds
        phase1_plan = plan_phase1(
            initial_grids[0],
            dynamic,
            n_seeds,
            queries_per_seed=p1_per_seed[0] if p1_per_seed else 2,
        )

        # Solve each seed — fault tolerance: failed seeds get uniform prediction
        predictions: list[NDArray[np.float64]] = []
        all_counts: list[NDArray[np.int32]] = []

        for s in range(n_seeds):
            p1_q = p1_per_seed[s] if s < len(p1_per_seed) else 2
            p2_q = p2_per_seed[s] if s < len(p2_per_seed) else 6
            p3_q = p3_per_seed[s] if s < len(p3_per_seed) else 2

            logger.info(
                "Solving seed %d/%d (%d+%d+%d queries)",
                s + 1,
                n_seeds,
                p1_q,
                p2_q,
                p3_q,
            )
            try:
                pred, counts = self._solve_seed(
                    client=client,
                    round_id=round_id,
                    seed_index=s,
                    initial_grid=initial_grids[s],
                    calibrator=calibrator,
                    p1_queries=p1_q,
                    p2_queries=p2_q,
                    p3_queries=p3_q,
                    phase1_plan=phase1_plan,
                )
            except Exception:
                logger.exception("Seed %d failed — using uniform prediction", s)
                pred = np.full((H, W, K), 1.0 / K, dtype=np.float64)
                counts = np.zeros((H, W, K), dtype=np.int32)
            predictions.append(pred)
            all_counts.append(counts)

            # Cross-seed recalibration after each seed
            calibrator.update(all_counts)
            rid, wt = calibrator.top_round_match
            logger.debug(
                "After seed %d, top match: round=%d weight=%.4f",
                s,
                rid,
                wt,
            )

        # Save predictions locally
        for s, pred in enumerate(predictions):
            save_prediction(round_id, s, pred)

        logger.info("Round %s solved — %d seeds", round_id, n_seeds)
        return predictions


def main() -> None:
    """CLI entry point: train offline, then solve the active round."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="A* Island hybrid solver")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit predictions to the competition API after solving",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain even if a saved model exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    model_path = (
        Path(__file__).resolve().parents[2] / "data" / "models" / "prior_v1.joblib"
    )
    config = Config()
    solver = HybridSolver(config)

    if model_path.exists() and not args.retrain:
        logger.info("Loading saved model from %s", model_path)
        solver.prior = PriorModel.load(model_path, config)
        rounds = load_all_rounds()
        if not rounds:
            logger.error(
                "No historical rounds found in cache. Run download_all() first."
            )
            sys.exit(1)
        theta_hist, round_ids = compute_archetype_frequencies(rounds)
        solver._theta_hist = theta_hist
        solver._round_ids = round_ids
        solver._trained = True
        logger.info("Model loaded, archetype tables built (%d rounds)", len(round_ids))
    else:
        logger.info("Training model from scratch")
        rounds = load_all_rounds()
        if not rounds:
            logger.info("No cached data. Downloading historical rounds...")
            download_all()
            rounds = load_all_rounds()
        if not rounds:
            logger.error("No rounds available for training.")
            sys.exit(1)
        solver.train_offline(rounds)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        solver.prior.save(model_path)
        logger.info("Model trained and saved to %s", model_path)

    client = _make_client()
    active = get_active_round(client)
    if active is None:
        logger.info("No active round found. Nothing to solve.")
        client.close()
        sys.exit(0)

    round_id = str(active.get("id", ""))
    round_number = active.get("round_number", "?")
    if not round_id:
        logger.error("Active round has no 'id' field: %s", active)
        client.close()
        sys.exit(1)

    logger.info("Active round: #%s (id=%s)", round_number, round_id)

    initial_grids: list[NDArray[np.int32] | None] = []

    # For active rounds, /analysis returns 400 — use /rounds/{id} instead
    round_detail_url = f"{API_BASE}/rounds/{round_id}"
    try:
        detail_resp = client.get(round_detail_url)
        detail_resp.raise_for_status()
        detail_data = detail_resp.json()
        initial_states = detail_data.get("initial_states", [])
    except Exception:
        logger.exception("Failed to fetch round detail")
        initial_states = []

    if initial_states and len(initial_states) >= N_SEEDS:
        for seed_idx in range(N_SEEDS):
            try:
                grid_data = initial_states[seed_idx]["grid"]
                grid = np.array(grid_data, dtype=np.int32)
                initial_grids.append(grid)
                logger.info(
                    "Fetched initial grid for seed %d from round detail: shape %s",
                    seed_idx,
                    grid.shape,
                )
            except Exception:
                logger.exception("Failed to parse grid for seed %d", seed_idx)
                initial_grids.append(None)
    else:
        # Fallback: try /analysis endpoint (works for completed rounds)
        for seed_idx in range(N_SEEDS):
            url = f"{API_BASE}/analysis/{round_id}/{seed_idx}"
            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.error(
                        "Failed to fetch initial grid for seed %d: HTTP %d",
                        seed_idx,
                        resp.status_code,
                    )
                    initial_grids.append(None)
                    continue
                data = resp.json()
                if not isinstance(data, dict) or "initial_grid" not in data:
                    logger.error(
                        "No initial_grid in analysis response for seed %d", seed_idx
                    )
                    initial_grids.append(None)
                    continue
                grid = np.array(data["initial_grid"], dtype=np.int32)
                initial_grids.append(grid)
                logger.info(
                    "Fetched initial grid for seed %d: shape %s", seed_idx, grid.shape
                )
            except Exception:
                logger.exception(
                    "Exception fetching initial grid for seed %d", seed_idx
                )
                initial_grids.append(None)

    # Build non-None grid list; use uniform predictions for failed seeds
    valid_grids: list[NDArray[np.int32]] = []
    valid_indices: list[int] = []
    for idx, g in enumerate(initial_grids):
        if g is not None:
            valid_grids.append(g)
            valid_indices.append(idx)

    if not valid_grids:
        logger.error("All seed grids failed to fetch. Cannot solve.")
        client.close()
        sys.exit(1)

    logger.info(
        "Starting solver for round %s with %d/%d valid seeds",
        round_id,
        len(valid_grids),
        N_SEEDS,
    )
    valid_predictions = solver.solve_round(client, round_id, valid_grids)

    # Assemble full prediction list: uniform for failed seeds, solved for valid
    uniform: NDArray[np.float64] = np.full(
        (H, W, K),
        1.0 / K,
        dtype=np.float64,
    )
    predictions: list[NDArray[np.float64]] = []
    valid_iter = iter(valid_predictions)
    for idx in range(N_SEEDS):
        if idx in valid_indices:
            predictions.append(next(valid_iter))
        else:
            logger.warning("Seed %d: using uniform prediction (fetch failed)", idx)
            predictions.append(uniform.copy())

    if args.submit:
        logger.info("Submitting %d predictions to the API...", len(predictions))
        for s, pred in enumerate(predictions):
            submit_prediction(client, round_id, s, pred)
        logger.info("All %d seeds submitted.", len(predictions))
    else:
        logger.info(
            "Done. %d predictions saved locally. Use --submit to send to the API.",
            len(predictions),
        )

    client.close()


if __name__ == "__main__":
    main()

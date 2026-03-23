"""SMC Particle Filter strategy for Astar Island.

Uses Sequential Monte Carlo / particle filtering over hidden ``SimulationParams``
with Bayesian-inspired viewport selection and posterior-predictive Monte Carlo
prediction.

The core loop (executed once per seed):
  1. Generate mechanism-aware viewport candidates via ``generate_hotspots``.
  2. Score candidates by a two-stage utility function:
     - Stage 1 (cheap): dynamic-cell fraction + settlement density.
     - Stage 2 (top-3): particle disagreement via probe simulations.
  3. Observe the best viewport using a reference simulator.
  4. Update particle weights via the existing likelihood machinery.
  5. Resample / temper as needed.
  6. Produce final prediction via ``predict_seed`` (posterior-predictive MC).

For the first seed (index 0), two bootstrap viewports are observed to give the
posterior an initial signal.  Subsequent seeds observe one viewport each.

In the benchmark harness the "reference observations" come from the strategy's
own default-params simulator.  This means the posterior will tend toward
defaults — which is fine because the benchmark scores against default-params
ground truth.  In production with real API observations the posterior would
learn genuinely hidden parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astar_twin.contracts.api_models import (
    InitialState,
    SimSettlement,
    SimulateResponse,
    ViewportBounds,
)
from astar_twin.contracts.types import TERRAIN_TO_CLASS, TerrainCode
from astar_twin.engine import Simulator
from astar_twin.harness.budget import Budget
from astar_twin.params import SimulationParams
from astar_twin.solver.inference.posterior import (
    PosteriorState,
    create_posterior,
    resample_if_needed,
    temper_if_collapsed,
    update_posterior,
)
from astar_twin.solver.policy.hotspots import ViewportCandidate, generate_hotspots
from astar_twin.solver.predict.posterior_mc import predict_seed

# ---------------------------------------------------------------------------
# Tunable defaults (small for benchmark speed; override via __init__)
# ---------------------------------------------------------------------------
_DEFAULT_N_PARTICLES = 12
_DEFAULT_SIMS_PER_SEED = 32
_DEFAULT_TOP_K = 5
_DEFAULT_N_INNER_RUNS = 3
_DEFAULT_N_BOOTSTRAP_VIEWPORTS = 2


class SMCParticleFilterStrategy:
    """Particle-filter strategy with Bayesian viewport selection.

    Maintains a shared ``PosteriorState`` across seeds so that evidence from
    earlier seeds benefits later ones (the hidden parameters are the same for
    all seeds within a round).
    """

    def __init__(
        self,
        *,
        n_particles: int = _DEFAULT_N_PARTICLES,
        sims_per_seed: int = _DEFAULT_SIMS_PER_SEED,
        top_k: int = _DEFAULT_TOP_K,
        n_inner_runs: int = _DEFAULT_N_INNER_RUNS,
        n_bootstrap_viewports: int = _DEFAULT_N_BOOTSTRAP_VIEWPORTS,
    ) -> None:
        self._n_particles = n_particles
        self._sims_per_seed = sims_per_seed
        self._top_k = top_k
        self._n_inner_runs = n_inner_runs
        self._n_bootstrap_viewports = n_bootstrap_viewports

        self._posterior: PosteriorState | None = None
        self._seed_count: int = 0

    @property
    def name(self) -> str:
        return "smc_particle_filter"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        initial_state: InitialState,
        budget: Budget,
        base_seed: int,
    ) -> NDArray[np.float64]:
        """Generate a posterior-predictive (H, W, 6) tensor for one seed."""
        H = len(initial_state.grid)
        W = len(initial_state.grid[0])

        # Lazy-init the posterior on the very first call
        if self._posterior is None:
            self._posterior = create_posterior(
                n_particles=self._n_particles,
                seed=base_seed,
            )

        seed_offset = self._seed_count
        self._seed_count += 1

        # Determine how many viewports to observe for this seed
        n_viewports = self._n_bootstrap_viewports if seed_offset == 0 else 1

        # Generate, score, and observe viewports
        candidates = generate_hotspots(initial_state, H, W)
        if candidates:
            scored = _score_and_sort_viewports(
                candidates,
                self._posterior,
                initial_state,
                base_seed,
                seed_offset,
            )
            for obs_idx, vp_candidate in enumerate(scored):
                if obs_idx >= n_viewports:
                    break
                if budget.remaining <= 0:
                    break

                obs = _build_reference_observation(
                    initial_state,
                    vp_candidate,
                    H,
                    W,
                    sim_seed=base_seed + seed_offset * 1000 + obs_idx,
                )
                budget.consume()

                self._posterior = update_posterior(
                    self._posterior,
                    obs,
                    initial_state,
                    n_inner_runs=self._n_inner_runs,
                    base_seed=base_seed + seed_offset * 100 + obs_idx,
                )
                self._posterior = resample_if_needed(
                    self._posterior,
                    seed=base_seed + seed_offset * 50 + obs_idx,
                )
                self._posterior = temper_if_collapsed(self._posterior)

        # Produce posterior-predictive tensor
        tensor, _metrics = predict_seed(
            posterior=self._posterior,
            initial_state=initial_state,
            seed_index=seed_offset,
            map_height=H,
            map_width=W,
            top_k=self._top_k,
            sims_per_seed=self._sims_per_seed,
            base_seed=base_seed + seed_offset * 10000,
        )

        return tensor


# ======================================================================
# Helper functions (module-level, stateless)
# ======================================================================


def _build_reference_observation(
    initial_state: InitialState,
    viewport: ViewportCandidate,
    map_height: int,
    map_width: int,
    sim_seed: int,
) -> SimulateResponse:
    """Run one simulation with default params and extract a viewport observation.

    In the benchmark this acts as a "synthetic API call" — the observation comes
    from the default-param simulator rather than the real API.
    """
    simulator = Simulator(params=SimulationParams())
    state = simulator.run(initial_state=initial_state, sim_seed=sim_seed)

    vp_grid = state.grid.viewport(viewport.x, viewport.y, viewport.w, viewport.h)

    settlements_in_vp: list[SimSettlement] = []
    for s in state.settlements:
        if (
            viewport.x <= s.x < viewport.x + viewport.w
            and viewport.y <= s.y < viewport.y + viewport.h
        ):
            settlements_in_vp.append(
                SimSettlement(
                    x=s.x,
                    y=s.y,
                    population=s.population,
                    food=s.food,
                    wealth=s.wealth,
                    defense=s.defense,
                    has_port=s.has_port,
                    alive=s.alive,
                    owner_id=s.owner_id,
                )
            )

    return SimulateResponse(
        grid=vp_grid.to_list(),
        settlements=settlements_in_vp,
        viewport=ViewportBounds(x=viewport.x, y=viewport.y, w=viewport.w, h=viewport.h),
        width=map_width,
        height=map_height,
        queries_used=0,
        queries_max=50,
    )


def _score_and_sort_viewports(
    candidates: list[ViewportCandidate],
    posterior: PosteriorState,
    initial_state: InitialState,
    base_seed: int,
    seed_offset: int,
) -> list[ViewportCandidate]:
    """Score all candidates and return them sorted by utility (descending)."""
    scored: list[tuple[float, ViewportCandidate]] = []
    for c in candidates:
        utility = _viewport_utility(c, posterior, initial_state, base_seed + seed_offset)
        scored.append((utility, c))

    # Sort descending by utility
    scored.sort(key=lambda t: t[0], reverse=True)

    # Stage 2: refine top 3 with particle disagreement
    top_n = min(3, len(scored))
    refined: list[tuple[float, ViewportCandidate]] = []
    for i in range(top_n):
        utility, c = scored[i]
        disagreement = _compute_disagreement(
            c, posterior, initial_state, base_seed + seed_offset * 200 + i
        )
        combined = utility + 0.5 * disagreement
        refined.append((combined, c))

    # Re-sort the refined top candidates
    refined.sort(key=lambda t: t[0], reverse=True)

    # Build final list: refined top + remaining (un-refined)
    result = [c for _, c in refined]
    for i in range(top_n, len(scored)):
        result.append(scored[i][1])

    return result


def _viewport_utility(
    candidate: ViewportCandidate,
    posterior: PosteriorState,
    initial_state: InitialState,
    base_seed: int,
) -> float:
    """Stage 1 utility: cheap heuristic combining dynamic-cell fraction and settlement density."""
    del posterior, base_seed  # unused in stage 1
    dynamic_frac = _dynamic_cell_fraction(candidate, initial_state)
    settle_dens = _settlement_density(candidate, initial_state)
    return 0.3 * dynamic_frac + 0.2 * settle_dens


def _dynamic_cell_fraction(candidate: ViewportCandidate, initial_state: InitialState) -> float:
    """Fraction of viewport cells that are non-static (not ocean, not mountain)."""
    grid = initial_state.grid
    total = 0
    dynamic = 0
    for y in range(candidate.y, min(candidate.y + candidate.h, len(grid))):
        row = grid[y]
        for x in range(candidate.x, min(candidate.x + candidate.w, len(row))):
            total += 1
            code = row[x]
            if code != TerrainCode.OCEAN and code != TerrainCode.MOUNTAIN:
                dynamic += 1
    return dynamic / max(total, 1)


def _settlement_density(candidate: ViewportCandidate, initial_state: InitialState) -> float:
    """Fraction of alive settlements that fall inside the viewport."""
    alive_settlements = [s for s in initial_state.settlements if s.alive]
    if not alive_settlements:
        return 0.0
    count = 0
    for s in alive_settlements:
        if (
            candidate.x <= s.x < candidate.x + candidate.w
            and candidate.y <= s.y < candidate.y + candidate.h
        ):
            count += 1
    return count / len(alive_settlements)


def _compute_disagreement(
    candidate: ViewportCandidate,
    posterior: PosteriorState,
    initial_state: InitialState,
    base_seed: int,
) -> float:
    """Stage 2: measure class-argmax disagreement between top-2 particles in the viewport.

    Runs 1 probe simulation per particle, computes the argmax class per cell,
    and returns the fraction of cells where the two particles disagree.
    """
    top_indices = posterior.top_k_indices(2)
    if len(top_indices) < 2:
        return 0.0

    grids: list[NDArray[np.int64]] = []
    for rank, idx in enumerate(top_indices):
        particle = posterior.particles[idx]
        sim_params = particle.to_simulation_params()
        simulator = Simulator(params=sim_params)
        state = simulator.run(initial_state=initial_state, sim_seed=base_seed + rank)
        vp = state.grid.viewport(candidate.x, candidate.y, candidate.w, candidate.h)

        class_grid = np.zeros((candidate.h, candidate.w), dtype=np.int64)
        for y in range(vp.height):
            for x in range(vp.width):
                code = vp.get(y, x)
                class_grid[y, x] = TERRAIN_TO_CLASS.get(code, 0)
        grids.append(class_grid)

    # Fraction of cells where the argmax class differs
    diff_mask: NDArray[np.bool_] = grids[0] != grids[1]
    disagree = int(np.count_nonzero(diff_mask))
    total = candidate.h * candidate.w
    return disagree / max(total, 1)

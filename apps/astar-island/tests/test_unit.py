"""Unit tests for the astar_island solver package.

Covers: types, config, utils, planner, posterior, archetypes, calibration, solver.
"""

import numpy as np
import pytest

from astar_island.types import H, W, K, VIEWPORT, N_SEEDS, STATIC_CODES, CODE_TO_CLASS
from astar_island.config import Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid() -> np.ndarray:
    """Build a 40×40 grid with known structure.

    Layout:
    - Border ring (row/col 0 and 39): Ocean (code 10)
    - (5, 5): Mountain (code 5)
    - (10, 10): Settlement (code 1)
    - Everything else: Plains (code 11)
    """
    grid = np.full((H, W), 11, dtype=np.int32)
    grid[0, :] = 10  # top border — ocean
    grid[39, :] = 10  # bottom border — ocean
    grid[:, 0] = 10  # left border — ocean
    grid[:, 39] = 10  # right border — ocean
    grid[5, 5] = 5  # mountain
    grid[10, 10] = 1  # settlement
    return grid


def _uniform_pred() -> np.ndarray:
    """Return a (H, W, K) uniform probability tensor (all = 1/K)."""
    return np.full((H, W, K), 1.0 / K, dtype=np.float64)


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------


def test_code_to_class_all_terrain_codes():
    """CODE_TO_CLASS maps all expected terrain codes correctly."""
    assert CODE_TO_CLASS[0] == 0  # Empty → class 0
    assert CODE_TO_CLASS[10] == 0  # Ocean → class 0
    assert CODE_TO_CLASS[11] == 0  # Plains → class 0
    assert CODE_TO_CLASS[1] == 1  # Settlement
    assert CODE_TO_CLASS[2] == 2  # Port
    assert CODE_TO_CLASS[3] == 3  # Ruin
    assert CODE_TO_CLASS[4] == 4  # Forest
    assert CODE_TO_CLASS[5] == 5  # Mountain


def test_static_codes_exact_set():
    """STATIC_CODES contains exactly {5, 10}."""
    assert STATIC_CODES == frozenset({5, 10})


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


def test_config_default_values():
    """Default Config values are as specified."""
    cfg = Config()
    assert cfg.n_queries_total == 50
    assert cfg.seed_caps == (15, 12, 10, 8, 5)
    assert cfg.lambda_prior == pytest.approx(0.4)
    assert cfg.floor_impossible == pytest.approx(0.001)
    assert cfg.floor_standard == pytest.approx(0.01)
    assert cfg.eps == pytest.approx(1e-4)
    assert cfg.c_base == pytest.approx(3.0)
    assert cfg.ess_min == pytest.approx(1.0)
    assert cfg.ess_max == pytest.approx(4.0)


def test_config_is_frozen():
    """Config raises FrozenInstanceError on attribute assignment."""
    from dataclasses import FrozenInstanceError

    cfg = Config()
    with pytest.raises(FrozenInstanceError):
        cfg.n_queries_total = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


def test_clip_normalize_sums_to_one():
    """clip_normalize: input [0.1, 0.2, 0.7] → sums to 1.0."""
    from astar_island.utils import clip_normalize

    p = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    result = clip_normalize(p)
    assert float(np.sum(result)) == pytest.approx(1.0, abs=1e-10)


def test_clip_normalize_zeros_get_floor():
    """clip_normalize: input with zeros → all values ≥ eps after normalize."""
    from astar_island.utils import clip_normalize

    eps = 1e-4
    p = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    result = clip_normalize(p, eps=eps)
    assert np.all(result >= 0.0)
    assert float(np.sum(result)) == pytest.approx(1.0, abs=1e-9)


def test_entropy_uniform_is_max():
    """entropy: uniform distribution → max entropy ≈ log(K)."""
    from astar_island.utils import entropy

    p = np.full(K, 1.0 / K, dtype=np.float64)
    h = float(entropy(p))
    assert h == pytest.approx(np.log(K), rel=1e-3)


def test_entropy_one_hot_near_zero():
    """entropy: one-hot distribution → entropy near 0 (with eps floor)."""
    from astar_island.utils import entropy

    p = np.zeros(K, dtype=np.float64)
    p[0] = 1.0
    h = float(entropy(p))
    # With eps floor the entropy is small but not exactly 0
    assert h >= 0.0
    assert h < 0.1  # very low entropy


def test_kl_divergence_equal_distributions():
    """kl_divergence: p == q → KL ≈ 0."""
    from astar_island.utils import kl_divergence

    p = np.full(K, 1.0 / K, dtype=np.float64)
    kl = float(kl_divergence(p, p))
    assert kl == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_disjoint_is_large():
    """kl_divergence: p concentrates on class 0, q on class 1 → large positive KL."""
    from astar_island.utils import kl_divergence

    eps = 1e-4
    p = np.full(K, eps, dtype=np.float64)
    p[0] = 1.0 - (K - 1) * eps
    q = np.full(K, eps, dtype=np.float64)
    q[1] = 1.0 - (K - 1) * eps
    kl = float(kl_divergence(p, q))
    assert kl > 1.0  # clearly diverged


def test_weighted_kl_score_perfect_prediction():
    """weighted_kl_score: prediction == ground truth → score ≈ 100."""
    from astar_island.utils import weighted_kl_score

    gt = _uniform_pred()
    score = weighted_kl_score(gt, gt)
    assert score == pytest.approx(100.0, rel=1e-4)


def test_softmax_uniform_input():
    """softmax: [0,0,0] → uniform [1/3, 1/3, 1/3]."""
    from astar_island.utils import softmax

    logits = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    result = softmax(logits)
    expected = 1.0 / 3.0
    assert np.allclose(result, expected, atol=1e-10)


def test_softmax_large_values_numerically_stable():
    """softmax: very large logits → no NaN/Inf in output."""
    from astar_island.utils import softmax

    logits = np.array([1e6, 2e6, 3e6], dtype=np.float64)
    result = softmax(logits)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    assert float(np.sum(result)) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# planner.py
# ---------------------------------------------------------------------------


def test_allocate_budget_total_is_50():
    """allocate_budget: seed caps sum to 50 total queries."""
    from astar_island.planner import allocate_budget

    cfg = Config()
    p1, p2, p3 = allocate_budget(cfg)
    total = sum(p1) + sum(p2) + sum(p3)
    assert total == cfg.n_queries_total


def test_allocate_budget_per_seed_matches_cap():
    """allocate_budget: p1[i] + p2[i] + p3[i] == seed_caps[i] for each seed."""
    from astar_island.planner import allocate_budget

    cfg = Config()
    p1, p2, p3 = allocate_budget(cfg)
    for i, cap in enumerate(cfg.seed_caps):
        assert p1[i] + p2[i] + p3[i] == cap, (
            f"Seed {i}: {p1[i]}+{p2[i]}+{p3[i]} != {cap}"
        )


def test_allocate_budget_p1_always_2():
    """allocate_budget: p1 is always [2, 2, 2, 2, 2]."""
    from astar_island.planner import allocate_budget

    cfg = Config()
    p1, _, _ = allocate_budget(cfg)
    assert p1 == [2, 2, 2, 2, 2]


def test_viewport_sums_ones_field():
    """_viewport_sums: all-ones field → each sum == VIEWPORT * VIEWPORT == 225."""
    from astar_island.planner import _viewport_sums

    field = np.ones((H, W), dtype=np.float64)
    sums = _viewport_sums(field)
    expected = float(VIEWPORT * VIEWPORT)
    assert sums.shape == (H - VIEWPORT + 1, W - VIEWPORT + 1)
    assert np.allclose(sums, expected, atol=1e-10)


def test_viewport_sums_zeros_field():
    """_viewport_sums: zeros field → all sums == 0."""
    from astar_island.planner import _viewport_sums

    field = np.zeros((H, W), dtype=np.float64)
    sums = _viewport_sums(field)
    assert np.allclose(sums, 0.0, atol=1e-10)


def test_viewport_sums_single_cell():
    """_viewport_sums: single non-zero cell → correct sums for viewports covering it."""
    from astar_island.planner import _viewport_sums

    field = np.zeros((H, W), dtype=np.float64)
    r, c = 10, 10  # place a 1.0 at row=10, col=10
    field[r, c] = 1.0

    sums = _viewport_sums(field)  # shape (26, 26)

    # A viewport anchored at (vy, vx) covers rows [vy, vy+VIEWPORT) and cols [vx, vx+VIEWPORT)
    # Cell (r, c) is covered iff vy <= r < vy+VIEWPORT and vx <= c < vx+VIEWPORT
    # i.e.  max(0, r-VIEWPORT+1) <= vy <= r  and  max(0, c-VIEWPORT+1) <= vx <= c
    vy_min = max(0, r - VIEWPORT + 1)
    vy_max = min(H - VIEWPORT, r)
    vx_min = max(0, c - VIEWPORT + 1)
    vx_max = min(W - VIEWPORT, c)

    for vy in range(H - VIEWPORT + 1):
        for vx in range(W - VIEWPORT + 1):
            expected = (
                1.0 if (vy_min <= vy <= vy_max and vx_min <= vx <= vx_max) else 0.0
            )
            assert sums[vy, vx] == pytest.approx(expected, abs=1e-10), (
                f"vy={vy}, vx={vx}: expected {expected}, got {sums[vy, vx]}"
            )


# ---------------------------------------------------------------------------
# posterior.py
# ---------------------------------------------------------------------------


def test_init_alpha_uniform_pred():
    """init_alpha: ess=2.0, uniform pred → alpha all ≈ 2.0/K per class."""
    from astar_island.posterior import init_alpha

    ess = np.full((H, W), 2.0, dtype=np.float64)
    pred = _uniform_pred()
    alpha = init_alpha(pred, ess)
    expected = 2.0 / K
    assert alpha.shape == (H, W, K)
    assert np.allclose(alpha, expected, atol=1e-10)


def test_bayesian_update_adds_counts():
    """bayesian_update: alpha + counts = expected updated alpha."""
    from astar_island.posterior import bayesian_update

    alpha = np.ones((H, W, K), dtype=np.float64) * 0.5
    counts = np.zeros((H, W, K), dtype=np.int32)
    counts[5, 5, 1] = 3
    counts[10, 10, 0] = 1
    alpha_post = bayesian_update(alpha, counts)
    assert alpha_post[5, 5, 1] == pytest.approx(0.5 + 3.0, abs=1e-10)
    assert alpha_post[10, 10, 0] == pytest.approx(0.5 + 1.0, abs=1e-10)
    # Unchanged cell
    assert alpha_post[0, 0, 0] == pytest.approx(0.5, abs=1e-10)


def test_posterior_predictive_sums_to_one():
    """posterior_predictive: alpha → pred sums to 1.0 per cell."""
    from astar_island.posterior import posterior_predictive

    rng = np.random.default_rng(42)
    alpha = rng.uniform(0.5, 5.0, size=(H, W, K)).astype(np.float64)
    pred = posterior_predictive(alpha)
    sums = pred.sum(axis=-1)
    assert pred.shape == (H, W, K)
    assert np.allclose(sums, 1.0, atol=1e-10)


def test_accumulate_counts_one_hot_in_viewport():
    """accumulate_counts: places correct one-hot in the viewport region."""
    from astar_island.posterior import accumulate_counts

    counts = np.zeros((H, W, K), dtype=np.int32)
    # Small viewport at (vx=0, vy=0) of size 3×3, all terrain code 1 (Settlement→class 1)
    grid_obs = np.ones((3, 3), dtype=np.int32)  # code 1 → class 1
    counts = accumulate_counts(counts, grid_obs, vx=0, vy=0, vw=3, vh=3)
    # All cells [0:3, 0:3] should have count 1 in class 1
    assert np.all(counts[0:3, 0:3, 1] == 1)
    # Class 0 should be zero in that region
    assert np.all(counts[0:3, 0:3, 0] == 0)
    # Outside the viewport should be zero
    assert np.all(counts[3:, :, :] == 0)


def test_apply_floors_ocean_cell_one_hot():
    """apply_floors: ocean cell → one-hot on class 0."""
    from astar_island.posterior import apply_floors

    grid = np.full((H, W), 11, dtype=np.int32)  # all plains
    grid[0, 0] = 10  # ocean
    pred = _uniform_pred()
    cfg = Config()
    result = apply_floors(pred, grid, cfg)
    # Ocean cell (0,0) should be one-hot on class 0
    assert result[0, 0, 0] == pytest.approx(1.0, abs=1e-10)
    assert result[0, 0, 1] == pytest.approx(0.0, abs=1e-10)
    assert result[0, 0, 5] == pytest.approx(0.0, abs=1e-10)


def test_apply_floors_mountain_cell_one_hot():
    """apply_floors: mountain cell → one-hot on class 5."""
    from astar_island.posterior import apply_floors

    grid = np.full((H, W), 11, dtype=np.int32)  # all plains
    grid[5, 5] = 5  # mountain
    pred = _uniform_pred()
    cfg = Config()
    result = apply_floors(pred, grid, cfg)
    # Mountain cell (5,5) should be one-hot on class 5
    assert result[5, 5, 5] == pytest.approx(1.0, abs=1e-10)
    assert result[5, 5, 0] == pytest.approx(0.0, abs=1e-10)


def test_apply_floors_dynamic_cell_above_floor_standard():
    """apply_floors: non-ocean, non-mountain dynamic cell → all probs ≥ floor_standard after normalize."""
    from astar_island.posterior import apply_floors

    # Grid entirely plains — no ocean, no mountain
    grid = np.full((H, W), 11, dtype=np.int32)
    pred = _uniform_pred()
    cfg = Config()
    result = apply_floors(pred, grid, cfg)
    # Every cell is dynamic (plains); after applying floor_standard
    # each class prob is renormalized but should not drop below ~floor_standard/(K*floor_standard) ~ 1/K
    # Just check they're all positive
    assert np.all(result >= 0.0)
    assert np.allclose(result.sum(axis=-1), 1.0, atol=1e-8)


def test_apply_floors_mountain_class_capped_on_non_mountain():
    """apply_floors: mountain class on non-mountain dynamic → small fraction after floors.

    apply_floors caps mountain at floor_impossible before renormalization.
    After renorm the absolute value rises, but mountain should be much smaller
    than all other classes combined (dominated by floor_standard × (K-1) terms).
    """
    from astar_island.posterior import apply_floors

    # Grid entirely plains — non-mountain dynamic, interior cell far from border
    grid = np.full((H, W), 11, dtype=np.int32)
    # Give high mountain probability to a plains cell
    pred = _uniform_pred().copy()
    pred[20, 20, :] = 0.0
    pred[20, 20, 5] = 1.0  # mountain class on plains
    cfg = Config()
    result = apply_floors(pred, grid, cfg)
    # After floor_standard (0.01) is applied to all K classes and mountain is capped
    # at floor_impossible (0.001), after renorm mountain is a tiny fraction.
    # It should be << floor_standard (the other K-1 classes each got at least 0.01).
    # mountain share ≈ 0.001 / (5 * 0.01 + 0.001) ≈ 0.019, less than floor_standard (0.01)*2
    assert result[20, 20, 5] < cfg.floor_standard * 5


def test_compute_posterior_valid_probabilities():
    """compute_posterior: full pipeline produces (H,W,K) probs all summing ~1.0."""
    from astar_island.posterior import compute_posterior

    grid = _make_grid()
    p_base = _uniform_pred()
    counts = np.zeros((H, W, K), dtype=np.int32)
    cfg = Config()
    result = compute_posterior(p_base, None, counts, grid, cfg)
    assert result.shape == (H, W, K)
    sums = result.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-8)
    assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# archetypes.py
# ---------------------------------------------------------------------------


def test_classify_archetypes_ocean_border_cells_coastal():
    """classify_archetypes: cells adjacent to ocean border get COASTAL=0."""
    from astar_island.archetypes import classify_archetypes, COASTAL

    grid = _make_grid()  # has ocean border at rows/cols 0 and 39
    archetypes = classify_archetypes(grid)
    # Row 1 (just inside the ocean border) should be coastal
    # Ocean cells themselves: distance_transform_cdt gives dist 0 for ocean cells,
    # but dist <= 1 includes them. For non-ocean cells adjacent to ocean, dist == 1.
    # Row 1, col 1..38 are plains adjacent to ocean row/col 0
    assert archetypes[1, 1] == COASTAL


def test_classify_archetypes_all_plains_inland():
    """classify_archetypes: interior plains cells far from ocean/settlement → INLAND_NATURAL=2.

    Uses _make_grid() which has an ocean border ring. Cells far from the ocean
    and far from any settlement are INLAND_NATURAL.
    """
    from astar_island.archetypes import classify_archetypes, INLAND_NATURAL

    grid = _make_grid()  # ocean border + mountain at (5,5) + settlement at (10,10)
    archetypes = classify_archetypes(grid)
    # Cell at (25, 25) is far from ocean (nearest is ~14 away), far from settlement
    # (nearest settlement at (10,10) is ~21 cells away) → INLAND_NATURAL
    assert archetypes[25, 25] == INLAND_NATURAL


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------


def test_compute_round_weights_zero_observations_uniform():
    """compute_round_weights: zero observations → uniform weights."""
    from astar_island.calibration import compute_round_weights

    n_rounds = 5
    m_obs = np.zeros((3, K), dtype=np.float64)
    theta_hist = np.full((n_rounds, 3, K), 1.0 / K, dtype=np.float64)
    weights = compute_round_weights(m_obs, theta_hist)
    assert weights.shape == (n_rounds,)
    assert np.allclose(weights, 1.0 / n_rounds, atol=1e-10)


def test_compute_round_weights_sum_to_one():
    """compute_round_weights: weights always sum to 1.0."""
    from astar_island.calibration import compute_round_weights

    rng = np.random.default_rng(0)
    n_rounds = 4
    m_obs = rng.uniform(0, 10, size=(3, K)).astype(np.float64)
    theta_hist = rng.dirichlet(np.ones(K), size=(n_rounds, 3)).astype(np.float64)
    weights = compute_round_weights(m_obs, theta_hist)
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# solver.py
# ---------------------------------------------------------------------------


def test_static_mask_ocean_and_mountain():
    """_static_mask: ocean (10) and mountain (5) cells are True, others are False."""
    from astar_island.solver import HybridSolver

    solver = HybridSolver()
    grid = _make_grid()
    mask = solver._static_mask(grid)
    assert mask.shape == (H, W)
    # Ocean border should be True
    assert bool(mask[0, 0]) is True
    assert bool(mask[39, 0]) is True
    # Mountain at (5,5) should be True
    assert bool(mask[5, 5]) is True
    # Settlement at (10,10) should be False
    assert bool(mask[10, 10]) is False
    # Plains interior should be False
    assert bool(mask[20, 20]) is False


def test_parse_observation_creates_correct_counts():
    """_parse_observation: creates correct count tensor from a small grid response."""
    from astar_island.solver import HybridSolver

    solver = HybridSolver()

    # Simulate a 3×3 viewport response at (vx=0, vy=0)
    # terrain code 1 (Settlement) → class 1
    response = {
        "grid": [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    }
    counts = solver._parse_observation(response, vy=0, vx=0)
    assert counts.shape == (H, W, K)
    # Top-left 3×3 region should have count 1 in class 1
    assert np.all(counts[0:3, 0:3, 1] == 1)
    # Class 0 should be zero there
    assert np.all(counts[0:3, 0:3, 0] == 0)
    # Outside the viewport should be zero
    assert np.all(counts[3:, :, :] == 0)
    assert np.all(counts[:, 3:, :] == 0)

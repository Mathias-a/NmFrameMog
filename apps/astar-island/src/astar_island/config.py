"""Hyperparameter configuration — single source of truth for all tuning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All tunable hyperparameters for the hybrid solver."""

    # --- Prior model ---
    prior_model: str = "lightgbm"  # "lightgbm" | "logreg"
    lgb_max_depth: int = 5
    lgb_num_leaves: int = 31
    lgb_min_data_in_leaf: int = 50
    lgb_n_estimators: int = 200
    lgb_learning_rate: float = 0.05
    temperature: float = 1.0  # calibration temperature (tuned offline)

    # --- Feature extraction ---
    feature_radii: tuple[int, ...] = (1, 2, 4)
    window_sizes: tuple[int, ...] = (5, 9)

    # --- ESS (Effective Sample Size for Dirichlet prior) ---
    c_base: float = 3.0  # base ESS
    ess_min: float = 1.0
    ess_max: float = 4.0
    ess_confidence_weight: float = 0.75  # boost for confident cells

    # --- Combination ---
    lambda_prior: float = 0.4  # geometric pooling weight for learned prior
    eps: float = 1e-4  # probability floor before log/division

    # --- Floor rules ---
    floor_impossible: float = 0.001  # mountain on non-mountain, port on non-coastal
    floor_standard: float = 0.01  # all other uncertain classes

    # --- Query allocation ---
    n_queries_total: int = 50
    phase1_queries: int = 10  # broad calibration
    phase2_queries: int = 30  # EIG-targeted refinement
    phase3_queries: int = 10  # exploitation
    seed_caps: tuple[int, ...] = (15, 12, 10, 8, 5)  # front-loaded per-seed

    # --- Calibration ---
    n_archetypes: int = 3  # coastal, settlement_adjacent, inland_natural
    calibration_checkpoints: tuple[int, ...] = (
        10,
        20,
        30,
    )  # recalibrate after N queries

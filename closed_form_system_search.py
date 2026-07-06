from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from tqdm import tqdm


# =============================================================================
# Search knobs
# =============================================================================

RANDOM_SEED = 7

# Try (1, 1), (2, 1), or (2, 2). The script is written for a scalar theta in
# all cases:
#   A(theta) = A0 + theta * DeltaA
#   B(theta) = B0 + theta * DeltaB
#   DeltaA  = DeltaB @ K_nom
STATE_DIM = 2
ACTION_DIM = 2

N_CANDIDATES = 350
TOP_K_TO_SAVE = 10

OUTPUT_ROOT = Path("experiments") / "closed_form_system_search"


# -----------------------------------------------------------------------------
# Candidate sampling ranges
# -----------------------------------------------------------------------------

# First version: keep the search interpretable. A0, Q, and R are diagonal.
# B0 and DeltaB are dense.
A_DIAG_RANGE = (0.85, 1.05)
B_ENTRY_RANGE = (-1.0, 1.0)
DELTA_B_ENTRY_RANGE = (-1.0, 1.0)
Q_DIAG_LOG_RANGE = (np.log(0.5), np.log(5.0))
R_DIAG_LOG_RANGE = (np.log(0.05), np.log(10.0))

# Optional fixed values. Set to None to sample.
FIXED_A_DIAG = None
FIXED_Q_DIAG = None


# -----------------------------------------------------------------------------
# Fixed rollout protocol
# -----------------------------------------------------------------------------

HORIZON = 512
WINDOW_LENGTH = 50
NOMINAL_WARMUP_STEPS = 49
PREDICTION_IGNORE_FIRST_STEPS = WINDOW_LENGTH
PROCESS_NOISE_STD = 0.05
INITIAL_STATE_LOW = -0.3
INITIAL_STATE_HIGH = 0.3

EVAL_THETA_POINTS = 13
EVAL_SEEDS = tuple(range(6))


# -----------------------------------------------------------------------------
# Theta-range selection
# -----------------------------------------------------------------------------

THETA_SCAN_MAX = 0.8
THETA_SCAN_POINTS = 161
THETA_MAX_CANDIDATES = np.linspace(0.08, 0.5, 15)

MIN_K_VARIATION = 0.05
MIN_NOMINAL_ORACLE_COST_RATIO = 1.05
THETA_RANGE_X0_MAGNITUDE = 0.2


# -----------------------------------------------------------------------------
# Beta inner loop
# -----------------------------------------------------------------------------

BETA_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
INTRINSIC_BONUS_CLIP_MARGIN = 1e-3


# -----------------------------------------------------------------------------
# Validity filters. These should remove numerical/pathological systems rather
# than over-optimize the search.
# -----------------------------------------------------------------------------

MIN_B_NORM = 0.05
MIN_DELTA_B_NORM = 0.05
MAX_DELTA_B_TO_B_NORM_RATIO = 10.0

MAX_CONTROLLABILITY_COND = 1e6
MAX_DARE_M_COND = 1e8
MAX_ORACLE_CLOSED_LOOP_RHO = 0.995

MAX_ROLLOUT_STATE_NORM = 1e4
MAX_ROLLOUT_ACTION_NORM = 1e4


# -----------------------------------------------------------------------------
# Search score weights
# -----------------------------------------------------------------------------

TAIL_ABS_THETA_FRACTION = 0.6
CENTER_ABS_THETA_FRACTION = 0.2

WEIGHT_TAIL_GAIN = 1.0
WEIGHT_MEAN_GAIN = 0.25
WEIGHT_CENTER_LOSS = 0.5
WEIGHT_FAILURE_RATE = 200.0
WEIGHT_CE_IDENTIFICATION_PENALTY = 10.0

# A controller that stays close to nominal should have roughly this average
# theta error on a symmetric sweep. For a dense uniform sweep, always predicting
# zero gives E[|theta|] = theta_max / 2. Penalizing much smaller CE error avoids
# selecting families where the certainty-equivalent baseline already identifies
# theta well by closed-loop bootstrapping.
MIN_CE_ERROR_FRACTION_OF_THETA_MAX = 0.45


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class Family:
    A0: np.ndarray
    B0: np.ndarray
    DeltaB: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    K_nom: np.ndarray
    DeltaA: np.ndarray


@dataclass
class ThetaRangeInfo:
    theta_max: float
    k_variation: float
    nominal_oracle_cost_ratio: float


@dataclass
class RolloutResult:
    physical_return: float
    theta_pred_post_warmup_mean: float
    theta_error_post_warmup_mean: float
    theta_std_post_warmup_mean: float
    failed: bool


@dataclass
class CandidateScore:
    candidate_id: int
    score: float
    best_beta: float
    best_beta_index: int
    best_beta_is_boundary: bool
    second_best_beta: float
    best_beta_neighbor_score_gap: float
    theta_max: float
    mean_gain: float
    tail_gain: float
    center_loss: float
    ce_mean_return: float
    ua_mean_return: float
    ce_tail_return: float
    ua_tail_return: float
    ce_center_return: float
    ua_center_return: float
    ce_theta_error_post_warmup: float
    ua_theta_error_post_warmup: float
    ce_identification_penalty: float
    failure_rate: float
    ce_failure_rate: float
    ua_failure_rate: float
    k_variation: float
    nominal_oracle_cost_ratio: float


# =============================================================================
# Linear algebra helpers
# =============================================================================


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float]:
    P = solve_discrete_are(A, B, Q, R)
    M = R + B.T @ P @ B
    K = np.linalg.solve(M, B.T @ P @ A)
    return K, float(np.linalg.cond(M))


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    blocks = []
    A_power = np.eye(A.shape[0])
    for _ in range(A.shape[0]):
        blocks.append(A_power @ B)
        A_power = A @ A_power
    return np.concatenate(blocks, axis=1)


def physical_stage_cost(x: np.ndarray, u: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    return float(x.T @ Q @ x + u.T @ R @ u)


def finite_mean_or_nan(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def system_at_theta(family: Family, theta: float) -> tuple[np.ndarray, np.ndarray]:
    A = family.A0 + theta * family.DeltaA
    B = family.B0 + theta * family.DeltaB
    return A, B


def lqr_action_for_theta(family: Family, theta_hat: float, x: np.ndarray) -> np.ndarray:
    A_hat, B_hat = system_at_theta(family, theta_hat)
    K_hat, _ = lqr_gain(A_hat, B_hat, family.Q, family.R)
    return -K_hat @ x


# =============================================================================
# Candidate generation and filtering
# =============================================================================


def sample_family(rng: np.random.Generator) -> Family:
    if FIXED_A_DIAG is None:
        A_diag = rng.uniform(A_DIAG_RANGE[0], A_DIAG_RANGE[1], size=STATE_DIM)
    else:
        A_diag = np.asarray(FIXED_A_DIAG, dtype=np.float64)
    A0 = np.diag(A_diag)

    B0 = rng.uniform(B_ENTRY_RANGE[0], B_ENTRY_RANGE[1], size=(STATE_DIM, ACTION_DIM))
    DeltaB = rng.uniform(
        DELTA_B_ENTRY_RANGE[0],
        DELTA_B_ENTRY_RANGE[1],
        size=(STATE_DIM, ACTION_DIM),
    )

    if FIXED_Q_DIAG is None:
        Q_diag = np.exp(rng.uniform(Q_DIAG_LOG_RANGE[0], Q_DIAG_LOG_RANGE[1], size=STATE_DIM))
    else:
        Q_diag = np.asarray(FIXED_Q_DIAG, dtype=np.float64)
    R_diag = np.exp(rng.uniform(R_DIAG_LOG_RANGE[0], R_DIAG_LOG_RANGE[1], size=ACTION_DIM))

    Q = np.diag(Q_diag)
    R = np.diag(R_diag)

    K_nom, _ = lqr_gain(A0, B0, Q, R)
    DeltaA = DeltaB @ K_nom
    return Family(A0=A0, B0=B0, DeltaB=DeltaB, Q=Q, R=R, K_nom=K_nom, DeltaA=DeltaA)


def passes_static_filters(family: Family) -> bool:
    b_norm = float(np.linalg.norm(family.B0))
    delta_b_norm = float(np.linalg.norm(family.DeltaB))
    if b_norm < MIN_B_NORM or delta_b_norm < MIN_DELTA_B_NORM:
        return False
    if delta_b_norm / b_norm > MAX_DELTA_B_TO_B_NORM_RATIO:
        return False

    ctrl = controllability_matrix(family.A0, family.B0)
    if np.linalg.matrix_rank(ctrl) < STATE_DIM:
        return False
    if float(np.linalg.cond(ctrl)) > MAX_CONTROLLABILITY_COND:
        return False

    _, dare_m_cond = lqr_gain(family.A0, family.B0, family.Q, family.R)
    return dare_m_cond <= MAX_DARE_M_COND


def passes_theta_filters(family: Family, theta: float) -> bool:
    A, B = system_at_theta(family, theta)
    ctrl = controllability_matrix(A, B)
    if np.linalg.matrix_rank(ctrl) < STATE_DIM:
        return False
    if float(np.linalg.cond(ctrl)) > MAX_CONTROLLABILITY_COND:
        return False

    K, dare_m_cond = lqr_gain(A, B, family.Q, family.R)
    if dare_m_cond > MAX_DARE_M_COND:
        return False

    rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    return rho < MAX_ORACLE_CLOSED_LOOP_RHO


# =============================================================================
# Theta range selection
# =============================================================================


def noiseless_cost_with_fixed_gain(
    family: Family,
    theta: float,
    K: np.ndarray,
    x0: np.ndarray,
    horizon: int,
) -> float:
    A, B = system_at_theta(family, theta)
    x = x0.copy()
    total = 0.0
    for _ in range(horizon):
        u = -K @ x
        total += physical_stage_cost(x, u, family.Q, family.R)
        x = A @ x + B @ u
        if np.linalg.norm(x) > MAX_ROLLOUT_STATE_NORM or np.linalg.norm(u) > MAX_ROLLOUT_ACTION_NORM:
            return np.inf
    return total


def theta_range_initial_states() -> list[np.ndarray]:
    states = [
        np.full((STATE_DIM,), THETA_RANGE_X0_MAGNITUDE, dtype=np.float64),
        np.full((STATE_DIM,), -THETA_RANGE_X0_MAGNITUDE, dtype=np.float64),
    ]
    for idx in range(STATE_DIM):
        x_pos = np.zeros((STATE_DIM,), dtype=np.float64)
        x_neg = np.zeros((STATE_DIM,), dtype=np.float64)
        x_pos[idx] = THETA_RANGE_X0_MAGNITUDE
        x_neg[idx] = -THETA_RANGE_X0_MAGNITUDE
        states.extend([x_pos, x_neg])
    return states


def choose_theta_range(family: Family) -> ThetaRangeInfo | None:
    theta_scan = np.linspace(-THETA_SCAN_MAX, THETA_SCAN_MAX, THETA_SCAN_POINTS)
    valid = np.asarray([passes_theta_filters(family, theta) for theta in theta_scan], dtype=bool)
    if not bool(valid[np.argmin(np.abs(theta_scan))]):
        return None

    best_info = None
    best_score = -np.inf
    x0s = theta_range_initial_states()

    for theta_max in THETA_MAX_CANDIDATES:
        mask = np.abs(theta_scan) <= theta_max
        if not bool(np.all(valid[mask])):
            continue

        interval_thetas = theta_scan[mask]
        k_diffs = []
        ratios = []
        for theta in interval_thetas:
            A, B = system_at_theta(family, float(theta))
            K_oracle, _ = lqr_gain(A, B, family.Q, family.R)
            k_diffs.append(float(np.linalg.norm(K_oracle - family.K_nom)))

            x0_ratios = []
            for x0 in x0s:
                oracle_cost = noiseless_cost_with_fixed_gain(
                    family,
                    float(theta),
                    K_oracle,
                    x0,
                    horizon=min(HORIZON, 200),
                )
                nominal_cost = noiseless_cost_with_fixed_gain(
                    family,
                    float(theta),
                    family.K_nom,
                    x0,
                    horizon=min(HORIZON, 200),
                )
                if not np.isfinite(oracle_cost) or not np.isfinite(nominal_cost):
                    x0_ratios = [np.inf]
                    break
                ratio = float(nominal_cost / max(oracle_cost, 1e-12))
                if np.isfinite(ratio):
                    x0_ratios.append(ratio)
            if not x0_ratios:
                ratios.append(np.inf)
            else:
                ratios.append(float(np.mean(x0_ratios)))

        k_variation = float(np.mean(k_diffs))
        nominal_oracle_cost_ratio = float(np.mean(ratios))
        if not np.isfinite(nominal_oracle_cost_ratio):
            continue
        if k_variation < MIN_K_VARIATION:
            continue
        if nominal_oracle_cost_ratio < MIN_NOMINAL_ORACLE_COST_RATIO:
            continue

        # Prefer intervals where K varies and the oracle actually matters, but
        # avoid selecting huge intervals only because they contain unstable edges.
        range_score = k_variation * np.log(nominal_oracle_cost_ratio)
        if range_score > best_score:
            best_score = range_score
            best_info = ThetaRangeInfo(
                theta_max=float(theta_max),
                k_variation=k_variation,
                nominal_oracle_cost_ratio=nominal_oracle_cost_ratio,
            )

    return best_info


# =============================================================================
# Windowed GLS estimator
# =============================================================================


def regression_terms_from_window(
    family: Family,
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    residuals = []
    for x, u, x_next in window:
        g = family.DeltaA @ x + family.DeltaB @ u
        r = x_next - family.A0 @ x - family.B0 @ u
        features.append(g)
        residuals.append(r)
    return np.asarray(features, dtype=np.float64), np.asarray(residuals, dtype=np.float64)


def gls_theta_posterior(
    family: Family,
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior_mean: float,
    prior_var: float,
) -> tuple[float, float]:
    if len(window) < NOMINAL_WARMUP_STEPS:
        return prior_mean, prior_var

    g, r = regression_terms_from_window(family, window)
    noise_var = PROCESS_NOISE_STD**2

    prior_precision = 1.0 / prior_var
    data_precision = float(np.sum(g * g) / noise_var)
    data_linear_term = float(np.sum(g * r) / noise_var)

    posterior_precision = prior_precision + data_precision
    posterior_var = 1.0 / posterior_precision
    posterior_mean = posterior_var * (prior_precision * prior_mean + data_linear_term)
    return float(posterior_mean), float(posterior_var)


# =============================================================================
# Controllers
# =============================================================================


def certainty_equivalent_gls_action(
    family: Family,
    x: np.ndarray,
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior_mean: float,
    prior_var: float,
    theta_clip: float,
) -> tuple[np.ndarray, float, float]:
    theta_hat, theta_var = gls_theta_posterior(family, window, prior_mean, prior_var)
    theta_hat = float(np.clip(theta_hat, -theta_clip, theta_clip))
    u = lqr_action_for_theta(family, theta_hat, x)
    return u, theta_hat, float(np.sqrt(theta_var))


def spectral_clip_psd(matrix: np.ndarray, max_eigenvalue: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(symmetrize(matrix))
    eigvals = np.clip(eigvals, 0.0, max_eigenvalue)
    return symmetrize(eigvecs @ np.diag(eigvals) @ eigvecs.T)


def clip_bonus_relative_to_cost(
    bonus_raw: np.ndarray,
    base_cost: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Clip bonus so base_cost - bonus remains positive definite.

    The clipping is performed in normalized coordinates:
        H = M0^{-1/2} bonus M0^{-1/2}.
    Enforcing eig(H) <= 1 - margin gives bonus <= (1-margin) M0,
    hence M0 - bonus >= margin * M0.
    """

    cost_eigvals, cost_eigvecs = np.linalg.eigh(symmetrize(base_cost))
    cost_sqrt = cost_eigvecs @ np.diag(np.sqrt(cost_eigvals)) @ cost_eigvecs.T
    cost_inv_sqrt = cost_eigvecs @ np.diag(1.0 / np.sqrt(cost_eigvals)) @ cost_eigvecs.T

    normalized_bonus = cost_inv_sqrt @ symmetrize(bonus_raw) @ cost_inv_sqrt
    clipped_normalized_bonus = spectral_clip_psd(normalized_bonus, max_eigenvalue=1.0 - margin)
    return symmetrize(cost_sqrt @ clipped_normalized_bonus @ cost_sqrt)


def uncertainty_aware_ir_lqr_action(
    family: Family,
    x: np.ndarray,
    theta_mean: float,
    theta_var: float,
    beta: float,
) -> np.ndarray:
    """IR-LQR specialized to the scalar-theta family.

    The theta-sensitive dynamics term is
        theta * (DeltaA x + DeltaB u) = theta * D [x; u].
    GLS gives theta_var, so model uncertainty in z=[x;u] directions is
        theta_var * z.T D.T Sigma_w^{-1} D z.
    We subtract beta times this matrix from block_diag(Q, R), clip the
    subtraction so the modified cost remains positive definite, and solve
    the resulting generalized LQR problem.
    """

    A_hat, B_hat = system_at_theta(family, theta_mean)
    theta_sensitivity = np.concatenate([family.DeltaA, family.DeltaB], axis=1)
    noise_precision = np.eye(STATE_DIM, dtype=np.float64) / (PROCESS_NOISE_STD**2)

    sigma_z = float(theta_var) * (theta_sensitivity.T @ noise_precision @ theta_sensitivity)
    base_cost = block_diag(family.Q, family.R)
    bonus = clip_bonus_relative_to_cost(
        beta * sigma_z,
        base_cost,
        margin=INTRINSIC_BONUS_CLIP_MARGIN,
    )
    modified_cost = symmetrize(base_cost - bonus)

    n = STATE_DIM
    q_tilde = modified_cost[:n, :n]
    n_tilde = modified_cost[:n, n:]
    r_tilde = modified_cost[n:, n:]

    P = solve_discrete_are(A_hat, B_hat, q_tilde, r_tilde, s=n_tilde)
    K = np.linalg.solve(r_tilde + B_hat.T @ P @ B_hat, B_hat.T @ P @ A_hat + n_tilde.T)
    return -K @ x


def uncertainty_aware_gls_action(
    family: Family,
    x: np.ndarray,
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior_mean: float,
    prior_var: float,
    beta: float,
    theta_clip: float,
) -> tuple[np.ndarray, float, float]:
    theta_mean, theta_var = gls_theta_posterior(family, window, prior_mean, prior_var)
    theta_mean = float(np.clip(theta_mean, -theta_clip, theta_clip))
    u = uncertainty_aware_ir_lqr_action(family, x, theta_mean, theta_var, beta)
    return u, theta_mean, float(np.sqrt(theta_var))


# =============================================================================
# Rollout and evaluation
# =============================================================================


def rollout(
    family: Family,
    theta: float,
    seed: int,
    controller: str,
    beta: float,
    prior_var: float,
    theta_clip: float,
) -> RolloutResult:
    rng = np.random.default_rng(seed)
    A_true, B_true = system_at_theta(family, theta)

    x = rng.uniform(INITIAL_STATE_LOW, INITIAL_STATE_HIGH, size=STATE_DIM)
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    total_cost = 0.0
    theta_preds = []
    theta_errors = []
    theta_stds = []

    failed = False
    for step in range(HORIZON):
        try:
            if controller == "ce_gls":
                u, theta_est, theta_std = certainty_equivalent_gls_action(
                    family,
                    x,
                    window,
                    prior_mean=0.0,
                    prior_var=prior_var,
                    theta_clip=theta_clip,
                )
            else:
                u, theta_est, theta_std = uncertainty_aware_gls_action(
                    family,
                    x,
                    window,
                    prior_mean=0.0,
                    prior_var=prior_var,
                    beta=beta,
                    theta_clip=theta_clip,
                )
        except (np.linalg.LinAlgError, ValueError):
            failed = True
            break

        if np.linalg.norm(u) > MAX_ROLLOUT_ACTION_NORM or np.linalg.norm(x) > MAX_ROLLOUT_STATE_NORM:
            failed = True
            break

        cost = physical_stage_cost(x, u, family.Q, family.R)
        total_cost += cost

        noise = rng.normal(0.0, PROCESS_NOISE_STD, size=STATE_DIM)
        x_next = A_true @ x + B_true @ u + noise

        window.append((x.copy(), u.copy(), x_next.copy()))
        if len(window) > WINDOW_LENGTH:
            window.pop(0)

        if step >= PREDICTION_IGNORE_FIRST_STEPS:
            theta_preds.append(theta_est)
            theta_errors.append(abs(theta_est - theta))
            if np.isfinite(theta_std):
                theta_stds.append(theta_std)

        x = x_next

    if failed:
        total_cost += 1e6

    theta_pred_post_warmup_mean = float(np.mean(theta_preds)) if theta_preds else float("nan")
    theta_error_post_warmup_mean = float(np.mean(theta_errors)) if theta_errors else float("nan")
    theta_std_post_warmup_mean = float(np.mean(theta_stds)) if theta_stds else float("nan")
    return RolloutResult(
        physical_return=-float(total_cost),
        theta_pred_post_warmup_mean=theta_pred_post_warmup_mean,
        theta_error_post_warmup_mean=theta_error_post_warmup_mean,
        theta_std_post_warmup_mean=theta_std_post_warmup_mean,
        failed=failed,
    )


def summarize_results(rows: list[dict], theta_max: float) -> dict:
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    errors = np.asarray([row["theta_error_post_warmup"] for row in rows], dtype=np.float64)
    failures = np.asarray([row["failed"] for row in rows], dtype=np.float64)
    abs_thetas = np.asarray([abs(row["theta"]) for row in rows], dtype=np.float64)

    tail_mask = abs_thetas >= TAIL_ABS_THETA_FRACTION * theta_max
    center_mask = abs_thetas <= CENTER_ABS_THETA_FRACTION * theta_max

    return {
        "mean_return": float(np.mean(returns)),
        "tail_return": float(np.mean(returns[tail_mask])),
        "center_return": float(np.mean(returns[center_mask])),
        "theta_error_post_warmup": finite_mean_or_nan(errors),
        "failure_rate": float(np.mean(failures)),
    }


def evaluate_controller(
    family: Family,
    theta_max: float,
    controller: str,
    beta: float,
) -> tuple[dict, list[dict]]:
    theta_grid = np.linspace(-theta_max, theta_max, EVAL_THETA_POINTS)
    prior_var = theta_max**2
    rows = []

    for theta in theta_grid:
        for seed in EVAL_SEEDS:
            result = rollout(
                family=family,
                theta=float(theta),
                seed=int(seed),
                controller=controller,
                beta=beta,
                prior_var=prior_var,
                theta_clip=theta_max,
            )
            rows.append(
                {
                    "theta": float(theta),
                    "seed": int(seed),
                    "controller": controller,
                    "beta": float(beta),
                    "return": result.physical_return,
                    "theta_pred_post_warmup": result.theta_pred_post_warmup_mean,
                    "theta_error_post_warmup": result.theta_error_post_warmup_mean,
                    "theta_std_post_warmup": result.theta_std_post_warmup_mean,
                    "failed": bool(result.failed),
                }
            )

    return summarize_results(rows, theta_max), rows


def candidate_score(
    candidate_id: int,
    theta_info: ThetaRangeInfo,
    ce_summary: dict,
    ua_summary: dict,
    beta: float,
    beta_index: int,
    beta_count: int,
) -> CandidateScore:
    mean_gain = ua_summary["mean_return"] - ce_summary["mean_return"]
    tail_gain = ua_summary["tail_return"] - ce_summary["tail_return"]
    center_loss = max(0.0, ce_summary["center_return"] - ua_summary["center_return"])
    ce_failure_rate = ce_summary["failure_rate"]
    ua_failure_rate = ua_summary["failure_rate"]
    failure_rate = max(ce_failure_rate, ua_failure_rate)
    min_ce_error = MIN_CE_ERROR_FRACTION_OF_THETA_MAX * theta_info.theta_max
    ce_identification_penalty = max(0.0, min_ce_error - ce_summary["theta_error_post_warmup"])

    score = (
        WEIGHT_TAIL_GAIN * tail_gain
        + WEIGHT_MEAN_GAIN * mean_gain
        - WEIGHT_CENTER_LOSS * center_loss
        - WEIGHT_FAILURE_RATE * failure_rate
        - WEIGHT_CE_IDENTIFICATION_PENALTY * ce_identification_penalty
    )

    return CandidateScore(
        candidate_id=candidate_id,
        score=float(score),
        best_beta=float(beta),
        best_beta_index=int(beta_index),
        best_beta_is_boundary=bool(beta_index == 0 or beta_index == beta_count - 1),
        second_best_beta=float("nan"),
        best_beta_neighbor_score_gap=float("nan"),
        theta_max=theta_info.theta_max,
        mean_gain=float(mean_gain),
        tail_gain=float(tail_gain),
        center_loss=float(center_loss),
        ce_mean_return=ce_summary["mean_return"],
        ua_mean_return=ua_summary["mean_return"],
        ce_tail_return=ce_summary["tail_return"],
        ua_tail_return=ua_summary["tail_return"],
        ce_center_return=ce_summary["center_return"],
        ua_center_return=ua_summary["center_return"],
        ce_theta_error_post_warmup=ce_summary["theta_error_post_warmup"],
        ua_theta_error_post_warmup=ua_summary["theta_error_post_warmup"],
        ce_identification_penalty=float(ce_identification_penalty),
        failure_rate=float(failure_rate),
        ce_failure_rate=float(ce_failure_rate),
        ua_failure_rate=float(ua_failure_rate),
        k_variation=theta_info.k_variation,
        nominal_oracle_cost_ratio=theta_info.nominal_oracle_cost_ratio,
    )


def add_beta_grid_diagnostics(scores: list[CandidateScore]) -> None:
    scores_sorted = sorted(scores, key=lambda item: item.score, reverse=True)
    best = scores_sorted[0]

    if len(scores_sorted) >= 2:
        best.second_best_beta = float(scores_sorted[1].best_beta)

    neighbor_scores = [
        score.score
        for score in scores
        if abs(score.best_beta_index - best.best_beta_index) == 1
    ]
    if neighbor_scores:
        best.best_beta_neighbor_score_gap = float(best.score - max(neighbor_scores))


# =============================================================================
# Output helpers
# =============================================================================


def family_to_jsonable(family: Family) -> dict:
    return {
        "A0": family.A0.tolist(),
        "B0": family.B0.tolist(),
        "DeltaB": family.DeltaB.tolist(),
        "DeltaA": family.DeltaA.tolist(),
        "Q": family.Q.tolist(),
        "R": family.R.tolist(),
        "K_nom": family.K_nom.tolist(),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_rollout_sweeps(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return

    controllers = sorted({str(row["controller"]) for row in rows})
    theta_values = sorted({float(row["theta"]) for row in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for controller in controllers:
        means = []
        stds = []
        for theta in theta_values:
            vals = np.asarray(
                [
                    float(row["return"])
                    for row in rows
                    if str(row["controller"]) == controller and float(row["theta"]) == theta
                ],
                dtype=np.float64,
            )
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        x = np.asarray(theta_values, dtype=np.float64)
        y = np.asarray(means, dtype=np.float64)
        y_std = np.asarray(stds, dtype=np.float64)
        ax.plot(x, y, linewidth=2, label=controller)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.15)
    ax.set_title("Return vs Theta")
    ax.set_xlabel("theta")
    ax.set_ylabel("episode return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "return_vs_theta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_all = np.asarray(theta_values, dtype=np.float64)
    ax.plot([float(x_all.min()), float(x_all.max())], [float(x_all.min()), float(x_all.max())], "k--", linewidth=1.5, label="ideal")
    for controller in controllers:
        means = []
        stds = []
        for theta in theta_values:
            vals = np.asarray(
                [
                    float(row["theta_pred_post_warmup"])
                    for row in rows
                    if str(row["controller"]) == controller
                    and float(row["theta"]) == theta
                    and np.isfinite(float(row["theta_pred_post_warmup"]))
                ],
                dtype=np.float64,
            )
            means.append(float(np.mean(vals)) if vals.size else float("nan"))
            stds.append(float(np.std(vals)) if vals.size else float("nan"))
        y = np.asarray(means, dtype=np.float64)
        y_std = np.asarray(stds, dtype=np.float64)
        ok = np.isfinite(y)
        if not bool(ok.any()):
            continue
        ax.plot(x_all[ok], y[ok], linewidth=2, label=controller)
        ax.fill_between(x_all[ok], y[ok] - y_std[ok], y[ok] + y_std[ok], alpha=0.15)
    ax.set_title("Theta Prediction vs Theta")
    ax.set_xlabel("true theta")
    ax.set_ylabel("post-warmup mean predicted theta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "theta_prediction_vs_theta.png", dpi=180)
    plt.close(fig)


def save_candidate(
    output_dir: Path,
    family: Family,
    score: CandidateScore,
    rollout_rows: list[dict],
) -> None:
    cand_dir = output_dir / f"candidate_{score.candidate_id:05d}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    with open(cand_dir / "system.json", "w") as f:
        json.dump(family_to_jsonable(family), f, indent=2)
    with open(cand_dir / "score.json", "w") as f:
        json.dump(asdict(score), f, indent=2)
    write_csv(cand_dir / "rollouts.csv", rollout_rows)
    plot_rollout_sweeps(cand_dir, rollout_rows)


# =============================================================================
# Main search loop
# =============================================================================


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    timestamp = datetime.now().strftime("%m-%d__%H-%M")
    output_dir = OUTPUT_ROOT / f"system_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[tuple[CandidateScore, Family, list[dict]]] = []
    summary_rows = []
    counters = {
        "sampled": 0,
        "rejected_static_filters": 0,
        "rejected_theta_range": 0,
        "rejected_numerical": 0,
        "evaluated": 0,
        "accepted_top_k": 0,
    }
    rollouts_per_evaluated_candidate = EVAL_THETA_POINTS * len(EVAL_SEEDS) * (1 + len(BETA_GRID))

    progress = tqdm(
        range(N_CANDIDATES),
        desc=f"closed-form search (~{rollouts_per_evaluated_candidate} rollouts/eval)",
        unit="candidate",
    )
    for candidate_id in progress:
        counters["sampled"] += 1
        try:
            family = sample_family(rng)
            if not passes_static_filters(family):
                counters["rejected_static_filters"] += 1
                continue

            theta_info = choose_theta_range(family)
            if theta_info is None:
                counters["rejected_theta_range"] += 1
                continue

            ce_summary, ce_rows = evaluate_controller(
                family,
                theta_max=theta_info.theta_max,
                controller="ce_gls",
                beta=0.0,
            )

            best_score = None
            best_ua_rows = None
            beta_scores = []
            for beta_index, beta in enumerate(BETA_GRID):
                ua_summary, ua_rows = evaluate_controller(
                    family,
                    theta_max=theta_info.theta_max,
                    controller="ua_gls",
                    beta=float(beta),
                )
                score = candidate_score(
                    candidate_id=candidate_id,
                    theta_info=theta_info,
                    ce_summary=ce_summary,
                    ua_summary=ua_summary,
                    beta=float(beta),
                    beta_index=int(beta_index),
                    beta_count=len(BETA_GRID),
                )
                beta_scores.append(score)
                if best_score is None or score.score > best_score.score:
                    best_score = score
                    best_ua_rows = ua_rows

            if best_score is None or best_ua_rows is None:
                raise RuntimeError("BETA_GRID is empty; cannot score uncertainty-aware controller.")
            add_beta_grid_diagnostics(beta_scores)

            rollout_rows = ce_rows + best_ua_rows
            accepted.append((best_score, family, rollout_rows))
            accepted.sort(key=lambda item: item[0].score, reverse=True)
            accepted = accepted[:TOP_K_TO_SAVE]

            counters["evaluated"] += 1
            counters["accepted_top_k"] = len(accepted)
            summary_rows.append(asdict(best_score))
            tqdm.write(
                f"candidate={candidate_id:05d} "
                f"score={best_score.score:9.3f} "
                f"tail_gain={best_score.tail_gain:9.3f} "
                f"mean_gain={best_score.mean_gain:9.3f} "
                f"beta={best_score.best_beta:g} "
                f"theta_max={best_score.theta_max:.3f}"
            )

        except (np.linalg.LinAlgError, ValueError):
            counters["rejected_numerical"] += 1
            continue
        finally:
            progress.set_postfix(
                evaluated=counters["evaluated"],
                top=len(accepted),
                static_rej=counters["rejected_static_filters"],
                theta_rej=counters["rejected_theta_range"],
                numeric_rej=counters["rejected_numerical"],
            )

    summary_rows.sort(key=lambda row: row["score"], reverse=True)
    write_csv(output_dir / "summary.csv", summary_rows)

    counters["accepted_top_k"] = len(accepted)
    with open(output_dir / "search_diagnostics.json", "w") as f:
        json.dump(counters, f, indent=2)

    with open(output_dir / "search_config.json", "w") as f:
        json.dump(
            {
                "STATE_DIM": STATE_DIM,
                "ACTION_DIM": ACTION_DIM,
                "N_CANDIDATES": N_CANDIDATES,
                "BETA_GRID": list(BETA_GRID),
                "INTRINSIC_BONUS_CLIP_MARGIN": INTRINSIC_BONUS_CLIP_MARGIN,
                "HORIZON": HORIZON,
                "WINDOW_LENGTH": WINDOW_LENGTH,
                "PREDICTION_IGNORE_FIRST_STEPS": PREDICTION_IGNORE_FIRST_STEPS,
                "PROCESS_NOISE_STD": PROCESS_NOISE_STD,
                "EVAL_THETA_POINTS": EVAL_THETA_POINTS,
                "EVAL_SEEDS": list(EVAL_SEEDS),
                "THETA_RANGE_X0_MAGNITUDE": THETA_RANGE_X0_MAGNITUDE,
                "ROLLOUTS_PER_EVALUATED_CANDIDATE": rollouts_per_evaluated_candidate,
            },
            f,
            indent=2,
        )

    for score, family, rollout_rows in accepted:
        save_candidate(output_dir, family, score, rollout_rows)

    print(f"\nSaved search results to: {output_dir}")


if __name__ == "__main__":
    main()

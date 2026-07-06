from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

import closed_form_system_search as search
import reevaluate_closed_form_candidates as reeval


OUTPUT_ROOT = Path("experiments") / "closed_form_old_two_action_blr_ua_lqr"

THETA_MAX = 0.25
THETA_POINTS = int(os.environ.get("BLR_UA_THETA_POINTS", "41"))


def _parse_float_tuple(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


SEED_COUNT = int(os.environ.get("BLR_UA_SEED_COUNT", "20"))
SEED_START = int(os.environ.get("BLR_UA_SEED_START", "12345"))
SEEDS = tuple(range(SEED_START, SEED_START + SEED_COUNT))
BETA_GRID = _parse_float_tuple(
    os.environ.get("BLR_UA_BETA_GRID", ""),
    (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
)
CACHE_THETA_DECIMALS = int(os.environ.get("BLR_UA_CACHE_THETA_DECIMALS", "4"))
CACHE_VAR_DECIMALS = int(os.environ.get("BLR_UA_CACHE_VAR_DECIMALS", "6"))
ID_UPDATE_INTERVAL = int(os.environ.get("BLR_UA_ID_UPDATE_INTERVAL", "10"))
if ID_UPDATE_INTERVAL <= 0:
    raise ValueError("BLR_UA_ID_UPDATE_INTERVAL must be positive.")
ACTION_CLIP = os.environ.get("BLR_UA_ACTION_CLIP", "1").lower() not in {"0", "false", "no"}
ACTION_LOW = float(os.environ.get("BLR_UA_ACTION_LOW", "-30.0"))
ACTION_HIGH = float(os.environ.get("BLR_UA_ACTION_HIGH", "30.0"))


OLD_TWO_ACTION_SYSTEM = {
    "name": "old_two_action",
    "theta_max": THETA_MAX,
    "A0": [[0.98, 0.0], [0.0, 0.95]],
    "B0": [[0.1, 0.03], [1.0, 0.2]],
    "DeltaB": [[0.5, 0.02], [0.15, 1.0]],
    "Q": [[2.0, 0.0], [0.0, 1.0]],
    "R": [[5.0, 0.0], [0.0, 0.2]],
}

_ACTIVE_FAMILY: search.Family | None = None


def _family() -> search.Family:
    if _ACTIVE_FAMILY is None:
        raise RuntimeError("No active family set.")
    return _ACTIVE_FAMILY


@lru_cache(maxsize=20_000)
def _cached_lqr_gain(theta_hat: float) -> np.ndarray:
    family = _family()
    A_hat, B_hat = search.system_at_theta(family, theta_hat)
    K, _ = search.lqr_gain(A_hat, B_hat, family.Q, family.R)
    return K


@lru_cache(maxsize=50_000)
def _cached_ua_gain(theta_mean: float, theta_var: float, beta: float) -> np.ndarray:
    family = _family()
    A_hat, B_hat = search.system_at_theta(family, theta_mean)
    theta_sensitivity = np.concatenate([family.DeltaA, family.DeltaB], axis=1)
    noise_precision = np.eye(search.STATE_DIM, dtype=np.float64) / (search.PROCESS_NOISE_STD**2)

    sigma_z = float(theta_var) * (theta_sensitivity.T @ noise_precision @ theta_sensitivity)
    base_cost = search.block_diag(family.Q, family.R)
    bonus = search.clip_bonus_relative_to_cost(
        beta * sigma_z,
        base_cost,
        margin=search.INTRINSIC_BONUS_CLIP_MARGIN,
    )
    modified_cost = search.symmetrize(base_cost - bonus)

    n = search.STATE_DIM
    q_tilde = modified_cost[:n, :n]
    n_tilde = modified_cost[:n, n:]
    r_tilde = modified_cost[n:, n:]

    P = solve_discrete_are(A_hat, B_hat, q_tilde, r_tilde, s=n_tilde)
    return np.linalg.solve(r_tilde + B_hat.T @ P @ B_hat, B_hat.T @ P @ A_hat + n_tilde.T)


def _cached_rollout(
    family: search.Family,
    theta: float,
    seed: int,
    controller: str,
    beta: float,
    prior_var: float,
    theta_clip: float,
) -> search.RolloutResult:
    rng = np.random.default_rng(seed)
    A_true, B_true = search.system_at_theta(family, theta)
    x = rng.uniform(search.INITIAL_STATE_LOW, search.INITIAL_STATE_HIGH, size=search.STATE_DIM)
    window: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    total_cost = 0.0
    theta_preds = []
    theta_errors = []
    theta_stds = []
    failed = False
    held_theta_mean = 0.0
    held_theta_var = prior_var
    uses_gls = controller in {"ce_gls", "ua_gls"}

    for step in range(search.HORIZON):
        try:
            if uses_gls and (step == 0 or step % ID_UPDATE_INTERVAL == 0):
                theta_mean, theta_var = search.gls_theta_posterior(
                    family,
                    window,
                    prior_mean=0.0,
                    prior_var=prior_var,
                )
                held_theta_mean = float(np.clip(theta_mean, -theta_clip, theta_clip))
                held_theta_var = float(theta_var)

            theta_mean = held_theta_mean
            theta_var = held_theta_var
            theta_std = float(np.sqrt(theta_var))

            if controller == "ce_gls":
                u = -_cached_lqr_gain(theta_mean) @ x
            elif controller == "ua_gls":
                beta_key = round(float(beta), 6)
                u = -_cached_ua_gain(theta_mean, float(theta_var), beta_key) @ x
            elif controller == "nominal_lqr":
                u = -family.K_nom @ x
            elif controller == "oracle_lqr":
                theta_key = round(float(theta), CACHE_THETA_DECIMALS)
                u = -_cached_lqr_gain(theta_key) @ x
            else:
                raise ValueError(f"Unknown controller: {controller}")
            if ACTION_CLIP:
                u = np.clip(u, ACTION_LOW, ACTION_HIGH)
        except (np.linalg.LinAlgError, ValueError):
            failed = True
            break

        if np.linalg.norm(u) > search.MAX_ROLLOUT_ACTION_NORM or np.linalg.norm(x) > search.MAX_ROLLOUT_STATE_NORM:
            failed = True
            break

        total_cost += search.physical_stage_cost(x, u, family.Q, family.R)
        noise = rng.normal(0.0, search.PROCESS_NOISE_STD, size=search.STATE_DIM)
        x_next = A_true @ x + B_true @ u + noise

        window.append((x.copy(), u.copy(), x_next.copy()))
        if len(window) > search.WINDOW_LENGTH:
            window.pop(0)

        if uses_gls and step >= search.PREDICTION_IGNORE_FIRST_STEPS:
            theta_preds.append(theta_mean)
            theta_errors.append(abs(theta_mean - theta))
            theta_stds.append(theta_std)

        x = x_next

    if failed:
        total_cost += 1e6

    return search.RolloutResult(
        physical_return=-float(total_cost),
        theta_pred_post_warmup_mean=float(np.mean(theta_preds)) if theta_preds else float("nan"),
        theta_error_post_warmup_mean=float(np.mean(theta_errors)) if theta_errors else float("nan"),
        theta_std_post_warmup_mean=float(np.mean(theta_stds)) if theta_stds else float("nan"),
        failed=failed,
    )


def _cached_evaluate_controller(
    family: search.Family,
    theta_max: float,
    controller: str,
    beta: float,
    label: str,
) -> tuple[dict, list[dict]]:
    theta_grid = np.linspace(-theta_max, theta_max, search.EVAL_THETA_POINTS)
    prior_var = theta_max**2
    rows = []
    total = len(theta_grid) * len(search.EVAL_SEEDS)
    with tqdm(total=total, desc=label, unit="rollout") as progress:
        for theta in theta_grid:
            for seed in search.EVAL_SEEDS:
                result = _cached_rollout(
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
                progress.update(1)
    return search.summarize_results(rows, theta_max), rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global _ACTIVE_FAMILY
    family = reeval.manual_family(OLD_TWO_ACTION_SYSTEM)
    _ACTIVE_FAMILY = family
    _cached_lqr_gain.cache_clear()
    _cached_ua_gain.cache_clear()
    reeval.set_search_dimensions(family)
    search.EVAL_THETA_POINTS = THETA_POINTS
    search.EVAL_SEEDS = SEEDS
    search.HORIZON = 512
    search.WINDOW_LENGTH = 50
    search.NOMINAL_WARMUP_STEPS = 49
    search.PREDICTION_IGNORE_FIRST_STEPS = 50
    search.PROCESS_NOISE_STD = 0.05
    search.INITIAL_STATE_LOW = -0.3
    search.INITIAL_STATE_HIGH = 0.3

    timestamp = datetime.now().strftime("%m-%d__%H-%M")
    out = OUTPUT_ROOT / f"old_two_action_blr_ua_lqr_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    _, ce_rows = _cached_evaluate_controller(
        family=family,
        theta_max=THETA_MAX,
        controller="ce_gls",
        beta=0.0,
        label="ce_gls",
    )
    rows.extend(ce_rows)

    for beta in BETA_GRID:
        _, beta_rows = _cached_evaluate_controller(
            family=family,
            theta_max=THETA_MAX,
            controller="ua_gls",
            beta=float(beta),
            label=f"ua_gls beta={beta:g}",
        )
        for row in beta_rows:
            row["controller"] = f"ua_gls_beta_{beta:g}"
        rows.extend(beta_rows)

    _, nominal_rows = _cached_evaluate_controller(
        family=family,
        theta_max=THETA_MAX,
        controller="nominal_lqr",
        beta=float("nan"),
        label="nominal_lqr",
    )
    rows.extend(nominal_rows)

    _, oracle_rows = _cached_evaluate_controller(
        family=family,
        theta_max=THETA_MAX,
        controller="oracle_lqr",
        beta=float("nan"),
        label="oracle_lqr",
    )
    rows.extend(oracle_rows)

    summary_rows = reeval.summarize_rows(rows, THETA_MAX)
    per_theta_rows = reeval.summarize_by_theta(rows)

    write_csv(out / "rollouts.csv", rows)
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "per_theta_summary.csv", per_theta_rows)
    with open(out / "system.json", "w") as f:
        json.dump(search.family_to_jsonable(family), f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(
            {
                "theta_max": THETA_MAX,
                "theta_points": THETA_POINTS,
                "seeds": list(SEEDS),
                "beta_grid": list(BETA_GRID),
                "horizon": search.HORIZON,
                "window_length": search.WINDOW_LENGTH,
                "nominal_warmup_steps": search.NOMINAL_WARMUP_STEPS,
                "prediction_ignore_first_steps": search.PREDICTION_IGNORE_FIRST_STEPS,
                "id_update_interval": ID_UPDATE_INTERVAL,
                "process_noise_std": search.PROCESS_NOISE_STD,
                "initial_state_low": search.INITIAL_STATE_LOW,
                "initial_state_high": search.INITIAL_STATE_HIGH,
                "cache_theta_decimals": None,
                "cache_var_decimals": None,
                "gain_rounding": False,
                "action_clip": ACTION_CLIP,
                "action_low": ACTION_LOW,
                "action_high": ACTION_HIGH,
            },
            f,
            indent=2,
        )

    search.plot_rollout_sweeps(out, rows)

    summary_by_controller = {row["controller"]: row for row in summary_rows}
    ua_rows = [row for row in summary_rows if row["controller"].startswith("ua_gls_beta_")]
    best_ua = max(ua_rows, key=lambda row: row["mean_return"])
    best_tail_ua = max(ua_rows, key=lambda row: row["tail_mean_return"])

    print(f"Saved BLR/UA-LQR old two-action evaluation to: {out}")
    for name in ["oracle_lqr", "nominal_lqr", "ce_gls"]:
        row = summary_by_controller[name]
        print(
            f"{name}: mean={row['mean_return']:.3f} tail={row['tail_mean_return']:.3f} "
            f"center={row['center_mean_return']:.3f} theta_err={row['theta_error_post_warmup']:.3f} "
            f"fail={row['failure_rate']:.3f}"
        )
    for label, row in [("best_ua_mean", best_ua), ("best_ua_tail", best_tail_ua)]:
        print(
            f"{label} ({row['controller']}): mean={row['mean_return']:.3f} "
            f"tail={row['tail_mean_return']:.3f} center={row['center_mean_return']:.3f} "
            f"theta_err={row['theta_error_post_warmup']:.3f} fail={row['failure_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

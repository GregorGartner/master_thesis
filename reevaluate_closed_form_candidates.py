from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import closed_form_system_search as search


# =============================================================================
# Re-evaluation knobs
# =============================================================================

SOURCE_RUN = Path("experiments") / "closed_form_system_search" / "system_search_05-23__18-50"
CANDIDATE_IDS = (12, 184, 258, 75)

# Optional hand-written systems. Leave empty to evaluate only search candidates.
# DeltaA is computed from DeltaB @ K_nom so the indistinguishability condition is
# preserved. Example:
#
# MANUAL_SYSTEMS = [
#     {
#         "name": "my_system",
#         "theta_max": 0.5,
#         "beta": 0.1,
#         "A0": [[0.98, 0.0], [0.0, 0.95]],
#         "B0": [[0.1, 0.03], [1.0, 0.2]],
#         "DeltaB": [[0.5, 0.02], [0.15, 1.0]],
#         "Q": [[2.0, 0.0], [0.0, 1.0]],
#         "R": [[5.0, 0.0], [0.0, 0.2]],
#     },
# ]
MANUAL_SYSTEMS = []

OUTPUT_ROOT = Path("experiments") / "closed_form_candidate_reeval"

THETA_POINTS = 41
SEEDS = tuple(range(30))

INCLUDE_REFERENCES = True


# =============================================================================
# Loading
# =============================================================================


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_family(path: Path) -> search.Family:
    data = load_json(path)
    return search.Family(
        A0=np.asarray(data["A0"], dtype=np.float64),
        B0=np.asarray(data["B0"], dtype=np.float64),
        DeltaB=np.asarray(data["DeltaB"], dtype=np.float64),
        Q=np.asarray(data["Q"], dtype=np.float64),
        R=np.asarray(data["R"], dtype=np.float64),
        K_nom=np.asarray(data["K_nom"], dtype=np.float64),
        DeltaA=np.asarray(data["DeltaA"], dtype=np.float64),
    )


def manual_family(spec: dict) -> search.Family:
    A0 = np.asarray(spec["A0"], dtype=np.float64)
    B0 = np.asarray(spec["B0"], dtype=np.float64)
    DeltaB = np.asarray(spec["DeltaB"], dtype=np.float64)
    Q = np.asarray(spec["Q"], dtype=np.float64)
    R = np.asarray(spec["R"], dtype=np.float64)
    K_nom, _ = search.lqr_gain(A0, B0, Q, R)
    DeltaA = DeltaB @ K_nom
    return search.Family(A0=A0, B0=B0, DeltaB=DeltaB, Q=Q, R=R, K_nom=K_nom, DeltaA=DeltaA)


def set_search_dimensions(family: search.Family) -> None:
    search.STATE_DIM = int(family.A0.shape[0])
    search.ACTION_DIM = int(family.B0.shape[1])


# =============================================================================
# Reference controllers
# =============================================================================


def reference_rollout(
    family: search.Family,
    theta: float,
    seed: int,
    controller: str,
) -> search.RolloutResult:
    rng = np.random.default_rng(seed)
    A_true, B_true = search.system_at_theta(family, theta)
    x = rng.uniform(search.INITIAL_STATE_LOW, search.INITIAL_STATE_HIGH, size=search.STATE_DIM)

    total_cost = 0.0
    failed = False
    for _ in range(search.HORIZON):
        try:
            if controller == "oracle_lqr":
                u = search.lqr_action_for_theta(family, theta, x)
            else:
                u = -family.K_nom @ x
        except (np.linalg.LinAlgError, ValueError):
            failed = True
            break

        if (
            np.linalg.norm(u) > search.MAX_ROLLOUT_ACTION_NORM
            or np.linalg.norm(x) > search.MAX_ROLLOUT_STATE_NORM
        ):
            failed = True
            break

        total_cost += search.physical_stage_cost(x, u, family.Q, family.R)
        noise = rng.normal(0.0, search.PROCESS_NOISE_STD, size=search.STATE_DIM)
        x = A_true @ x + B_true @ u + noise

    if failed:
        total_cost += 1e6

    return search.RolloutResult(
        physical_return=-float(total_cost),
        theta_pred_post_warmup_mean=float("nan"),
        theta_error_post_warmup_mean=float("nan"),
        theta_std_post_warmup_mean=float("nan"),
        failed=failed,
    )


def evaluate_reference_controller(
    family: search.Family,
    theta_max: float,
    controller: str,
) -> list[dict]:
    rows = []
    theta_grid = np.linspace(-theta_max, theta_max, THETA_POINTS)

    for theta in theta_grid:
        for seed in SEEDS:
            result = reference_rollout(
                family=family,
                theta=float(theta),
                seed=int(seed),
                controller=controller,
            )
            rows.append(
                {
                    "theta": float(theta),
                    "seed": int(seed),
                    "controller": controller,
                    "beta": float("nan"),
                    "return": result.physical_return,
                    "theta_pred_post_warmup": result.theta_pred_post_warmup_mean,
                    "theta_error_post_warmup": result.theta_error_post_warmup_mean,
                    "theta_std_post_warmup": result.theta_std_post_warmup_mean,
                    "failed": bool(result.failed),
                }
            )
    return rows


# =============================================================================
# Summaries
# =============================================================================


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_percentile(values: list[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, percentile))


def summarize_rows(rows: list[dict], theta_max: float) -> list[dict]:
    controllers = sorted({str(row["controller"]) for row in rows})
    summary_rows = []

    for controller in controllers:
        ctrl_rows = [row for row in rows if str(row["controller"]) == controller]
        tail_rows = [row for row in ctrl_rows if abs(float(row["theta"])) >= search.TAIL_ABS_THETA_FRACTION * theta_max]
        neg_tail_rows = [row for row in ctrl_rows if float(row["theta"]) <= -search.TAIL_ABS_THETA_FRACTION * theta_max]
        pos_tail_rows = [row for row in ctrl_rows if float(row["theta"]) >= search.TAIL_ABS_THETA_FRACTION * theta_max]
        center_rows = [row for row in ctrl_rows if abs(float(row["theta"])) <= search.CENTER_ABS_THETA_FRACTION * theta_max]

        returns = [float(row["return"]) for row in ctrl_rows]
        summary_rows.append(
            {
                "controller": controller,
                "beta": finite_mean([float(row["beta"]) for row in ctrl_rows]),
                "mean_return": finite_mean(returns),
                "median_return": finite_percentile(returns, 50),
                "q10_return": finite_percentile(returns, 10),
                "q90_return": finite_percentile(returns, 90),
                "tail_mean_return": finite_mean([float(row["return"]) for row in tail_rows]),
                "negative_tail_mean_return": finite_mean([float(row["return"]) for row in neg_tail_rows]),
                "positive_tail_mean_return": finite_mean([float(row["return"]) for row in pos_tail_rows]),
                "center_mean_return": finite_mean([float(row["return"]) for row in center_rows]),
                "theta_error_post_warmup": finite_mean(
                    [float(row["theta_error_post_warmup"]) for row in ctrl_rows]
                ),
                "theta_std_post_warmup": finite_mean([float(row["theta_std_post_warmup"]) for row in ctrl_rows]),
                "failure_rate": finite_mean([float(row["failed"]) for row in ctrl_rows]),
            }
        )
    return summary_rows


def summarize_by_theta(rows: list[dict]) -> list[dict]:
    controllers = sorted({str(row["controller"]) for row in rows})
    theta_values = sorted({float(row["theta"]) for row in rows})
    summary_rows = []

    for controller in controllers:
        for theta in theta_values:
            slice_rows = [
                row
                for row in rows
                if str(row["controller"]) == controller and float(row["theta"]) == theta
            ]
            returns = [float(row["return"]) for row in slice_rows]
            summary_rows.append(
                {
                    "controller": controller,
                    "theta": theta,
                    "mean_return": finite_mean(returns),
                    "median_return": finite_percentile(returns, 50),
                    "q10_return": finite_percentile(returns, 10),
                    "q90_return": finite_percentile(returns, 90),
                    "theta_pred_post_warmup": finite_mean(
                        [float(row["theta_pred_post_warmup"]) for row in slice_rows]
                    ),
                    "theta_error_post_warmup": finite_mean(
                        [float(row["theta_error_post_warmup"]) for row in slice_rows]
                    ),
                    "theta_std_post_warmup": finite_mean(
                        [float(row["theta_std_post_warmup"]) for row in slice_rows]
                    ),
                    "failure_rate": finite_mean([float(row["failed"]) for row in slice_rows]),
                }
            )
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reevaluate_one(
    output_dir: Path,
    label: str,
    family: search.Family,
    theta_max: float,
    beta: float,
    metadata: dict,
) -> dict:
    set_search_dimensions(family)

    _, ce_rows = search.evaluate_controller(
        family=family,
        theta_max=theta_max,
        controller="ce_gls",
        beta=0.0,
    )
    _, ua_rows = search.evaluate_controller(
        family=family,
        theta_max=theta_max,
        controller="ua_gls",
        beta=beta,
    )

    rows = ce_rows + ua_rows
    if INCLUDE_REFERENCES:
        rows += evaluate_reference_controller(family, theta_max, "nominal_lqr")
        rows += evaluate_reference_controller(family, theta_max, "oracle_lqr")

    out = output_dir / label
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "system.json", "w") as f:
        json.dump(search.family_to_jsonable(family), f, indent=2)
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    summary_rows = summarize_rows(rows, theta_max)
    per_theta_rows = summarize_by_theta(rows)

    write_csv(out / "rollouts.csv", rows)
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "per_theta_summary.csv", per_theta_rows)
    search.plot_rollout_sweeps(out, rows)

    by_controller = {row["controller"]: row for row in summary_rows}
    ce = by_controller["ce_gls"]
    ua = by_controller["ua_gls"]
    return {
        "label": label,
        "theta_max": theta_max,
        "beta": beta,
        "mean_gain": ua["mean_return"] - ce["mean_return"],
        "tail_gain": ua["tail_mean_return"] - ce["tail_mean_return"],
        "center_loss": max(0.0, ce["center_mean_return"] - ua["center_mean_return"]),
        "ce_mean_return": ce["mean_return"],
        "ua_mean_return": ua["mean_return"],
        "ce_tail_return": ce["tail_mean_return"],
        "ua_tail_return": ua["tail_mean_return"],
        "ce_center_return": ce["center_mean_return"],
        "ua_center_return": ua["center_mean_return"],
        "ce_theta_error_post_warmup": ce["theta_error_post_warmup"],
        "ua_theta_error_post_warmup": ua["theta_error_post_warmup"],
        "ce_failure_rate": ce["failure_rate"],
        "ua_failure_rate": ua["failure_rate"],
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    search.EVAL_THETA_POINTS = THETA_POINTS
    search.EVAL_SEEDS = SEEDS

    timestamp = datetime.now().strftime("%m-%d__%H-%M")
    output_dir = OUTPUT_ROOT / f"reeval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "source_run": str(SOURCE_RUN),
        "candidate_ids": list(CANDIDATE_IDS),
        "manual_system_names": [spec["name"] for spec in MANUAL_SYSTEMS],
        "theta_points": THETA_POINTS,
        "seeds": list(SEEDS),
        "include_references": INCLUDE_REFERENCES,
    }
    with open(output_dir / "reeval_config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_candidate_summaries = []
    saved_progress = tqdm(CANDIDATE_IDS, desc="saved candidates", unit="candidate")
    for candidate_id in saved_progress:
        candidate_dir = SOURCE_RUN / f"candidate_{candidate_id:05d}"
        family = load_family(candidate_dir / "system.json")
        old_score = load_json(candidate_dir / "score.json")
        theta_max = float(old_score["theta_max"])
        beta = float(old_score["best_beta"])
        label = f"candidate_{candidate_id:05d}"

        saved_progress.set_postfix(candidate=f"{candidate_id:05d}", beta=beta)

        candidate_summary = reevaluate_one(
            output_dir=output_dir,
            label=label,
            family=family,
            theta_max=theta_max,
            beta=beta,
            metadata={"source": "search", "candidate_id": candidate_id, "original_score": old_score},
        )
        all_candidate_summaries.append(candidate_summary)

        tqdm.write(
            f"{label} "
            f"mean_gain={candidate_summary['mean_gain']:.3f} "
            f"tail_gain={candidate_summary['tail_gain']:.3f} "
            f"center_loss={candidate_summary['center_loss']:.3f} "
            f"ce_err={candidate_summary['ce_theta_error_post_warmup']:.3f} "
            f"ua_err={candidate_summary['ua_theta_error_post_warmup']:.3f}"
        )

    manual_progress = tqdm(MANUAL_SYSTEMS, desc="manual systems", unit="system")
    for spec in manual_progress:
        family = manual_family(spec)
        theta_max = float(spec["theta_max"])
        beta = float(spec["beta"])
        label = f"manual_{spec['name']}"
        manual_progress.set_postfix(system=spec["name"], beta=beta)

        candidate_summary = reevaluate_one(
            output_dir=output_dir,
            label=label,
            family=family,
            theta_max=theta_max,
            beta=beta,
            metadata={"source": "manual", "spec": spec},
        )
        all_candidate_summaries.append(candidate_summary)

        tqdm.write(
            f"{label} "
            f"mean_gain={candidate_summary['mean_gain']:.3f} "
            f"tail_gain={candidate_summary['tail_gain']:.3f} "
            f"center_loss={candidate_summary['center_loss']:.3f} "
            f"ce_err={candidate_summary['ce_theta_error_post_warmup']:.3f} "
            f"ua_err={candidate_summary['ua_theta_error_post_warmup']:.3f}"
        )

    all_candidate_summaries.sort(key=lambda row: row["tail_gain"], reverse=True)
    write_csv(output_dir / "candidate_summary.csv", all_candidate_summaries)
    print(f"\nSaved re-evaluation results to: {output_dir}")


if __name__ == "__main__":
    main()

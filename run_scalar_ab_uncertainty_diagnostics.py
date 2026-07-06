from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from scalar_ab_lqr_utils import (
    DEFAULT_RANGE,
    EPISODE_STEPS,
    Q,
    R,
    SELECTION_ROOT,
    _episode_metrics,
    _rollout_lqr,
    _rollout_ppo,
    closed_loop_signature,
    discounted_cost_for_gain,
    load_ppo,
    make_eval_env,
    scalar_env_config,
    scalar_gain,
    save_json,
    write_csv,
)
from unified_context_ppo import UnifiedContextPPO


BLR_POINTER = SELECTION_ROOT / "latest_scalar_ab_blr_ppo.txt"
DIAGNOSTIC_POINTER = SELECTION_ROOT / "latest_scalar_ab_diagnostics.txt"
NEURAL_POINTER = SELECTION_ROOT / "latest_scalar_ab_neural_rma.txt"

MIN_X_FOR_GAIN = 5e-2


def _smoke() -> bool:
    return os.environ.get("SCALAR_AB_UNCERTAINTY_DIAGNOSTIC_SMOKE", "0").lower() in {"1", "true", "yes"}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_run_root(pointer: Path, required_manifest: str) -> tuple[Path | None, dict[str, Any] | None]:
    if not pointer.exists():
        return None, None
    root = Path(pointer.read_text().strip()).resolve()
    manifest_path = root / required_manifest
    if not manifest_path.exists():
        return None, None
    return root, _read_json(manifest_path)


def _load_required_runs() -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path | None, dict[str, Any] | None]:
    blr_root, blr_manifest = _load_run_root(BLR_POINTER, "scalar_ab_blr_ppo.json")
    if blr_root is None or blr_manifest is None:
        raise FileNotFoundError("Missing scalar-ab BLR+PPO run. Run run_scalar_ab_blr_ppo.py first.")
    diag_root, diag_manifest = _load_run_root(DIAGNOSTIC_POINTER, "diagnostics_manifest.json")
    if diag_root is None or diag_manifest is None:
        raise FileNotFoundError("Missing scalar-ab diagnostics run. Run run_scalar_ab_lqr_diagnostics.py first.")
    neural_root, neural_manifest = _load_run_root(NEURAL_POINTER, "scalar_ab_neural_rma.json")
    return blr_root, blr_manifest, diag_root, diag_manifest, neural_root, neural_manifest


def _checkpoint_exists(exp_dir: Path, weights_name: str) -> bool:
    return (exp_dir / f"{weights_name}.zip").exists()


def _controller_specs(blr_manifest: dict[str, Any], neural_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    blr_dirs = {key: Path(value).resolve() for key, value in blr_manifest["dirs"].items()}
    specs = [
        {"label": "privileged", "kind": "ppo", "experiment": str(blr_dirs["privileged"]), "weights_name": "weights"},
        {"label": "blr_mean_P06", "kind": "ppo", "experiment": str(blr_dirs["blr_mean"]), "weights_name": "weights_P06"},
        {
            "label": "blr_mean_cov_P06",
            "kind": "ppo",
            "experiment": str(blr_dirs["blr_mean_cov"]),
            "weights_name": "weights_P06",
        },
        {"label": "oracle_lqr", "kind": "oracle_lqr"},
    ]
    if neural_manifest is not None:
        neural_dirs = {key: Path(value).resolve() for key, value in neural_manifest["dirs"].items()}
        neural_candidates = [
            ("gradual_mle_P06", neural_dirs["gradual_mle"], "weights_P06"),
            ("gradual_nll_P06", neural_dirs["gradual_nll"], "weights_P06"),
            ("staged_vanilla_S02", neural_dirs["staged_policy"], "weights_S02"),
        ]
        for label, exp_dir, weights_name in neural_candidates:
            if _checkpoint_exists(exp_dir, weights_name):
                specs.append({"label": label, "kind": "ppo", "experiment": str(exp_dir), "weights_name": weights_name})
    return specs


def _load_runtimes(specs: list[dict[str, Any]]) -> dict[str, tuple[UnifiedContextPPO, dict[str, Any]] | None]:
    runtimes: dict[str, tuple[UnifiedContextPPO, dict[str, Any]] | None] = {}
    for spec in specs:
        if spec["kind"] == "ppo":
            runtimes[spec["label"]] = load_ppo(Path(spec["experiment"]), spec.get("weights_name", "weights"))
        else:
            runtimes[spec["label"]] = None
    return runtimes


def _select_pairs(diag_root: Path, *, top_pairs: int, max_delta_c: float) -> pd.DataFrame:
    pairs = pd.read_csv(diag_root / "ambiguous_pairs.csv")
    pairs = pairs[pairs["delta_closed_loop"].abs() <= max_delta_c].copy()
    pairs = pairs.sort_values(["max_excess_cost", "delta_K"], ascending=[False, False]).head(top_pairs)
    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = np.arange(len(pairs), dtype=int)
    return pairs


def _selected_endpoint_rows(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        for side in (1, 2):
            rows.append(
                {
                    "pair_id": int(pair["pair_id"]),
                    "side": side,
                    "a": float(pair[f"a{side}"]),
                    "b": float(pair[f"b{side}"]),
                    "K": float(pair[f"K{side}"]),
                    "closed_loop": float(pair[f"c{side}"]),
                    "delta_closed_loop": float(pair["delta_closed_loop"]),
                    "delta_K": float(pair["delta_K"]),
                    "max_excess_cost": float(pair["max_excess_cost"]),
                }
            )
    return rows


def _context_numpy(model: UnifiedContextPPO) -> np.ndarray:
    ctx = np.asarray(getattr(model, "_predict_cached_context"), dtype=np.float64)
    if ctx.ndim != 2 or ctx.shape[0] != 1:
        raise RuntimeError("Expected a single cached context row.")
    return ctx[0].copy()


def _true_context(a: float, b: float, model: UnifiedContextPPO) -> th.Tensor:
    true = np.asarray([[a, b]], dtype=np.float32)
    z_scale = float(getattr(model, "z_scale", 1.0))
    true = true * z_scale
    return th.as_tensor(true, device=model.device, dtype=th.float32)


def _action_from_context(model: UnifiedContextPPO, obs: np.ndarray, context: np.ndarray) -> np.ndarray:
    model.policy.eval()
    with th.no_grad():
        obs_th = th.as_tensor(obs.reshape(1, -1), device=model.device, dtype=th.float32)
        ctx_th = th.as_tensor(context.reshape(1, -1), device=model.device, dtype=th.float32)
        action_th, _, _ = model.policy.forward_with_z(obs_th, ctx_th, deterministic=True)
    action = action_th.detach().cpu().numpy().reshape(-1)
    return np.clip(action, model.action_space.low, model.action_space.high)


def _privileged_action(model: UnifiedContextPPO, obs: np.ndarray, a: float, b: float) -> np.ndarray:
    model.policy.eval()
    with th.no_grad():
        obs_th = th.as_tensor(obs.reshape(1, -1), device=model.device, dtype=th.float32)
        ctx_th = _true_context(a, b, model)
        action_th, _, _ = model.policy.forward_with_z(obs_th, ctx_th, deterministic=True)
    action = action_th.detach().cpu().numpy().reshape(-1)
    return np.clip(action, model.action_space.low, model.action_space.high)


def _cov_from_blr_context(context: np.ndarray) -> np.ndarray:
    std_a = max(float(context[2]), 1e-8)
    std_b = max(float(context[3]), 1e-8)
    corr = float(np.clip(context[4], -0.999, 0.999))
    return np.asarray([[std_a**2, corr * std_a * std_b], [corr * std_a * std_b, std_b**2]], dtype=np.float64)


def _context_from_cov(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    std_a = math.sqrt(max(float(cov[0, 0]), 1e-12))
    std_b = math.sqrt(max(float(cov[1, 1]), 1e-12))
    corr = float(cov[0, 1] / max(std_a * std_b, 1e-12))
    return np.asarray([mean[0], mean[1], std_a, std_b, np.clip(corr, -0.999, 0.999)], dtype=np.float32)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return vec.astype(np.float64) / norm


def _blr_context_variants(context: np.ndarray) -> dict[str, np.ndarray]:
    mean = context[:2].astype(np.float64)
    cov = _cov_from_blr_context(context)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-10))
    k = scalar_gain(float(mean[0]), max(float(mean[1]), 1e-6))
    tangent = _unit(np.asarray([k, 1.0], dtype=np.float64))
    orth = _unit(np.asarray([-tangent[1], tangent[0]], dtype=np.float64))
    major = float(eigvals[-1])
    minor = float(eigvals[0])
    cov_tangent = major * np.outer(tangent, tangent) + minor * np.outer(orth, orth)
    cov_orth = major * np.outer(orth, orth) + minor * np.outer(tangent, tangent)
    high_std = math.sqrt(max(major, 1e-8))
    return {
        "predicted": context.astype(np.float32),
        "zero": np.asarray([mean[0], mean[1], 1e-6, 1e-6, 0.0], dtype=np.float32),
        "diag_only": np.asarray([mean[0], mean[1], context[2], context[3], 0.0], dtype=np.float32),
        "corr_flipped": np.asarray([mean[0], mean[1], context[2], context[3], -context[4]], dtype=np.float32),
        "high_iso": np.asarray([mean[0], mean[1], high_std, high_std, 0.0], dtype=np.float32),
        "tangent_aligned": _context_from_cov(mean, cov_tangent),
        "orthogonal_aligned": _context_from_cov(mean, cov_orth),
    }


def _neural_context_variants(context: np.ndarray) -> dict[str, np.ndarray]:
    mean = context[:2].astype(np.float64)
    std = np.maximum(context[2:4].astype(np.float64), 1e-8)
    return {
        "predicted": context.astype(np.float32),
        "zero": np.asarray([mean[0], mean[1], 1e-6, 1e-6], dtype=np.float32),
        "scaled_0p5": np.asarray([mean[0], mean[1], 0.5 * std[0], 0.5 * std[1]], dtype=np.float32),
        "scaled_2p0": np.asarray([mean[0], mean[1], 2.0 * std[0], 2.0 * std[1]], dtype=np.float32),
        "high_a": np.asarray([mean[0], mean[1], 2.0 * std[0], std[1]], dtype=np.float32),
        "high_b": np.asarray([mean[0], mean[1], std[0], 2.0 * std[1]], dtype=np.float32),
        "high_both": np.asarray([mean[0], mean[1], 2.0 * std[0], 2.0 * std[1]], dtype=np.float32),
    }


def _collect_anchor_rows(
    specs: list[dict[str, Any]],
    runtimes: dict[str, tuple[UnifiedContextPPO, dict[str, Any]] | None],
    endpoints: list[dict[str, Any]],
    *,
    max_anchors: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    labels = {"blr_mean_cov_P06", "gradual_nll_P06"}
    anchor_specs = [spec for spec in specs if spec["label"] in labels and spec["label"] in runtimes]
    privileged_runtime = runtimes.get("privileged")
    if privileged_runtime is None:
        raise RuntimeError("Privileged model is required for intervention distances.")
    privileged_model, _ = privileged_runtime
    anchors: list[dict[str, Any]] = []
    for spec in anchor_specs:
        model_anchor_count = 0
        runtime = runtimes[spec["label"]]
        if runtime is None:
            continue
        model, cfg = runtime
        if not bool(getattr(model, "condition_on_uncertainty", False)):
            continue
        for endpoint_idx, endpoint in enumerate(endpoints):
            if model_anchor_count >= max_anchors:
                break
            a = float(endpoint["a"])
            b = float(endpoint["b"])
            env = make_eval_env(cfg, a, b, base_seed + endpoint_idx)
            model.set_env(DummyVecEnv([lambda: env]))
            obs, _ = env.reset()
            state = None
            episode_start = np.asarray([True])
            for step in range(EPISODE_STEPS):
                action, state = model.predict(obs.reshape(1, -1), state=state, episode_start=episode_start, deterministic=True)
                episode_start = np.asarray([False])
                context = _context_numpy(model)
                x = float(obs[0])
                obs_before = np.asarray(obs, dtype=np.float32).copy()
                u = float(action.reshape(-1)[0])
                if (
                    step >= int(getattr(model, "window_length", 32))
                    and step % int(getattr(model, "id_update_interval", 8)) == 0
                    and abs(x) >= MIN_X_FOR_GAIN
                ):
                    priv_action = _privileged_action(privileged_model, obs_before, a, b)
                    anchors.append(
                        {
                            "anchor_id": len(anchors),
                            "model_label": spec["label"],
                            "pair_id": endpoint.get("pair_id", -1),
                            "side": endpoint.get("side", -1),
                            "a": a,
                            "b": b,
                            "step": step,
                            "x": x,
                            "observed_action": u,
                            "privileged_action": float(priv_action[0]),
                            "context": context,
                            "obs": obs_before,
                        }
                    )
                    model_anchor_count += 1
                    if model_anchor_count >= max_anchors:
                        break
                obs, _, terminated, truncated, _ = env.step(action.reshape(-1))
                if bool(terminated or truncated):
                    break
    return anchors


def _run_fixed_context_intervention(
    specs: list[dict[str, Any]],
    runtimes: dict[str, tuple[UnifiedContextPPO, dict[str, Any]] | None],
    endpoints: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_anchors: int,
) -> list[dict[str, Any]]:
    anchors = _collect_anchor_rows(specs, runtimes, endpoints, max_anchors=max_anchors, base_seed=22001)
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        runtime = runtimes[anchor["model_label"]]
        if runtime is None:
            continue
        model, _ = runtime
        context = anchor["context"]
        if context.shape[0] == 5:
            variants = _blr_context_variants(context)
        elif context.shape[0] == 4:
            variants = _neural_context_variants(context)
        else:
            continue
        predicted_action: np.ndarray | None = None
        for variant_name, variant_context in variants.items():
            action = _action_from_context(model, anchor["obs"], variant_context)
            if variant_name == "predicted":
                predicted_action = action
            x = float(anchor["x"])
            gain = float(action[0] / x)
            rows.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "model_label": anchor["model_label"],
                    "uncertainty_variant": variant_name,
                    "pair_id": anchor["pair_id"],
                    "side": anchor["side"],
                    "a": anchor["a"],
                    "b": anchor["b"],
                    "step": anchor["step"],
                    "x": x,
                    "mean_a": float(context[0]),
                    "mean_b": float(context[1]),
                    "context_dim": int(context.shape[0]),
                    "variant_action": float(action[0]),
                    "variant_gain": gain,
                    "observed_action": float(anchor["observed_action"]),
                    "privileged_action": float(anchor["privileged_action"]),
                    "distance_to_privileged": abs(float(action[0]) - float(anchor["privileged_action"])),
                    "action_cost": float(R * action[0] * action[0]),
                    "distance_to_predicted": 0.0 if predicted_action is None else abs(float(action[0] - predicted_action[0])),
                }
            )
    write_csv(out_dir / "fixed_context_intervention.csv", rows)
    _plot_fixed_context_intervention(rows, out_dir)
    return rows


def _rollout_controller_exact(
    spec: dict[str, Any],
    runtime: tuple[UnifiedContextPPO, dict[str, Any]] | None,
    a: float,
    b: float,
    seed: int,
) -> dict[str, Any]:
    if spec["kind"] == "oracle_lqr":
        cfg = scalar_env_config(DEFAULT_RANGE)
        env = make_eval_env(cfg, a, b, seed)
        ep = _rollout_lqr(env, assumed_a=a, assumed_b=b)
    else:
        if runtime is None:
            raise RuntimeError(f"Missing runtime for {spec['label']}")
        model, cfg = runtime
        env = make_eval_env(cfg, a, b, seed)
        ep, _ = _rollout_ppo(env, model, collect_trace=False)
    ep.update({"controller": spec["label"], "a": a, "b": b, "seed": seed})
    k_opt = scalar_gain(a, b)
    oracle_cost = discounted_cost_for_gain(a, b, k_opt)
    ep["oracle_x1_cost"] = oracle_cost
    ep["regret_to_oracle_x1_cost"] = float(ep["true_cost"] - oracle_cost)
    return ep


def _run_ambiguous_stress(
    specs: list[dict[str, Any]],
    runtimes: dict[str, tuple[UnifiedContextPPO, dict[str, Any]] | None],
    endpoints: list[dict[str, Any]],
    out_dir: Path,
    *,
    seeds_per_endpoint: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint_idx, endpoint in enumerate(endpoints):
        for seed_idx in range(seeds_per_endpoint):
            seed = 31001 + seed_idx
            for spec in specs:
                runtime = runtimes.get(spec["label"])
                ep = _rollout_controller_exact(spec, runtime, float(endpoint["a"]), float(endpoint["b"]), seed)
                ep.update({key: endpoint[key] for key in ["pair_id", "side", "K", "closed_loop", "delta_closed_loop", "delta_K", "max_excess_cost"]})
                rows.append(ep)
    write_csv(out_dir / "ambiguous_stress_episode_summary.csv", rows)
    scorecard = _stress_scorecard(rows)
    write_csv(out_dir / "ambiguous_stress_scorecard.csv", scorecard)
    pair_scorecard = _stress_pair_scorecard(rows)
    write_csv(out_dir / "ambiguous_stress_pair_scorecard.csv", pair_scorecard)
    _plot_ambiguous_stress(rows, scorecard, out_dir)
    return rows


def _stress_scorecard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["controller"])].append(row)
    out: list[dict[str, Any]] = []
    for controller, bucket in sorted(grouped.items()):
        item = {"controller": controller, "n": len(bucket)}
        for key in ["return", "true_cost", "state_cost", "action_cost", "regret_to_oracle_x1_cost", "gram_logdet", "gram_condition", "u_over_x_var"]:
            vals = np.asarray([float(row[key]) for row in bucket], dtype=np.float64)
            item[f"{key}_mean"] = float(np.mean(vals))
            item[f"{key}_std"] = float(np.std(vals))
        out.append(item)
    return out


def _stress_pair_scorecard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["pair_id"]), str(row["controller"]))].append(row)
    out: list[dict[str, Any]] = []
    for (pair_id, controller), bucket in sorted(grouped.items()):
        item = {"pair_id": pair_id, "controller": controller, "n": len(bucket)}
        for key in ["return", "true_cost", "regret_to_oracle_x1_cost", "gram_logdet"]:
            vals = np.asarray([float(row[key]) for row in bucket], dtype=np.float64)
            item[f"{key}_mean"] = float(np.mean(vals))
        out.append(item)
    return out


def _plot_selected_pairs(diag_root: Path, pairs: pd.DataFrame, out_dir: Path) -> None:
    analytic = pd.read_csv(diag_root / "analytic_grid.csv")
    a_vals = np.sort(analytic["a"].unique())
    b_vals = np.sort(analytic["b"].unique())
    grid = analytic.pivot(index="b", columns="a", values="closed_loop").loc[b_vals, a_vals].values
    extent = [a_vals.min(), a_vals.max(), b_vals.min(), b_vals.max()]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="closed-loop signature c")
    for _, pair in pairs.iterrows():
        ax.plot([pair["a1"], pair["a2"]], [pair["b1"], pair["b2"]], color="white", lw=1.0, alpha=0.6)
    ax.scatter(pairs["a1"], pairs["b1"], color="tab:orange", s=18, label="endpoint 1")
    ax.scatter(pairs["a2"], pairs["b2"], color="tab:red", s=18, label="endpoint 2")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("Selected close-signature ambiguous pairs")
    ax.legend()
    fig.savefig(out_dir / "ambiguous_manifold_selected_pairs.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title in [
        (axs[0], "delta_closed_loop", "|delta closed-loop|"),
        (axs[1], "delta_K", "|delta K|"),
        (axs[2], "max_excess_cost", "max excess cost"),
    ]:
        ax.hist(pairs[key], bins=20, alpha=0.85)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "ambiguous_pair_metric_distribution.png", dpi=180)
    plt.close(fig)


def _plot_fixed_context_intervention(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    metrics = [
        ("distance_to_predicted", "Action change from predicted"),
        ("distance_to_privileged", "Distance to privileged action"),
        ("action_cost", "Immediate action cost"),
        ("variant_gain", "Effective gain u/x"),
    ]
    for model_label, sub in df.groupby("model_label"):
        order = list(dict.fromkeys(sub["uncertainty_variant"].tolist()))
        fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for ax, (metric, title) in zip(axs.ravel(), metrics):
            vals = sub.groupby("uncertainty_variant")[metric].mean().reindex(order)
            errs = sub.groupby("uncertainty_variant")[metric].std().reindex(order).fillna(0.0)
            ax.bar(np.arange(len(order)), vals.to_numpy(), yerr=errs.to_numpy(), capsize=3, alpha=0.85)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order, rotation=35, ha="right")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(f"Fixed-context uncertainty intervention: {model_label}", fontsize=15)
        fig.savefig(out_dir / f"fixed_context_action_response_{model_label}.png", dpi=180)
        plt.close(fig)


def _plot_ambiguous_stress(rows: list[dict[str, Any]], scorecard: list[dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    sc = pd.DataFrame(scorecard)
    order = sc.sort_values("return_mean", ascending=False)["controller"].tolist()
    fig, axs = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    for ax, metric, title in [
        (axs[0], "return_mean", "Return"),
        (axs[1], "regret_to_oracle_x1_cost_mean", "Regret to oracle x=1"),
        (axs[2], "gram_logdet_mean", "Gram logdet"),
        (axs[3], "action_cost_mean", "Action cost"),
    ]:
        vals = sc.set_index("controller").loc[order, metric]
        ax.bar(np.arange(len(order)), vals.to_numpy(), alpha=0.85)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Ambiguous-pair stress evaluation", fontsize=15)
    fig.savefig(out_dir / "ambiguous_stress_return_bars.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for controller, sub in df.groupby("controller"):
        ax.scatter(sub["gram_logdet"], sub["return"], s=14, alpha=0.45, label=controller)
    ax.set_xlabel("Gram logdet")
    ax.set_ylabel("Return")
    ax.set_title("Information/return tradeoff on ambiguous endpoints")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "ambiguous_stress_information_tradeoff.png", dpi=180)
    plt.close(fig)

    _plot_pair_delta(df, out_dir, "blr_mean_cov_P06", "blr_mean_P06", "mean+cov - mean")
    _plot_pair_delta(df, out_dir, "gradual_nll_P06", "gradual_mle_P06", "NLL - MLE")


def _plot_pair_delta(df: pd.DataFrame, out_dir: Path, left: str, right: str, label: str) -> None:
    if left not in set(df["controller"]) or right not in set(df["controller"]):
        return
    grouped = df.groupby(["pair_id", "side", "a", "b", "controller"])["return"].mean().reset_index()
    wide = grouped.pivot_table(index=["pair_id", "side", "a", "b"], columns="controller", values="return").reset_index()
    wide["delta"] = wide[left] - wide[right]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lim = float(np.nanpercentile(np.abs(wide["delta"]), 98)) or 1.0
    sc = axs[0].scatter(wide["a"], wide["b"], c=wide["delta"], cmap="coolwarm", vmin=-lim, vmax=lim, s=38)
    axs[0].set_xlabel("a")
    axs[0].set_ylabel("b")
    axs[0].set_title(f"Endpoint return delta: {label}")
    plt.colorbar(sc, ax=axs[0], label="return delta")
    axs[1].hist(wide["delta"], bins=25, alpha=0.85)
    axs[1].axvline(0, color="black", lw=1.5)
    axs[1].axvline(wide["delta"].mean(), color="crimson", lw=2, label=f"mean {wide['delta'].mean():.4f}")
    axs[1].set_title(f"Distribution: {label}")
    axs[1].set_xlabel("return delta")
    axs[1].legend()
    fig.savefig(out_dir / f"ambiguous_stress_return_delta_{left}_minus_{right}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    blr_root, blr_manifest, diag_root, _, neural_root, neural_manifest = _load_required_runs()
    smoke = _smoke()
    out_dir = blr_root / ("scalar_ab_uncertainty_diagnostic_smoke" if smoke else "scalar_ab_uncertainty_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    top_pairs = 3 if smoke else 50
    seeds_per_endpoint = 1 if smoke else 20
    max_anchors = 20 if smoke else 240
    max_delta_c = 0.002
    pairs = _select_pairs(diag_root, top_pairs=top_pairs, max_delta_c=max_delta_c)
    if pairs.empty:
        raise RuntimeError(f"No ambiguous pairs found with |delta_closed_loop| <= {max_delta_c}.")
    pairs.to_csv(out_dir / "selected_ambiguous_pairs.csv", index=False)
    endpoints = _selected_endpoint_rows(pairs)
    write_csv(out_dir / "selected_ambiguous_endpoints.csv", endpoints)
    _plot_selected_pairs(diag_root, pairs, out_dir)

    specs = _controller_specs(blr_manifest, neural_manifest)
    runtimes = _load_runtimes(specs)
    fixed_rows = _run_fixed_context_intervention(specs, runtimes, endpoints, out_dir, max_anchors=max_anchors)
    stress_rows = _run_ambiguous_stress(specs, runtimes, endpoints, out_dir, seeds_per_endpoint=seeds_per_endpoint)

    manifest = {
        "blr_root": str(blr_root.resolve()),
        "diagnostics_root": str(diag_root.resolve()),
        "neural_root": str(neural_root.resolve()) if neural_root is not None else None,
        "smoke": smoke,
        "top_pairs": top_pairs,
        "seeds_per_endpoint": seeds_per_endpoint,
        "max_delta_closed_loop": max_delta_c,
        "controllers": [spec["label"] for spec in specs],
        "num_fixed_context_rows": len(fixed_rows),
        "num_stress_rows": len(stress_rows),
    }
    save_json(out_dir / "scalar_ab_uncertainty_diagnostic.json", manifest)
    print(f"Saved scalar-ab uncertainty diagnostics to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

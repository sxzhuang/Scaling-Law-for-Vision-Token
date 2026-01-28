import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path("data_analysis/data4analysis_preprocess.csv")
DEFAULT_PIECEWISE_PARAMS_PATH = Path("data_analysis/law_relation/piecewise_fit_params.csv")
DEFAULT_OUTPUT_DIR = Path("data_analysis/law_relation")

DEFAULT_EPS = 1e-6
DEFAULT_MIN_GROUP_POINTS = 8
DEFAULT_PLOT_DPI = 200


def make_group_id(resolution: int, fs: float, ls: float, cs: float) -> str:
    return f"R={int(resolution)}|fs={fs:g}|ls={ls:g}|cs={cs:g}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def load_piecewise_maps(piecewise_params_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    df_piecewise = pd.read_csv(piecewise_params_path)
    if "group_id" in df_piecewise.columns:
        key = df_piecewise["group_id"].astype(str)
    else:
        key = df_piecewise.apply(lambda r: make_group_id(r["resolution"], r["font_size"], r["line_spacing"], r["letter_spacing"]), axis=1).astype(str)
    if "L_break" not in df_piecewise.columns:
        raise ValueError(f"Missing column 'L_break' in {piecewise_params_path}")
    if "L0" not in df_piecewise.columns:
        raise ValueError(f"Missing column 'L0' in {piecewise_params_path}")
    l_break = df_piecewise["L_break"].astype(float)
    l0 = df_piecewise["L0"].astype(float)
    l_break_map = {gid: float(lb) for gid, lb in zip(key.to_numpy(), l_break.to_numpy())}
    l0_map = {gid: float(lv) for gid, lv in zip(key.to_numpy(), l0.to_numpy())}
    return l_break_map, l0_map


def fit_group_powerlaw(gdf: pd.DataFrame, l0: float, eps: float) -> dict[str, float]:
    token_len = gdf["token_len"].astype(float).to_numpy()
    edit_distance = gdf["edit_distance"].astype(float).to_numpy()
    keep = token_len > float(l0)
    if int(np.sum(keep)) < 3:
        return {"n_points": int(np.sum(keep)), "theta": float("nan"), "logA": float("nan"), "se_theta": float("nan"), "se_logA": float("nan"), "r2": float("nan"), "rmse": float("nan")}

    x = np.log((token_len[keep] - float(l0)) + float(eps))
    y = np.log(edit_distance[keep] + float(eps))
    n = int(x.size)
    X = np.column_stack([np.ones(n, dtype=float), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    log_a = float(coef[0])
    theta = float(coef[1])
    y_hat = X @ coef

    resid = y - y_hat
    rss = float(np.sum(resid ** 2))
    rmse = float(math.sqrt(rss / max(n, 1)))
    r2 = float(_safe_r2(y_true=y, y_pred=y_hat))

    if n <= 2:
        return {"n_points": n, "theta": theta, "logA": log_a, "se_theta": float("nan"), "se_logA": float("nan"), "r2": r2, "rmse": rmse}

    s2 = rss / float(n - 2)
    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx <= 1e-12:
        se_theta = float("nan")
        se_log_a = float("nan")
    else:
        se_theta = float(math.sqrt(s2 / sxx))
        se_log_a = float(math.sqrt(s2 * (1.0 / float(n) + (x_bar ** 2) / sxx)))

    return {"n_points": n, "theta": theta, "logA": log_a, "se_theta": se_theta, "se_logA": se_log_a, "r2": r2, "rmse": rmse}


def _standardize(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=float)
    means = Z.mean(axis=0)
    stds = Z.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    Zs = (Z - means) / stds
    return Zs, means, stds


def _fit_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta.astype(float)


def _coef_standardized_to_raw(beta_std: np.ndarray, feat_means: np.ndarray, feat_stds: np.ndarray) -> np.ndarray:
    beta_std = np.asarray(beta_std, dtype=float)
    feat_means = np.asarray(feat_means, dtype=float)
    feat_stds = np.asarray(feat_stds, dtype=float)
    beta_raw = beta_std.copy()
    beta_raw[1:] = beta_std[1:] / feat_stds
    beta_raw[0] = beta_std[0] - float(np.sum(beta_std[1:] * feat_means / feat_stds))
    return beta_raw.astype(float)


def _pair_list(values: list[float]) -> list[tuple[float, float | None]]:
    pairs: list[tuple[float, float | None]] = []
    idx = 0
    while idx < len(values):
        left = values[idx]
        right = values[idx + 1] if (idx + 1) < len(values) else None
        pairs.append((left, right))
        idx += 2
    return pairs


def _robust_limits(vals: list[float], pad: float = 0.05) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return -1.0, 1.0
    lo = float(np.quantile(arr, 0.01))
    hi = float(np.quantile(arr, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if abs(hi - lo) < 1e-9:
        lo -= 1.0
        hi += 1.0
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def _compute_training_xy(gdf: pd.DataFrame, l0: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    token_len = gdf["token_len"].astype(float).to_numpy()
    edit_distance = gdf["edit_distance"].astype(float).to_numpy()
    keep = token_len > float(l0)
    x = np.log((token_len[keep] - float(l0)) + float(eps))
    y = np.log(edit_distance[keep] + float(eps))
    return x.astype(float), y.astype(float)


def plot_loglog_resolution(
    df_r: pd.DataFrame,
    res: int,
    lbreak_map: dict[str, float],
    l0_map: dict[str, float],
    group_fit_map: dict[str, dict[str, float]],
    output_dir: Path,
    eps: float,
    dpi: int,
) -> Path:
    fs_list = sorted(df_r["font_size"].astype(float).unique().tolist())
    ls_list = sorted(df_r["line_spacing"].astype(float).unique().tolist())
    cs_list = sorted(df_r["letter_spacing"].astype(float).unique().tolist())
    cs_fixed = cs_list[:3]
    while len(cs_fixed) < 3:
        cs_fixed.append(None)

    pairs = _pair_list(ls_list)
    ncols = 6
    nrows = max(1, len(fs_list) * len(pairs))

    all_x: list[float] = []
    all_y: list[float] = []
    for fs in fs_list:
        for (ls_a, ls_b) in pairs:
            for ls_val in [ls_a, ls_b]:
                if ls_val is None:
                    continue
                for cs in cs_fixed:
                    if cs is None:
                        continue
                    gid = make_group_id(resolution=res, fs=float(fs), ls=float(ls_val), cs=float(cs))
                    if gid not in lbreak_map or gid not in l0_map:
                        continue
                    gdf = df_r[(df_r["font_size"] == fs) & (df_r["line_spacing"] == ls_val) & (df_r["letter_spacing"] == cs)]
                    if gdf.empty:
                        continue
                    x, y = _compute_training_xy(gdf=gdf, l0=float(l0_map[gid]), eps=eps)
                    all_x.extend(x.tolist())
                    all_y.extend(y.tolist())

    x_min, x_max = _robust_limits(all_x)
    y_min, y_max = _robust_limits(all_y)

    fig_w = 22
    fig_h = max(6, 2.8 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"log-log powerlaw fit | resolution={res} | eps={eps:g}", fontsize=14)

    row_idx = 0
    for fs in fs_list:
        for (ls_a, ls_b) in pairs:
            for j, ls_val in enumerate([ls_a, ls_b]):
                for k, cs in enumerate(cs_fixed):
                    col_idx = j * 3 + k
                    ax = axes[row_idx, col_idx]
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.grid(True, alpha=0.2)
                    ax.set_xlabel("log(L - L0 + eps)")
                    ax.set_ylabel("log(E + eps)")

                    if ls_val is None or cs is None:
                        ax.axis("off")
                        continue

                    gid = make_group_id(resolution=res, fs=float(fs), ls=float(ls_val), cs=float(cs))
                    gdf = df_r[(df_r["font_size"] == fs) & (df_r["line_spacing"] == ls_val) & (df_r["letter_spacing"] == cs)]
                    if gdf.empty or gid not in lbreak_map or gid not in l0_map:
                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g} (missing)")
                        continue

                    x, y = _compute_training_xy(gdf=gdf, l0=float(l0_map[gid]), eps=eps)
                    if x.size == 0:
                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g} (no train pts)")
                        continue

                    ax.scatter(x, y, s=8, alpha=0.6, linewidths=0.0)

                    fit = group_fit_map.get(gid)
                    if fit is not None:
                        theta = float(fit.get("theta", float("nan")))
                        log_a = float(fit.get("logA", float("nan")))
                    else:
                        theta = float("nan")
                        log_a = float("nan")

                    if np.isfinite(theta) and np.isfinite(log_a):
                        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                        yy = float(log_a) + float(theta) * xx
                        ax.plot(xx, yy, color="crimson", linewidth=1.5)
                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g} | n={x.size} | θ={theta:.3f}")
                    else:
                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g} | n={x.size}")

            row_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = output_dir / f"powerlaw_phase2_loglog_resolution_{res}.png"
    plt.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def plot_loglog_by_resolution(
    df: pd.DataFrame,
    lbreak_map: dict[str, float],
    l0_map: dict[str, float],
    group_fit_map: dict[str, dict[str, float]],
    output_dir: Path,
    eps: float,
    dpi: int,
) -> None:
    for res in sorted(df["resolution"].astype(int).unique().tolist()):
        df_r = df[df["resolution"].astype(int) == int(res)].copy()
        out_path = plot_loglog_resolution(
            df_r=df_r,
            res=int(res),
            lbreak_map=lbreak_map,
            l0_map=l0_map,
            group_fit_map=group_fit_map,
            output_dir=output_dir,
            eps=float(eps),
            dpi=int(dpi),
        )
        logging.info("Saved plot: %s", str(out_path))


def run(csv_path: Path, piecewise_params_path: Path, output_dir: Path, eps: float, min_group_points: int, r_feature: str, weight_mode: str, plot_dpi: int, use_interactions: bool) -> None:
    ensure_dir(output_dir)
    logging.info("Loading data: %s", str(csv_path))
    df = pd.read_csv(csv_path)
    expected_cols = {"resolution", "font_size", "line_spacing", "letter_spacing", "token_len", "edit_distance"}
    missing = sorted(expected_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    logging.info("Loading piecewise params: %s", str(piecewise_params_path))
    lbreak_map, l0_map = load_piecewise_maps(piecewise_params_path)

    group_rows: list[dict[str, float | str | int]] = []
    for (res, fs, ls, cs), gdf in df.groupby(["resolution", "font_size", "line_spacing", "letter_spacing"]):
        gid = make_group_id(resolution=int(res), fs=float(fs), ls=float(ls), cs=float(cs))
        if gid not in lbreak_map:
            continue
        l0 = float(l0_map[gid])
        fit = fit_group_powerlaw(gdf=gdf, l0=l0, eps=eps)
        group_rows.append(
            {
                "group_id": gid,
                "resolution": int(res),
                "font_size": float(fs),
                "line_spacing": float(ls),
                "letter_spacing": float(cs),
                "L0": float(l0),
                "L_break": float(lbreak_map[gid]),
                "n_points": int(fit["n_points"]),
                "theta": float(fit["theta"]),
                "logA": float(fit["logA"]),
                "se_theta": float(fit["se_theta"]),
                "se_logA": float(fit["se_logA"]),
                "r2": float(fit["r2"]),
                "rmse": float(fit["rmse"]),
            }
        )

    df_group = pd.DataFrame(group_rows)
    group_fit_path = output_dir / "powerlaw_right_of_lbreak_group_fit.csv"
    df_group.to_csv(group_fit_path, index=False)
    logging.info("Saved per-group power-law fit: %s (n_groups=%d)", str(group_fit_path), int(df_group.shape[0]))

    group_fit_map = {}
    for _, row in df_group.iterrows():
        gid = str(row["group_id"])
        group_fit_map[gid] = {"theta": float(row["theta"]), "logA": float(row["logA"])}

    plot_loglog_by_resolution(
        df=df,
        lbreak_map=lbreak_map,
        l0_map=l0_map,
        group_fit_map=group_fit_map,
        output_dir=output_dir,
        eps=float(eps),
        dpi=int(plot_dpi),
    )

    df_meta = df_group.copy()
    df_meta = df_meta[df_meta["n_points"].astype(int) >= int(min_group_points)].copy()
    df_meta = df_meta[np.isfinite(df_meta["theta"].astype(float))].copy()
    if df_meta.empty:
        raise RuntimeError("No valid groups for phase2 meta fit after filtering; try lowering --min_group_points.")

    if str(r_feature).lower() == "log":
        r_col = np.log(df_meta["resolution"].astype(float).to_numpy())
        r_name = "log_resolution"
    else:
        r_col = df_meta["resolution"].astype(float).to_numpy()
        r_name = "resolution"

    fs = df_meta["font_size"].astype(float).to_numpy()
    ls = df_meta["line_spacing"].astype(float).to_numpy()
    cs = df_meta["letter_spacing"].astype(float).to_numpy()
    base_cols = [fs, ls, cs, r_col]
    feat_names = ["font_size", "line_spacing", "letter_spacing", r_name]

    if use_interactions:
        base_cols.extend([r_col * fs, r_col * ls, r_col * cs])
        feat_names.extend([f"{r_name}_x_font_size", f"{r_name}_x_line_spacing", f"{r_name}_x_letter_spacing"])

    Z = np.column_stack(base_cols)
    Zs, feat_means, feat_stds = _standardize(Z)
    X = np.column_stack([np.ones(Zs.shape[0], dtype=float), Zs])
    y = df_meta["theta"].astype(float).to_numpy()

    if str(weight_mode).lower() == "se":
        se = df_meta["se_theta"].astype(float).to_numpy()
        w = np.where(np.isfinite(se) & (se > 1e-12), 1.0 / (se ** 2), np.nan)
        npts = df_meta["n_points"].astype(float).to_numpy()
        w = np.where(np.isfinite(w), w, np.maximum(npts, 1.0))
    elif str(weight_mode).lower() == "n":
        w = np.maximum(df_meta["n_points"].astype(float).to_numpy(), 1.0)
    else:
        w = np.ones_like(y, dtype=float)

    beta_std = _fit_wls(X=X, y=y, w=w)
    beta_raw = _coef_standardized_to_raw(beta_std=beta_std, feat_means=feat_means, feat_stds=feat_stds)
    y_pred = X @ beta_std
    r2_theta = float(_safe_r2(y_true=y, y_pred=y_pred))

    meta_pred_path = output_dir / "theta_meta_predictions.csv"
    df_out = df_meta.copy()
    df_out["theta_pred"] = y_pred.astype(float)
    df_out["weight"] = w.astype(float)
    df_out.to_csv(meta_pred_path, index=False)
    logging.info("Saved meta predictions: %s", str(meta_pred_path))

    meta_json_path = output_dir / "theta_meta_fit.json"
    meta_payload = {
        "model": "theta = beta0 + sum_j beta_j * x_j",
        "target": "theta",
        "features": feat_names,
        "feature_means": feat_means.tolist(),
        "feature_stds": feat_stds.tolist(),
        "beta_standardized": beta_std.tolist(),
        "beta_raw": beta_raw.tolist(),
        "r2_theta": r2_theta,
        "n_groups_used": int(df_meta.shape[0]),
        "min_group_points": int(min_group_points),
        "eps": float(eps),
        "weight_mode": str(weight_mode).lower(),
        "filter_rule": "L > L0",
        "use_interactions": bool(use_interactions),
    }
    meta_json_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Saved meta fit: %s (r2=%.4f)", str(meta_json_path), float(r2_theta))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase2: refit per-group power-law on L>L0 and regress theta on (fs,ls,cs,R) with weighting.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Path to data4analysis_preprocess.csv.")
    parser.add_argument("--piecewise_params_path", type=Path, default=DEFAULT_PIECEWISE_PARAMS_PATH, help="Path to piecewise_fit_params.csv (must contain L_break).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save phase2 outputs.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Epsilon for log(E+eps) and log(L-L_break+eps).")
    parser.add_argument("--min_group_points", type=int, default=DEFAULT_MIN_GROUP_POINTS, help="Minimum right-side samples per group to use in meta fit.")
    parser.add_argument("--plot_dpi", type=int, default=DEFAULT_PLOT_DPI, help="DPI for saved log-log plots.")
    parser.add_argument("--r_feature", type=str, default="raw", choices=["raw", "log"], help="Use raw resolution or log(resolution) as feature.")
    parser.add_argument("--weight_mode", type=str, default="se", choices=["se", "n", "none"], help="Weights for meta fit: inverse se(theta)^2, n_points, or none.")
    parser.add_argument("--use_interactions", action="store_true", help="Include interaction terms: resolution x (fs, ls, cs) in theta fit.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    run(csv_path=args.csv_path, piecewise_params_path=args.piecewise_params_path, output_dir=args.output_dir, eps=float(args.eps), min_group_points=int(args.min_group_points), r_feature=str(args.r_feature), weight_mode=str(args.weight_mode), plot_dpi=int(args.plot_dpi), use_interactions=bool(args.use_interactions))


if __name__ == "__main__":
    main()

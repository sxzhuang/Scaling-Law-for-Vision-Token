import os
import json
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, RANSACRegressor


warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "data_analysis/data4analysis_alltype_default_preprocess.csv"
OUTPUT_DIR = "data_analysis/fit_scaling_law_plots"
BIN_PLOT_DIR = "data_analysis/data4analysis_alltype_default_preprocess_bins"
LAW_RELATION_DIR = "data_analysis/law_relation_alltype"

# Threshold estimation (Step 1)
BIN_WIDTH = 100                # text_len bin width
Q_FOR_THRESHOLD = 0.90         # quantile to detect "lift-off"
EPS_THRESHOLD = 0.01           # treat <= EPS as ~0
K_CONSECUTIVE = 3              # require K consecutive bins above EPS

# Model fit (Step 3)
DELTA_RIGHT = 0                # only fit right side for text_len > L0
EPS_CLIP = 1e-6                # clip edit_distance into (0,1) for log transforms
SHRINK_W = 0.70                # for uncensored groups: blend local L0_hat with regression prediction

# Diagnostics (after Step 3)
MIN_GROUP_POINTS_DIAG = 8      # minimum samples per group for per-group linearity checks

RANDOM_SEED = 77

# Piecewise linear fit in log-log space (after L0 is chosen)
BREAK_Q_START = 0.20
BREAK_Q_END = 0.80
BREAK_Q_STEP = 0.10
LOG_E_MIN = -6.0
K_NEAREST = 10
SPARSE_TRIM_FRACTION_BY_RES = {
    512: 0.05,
    640: 0.05,
    1024: 0.10,
    1280: 0.15,
}
RANSAC_RESIDUAL_THRESHOLD = 0.05
RANSAC_MIN_SAMPLES = 0.9
RANSAC_MAX_TRIALS = 300
MIN_LOG_L_BY_RES = {
    1280: 9.15,
}


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def make_group_id(resolution: int, fs: float, ls: float, cs: float) -> str:
    return f"R={int(resolution)}|fs={fs:g}|ls={ls:g}|cs={cs:g}"


def make_style_id(fs: float, ls: float, cs: float) -> str:
    return f"fs={fs:g}|ls={ls:g}|cs={cs:g}"


def compute_bin_quantiles(
    text_len: np.ndarray,
    edit_distance: np.ndarray,
    bin_width: int = BIN_WIDTH,
    q: float = Q_FOR_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    if text_len.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    L = text_len.astype(float)
    E = edit_distance.astype(float)

    bin_idx = np.floor_divide(L.astype(int), int(bin_width)).astype(int)
    df_tmp = pd.DataFrame({"bin": bin_idx, "E": E})
    q_by_bin = df_tmp.groupby("bin")["E"].quantile(q)

    bmin = int(q_by_bin.index.min())
    bmax = int(q_by_bin.index.max())
    full_bins = pd.Index(range(bmin, bmax + 1), name="bin")
    q_by_bin = q_by_bin.reindex(full_bins).fillna(0.0)

    left_edges = q_by_bin.index.values.astype(float) * float(bin_width)
    return left_edges, q_by_bin.values.astype(float)


def estimate_threshold_local(
    text_len: np.ndarray,
    edit_distance: np.ndarray,
    bin_width: int = BIN_WIDTH,
    q: float = Q_FOR_THRESHOLD,
    eps: float = EPS_THRESHOLD,
    k_consecutive: int = K_CONSECUTIVE,
) -> Tuple[float, bool, float]:
    """
    Returns (L0_hat, is_censored, Lmax).
    - If no stable lift-off is found, returns is_censored=True and L0_hat=Lmax.
    """
    if text_len.size == 0:
        return 0.0, True, 0.0

    L = text_len.astype(float)
    E = edit_distance.astype(float)
    Lmax = float(np.max(L))

    left_edges, q_values = compute_bin_quantiles(
        text_len=L,
        edit_distance=E,
        bin_width=bin_width,
        q=q,
    )

    above = (q_values > float(eps))
    if above.size < k_consecutive:
        return Lmax, True, Lmax

    for i in range(0, above.size - k_consecutive + 1):
        if np.all(above[i : i + k_consecutive]):
            L0_hat = float(max(left_edges[i] - 0.5 * float(bin_width), 0.0))
            return L0_hat, False, Lmax

    return Lmax, True, Lmax


def _fit_hinge_ols(x: np.ndarray, y: np.ndarray, x_break: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    hinge = np.maximum(x - float(x_break), 0.0)
    X = np.column_stack([np.ones(x.size), x, hinge])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    return coef.astype(float), y_hat.astype(float)


def _fit_linear_ols(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(x.size), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    return coef.astype(float), y_hat.astype(float)


def _r2_from_y_hat(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    if y.size < 2 or y_hat.size != y.size:
        return float("nan")
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _sparse_trim_mask(
    x: np.ndarray,
    y: np.ndarray,
    k_nearest: int,
    sparse_trim_frac: float,
) -> np.ndarray:
    if float(sparse_trim_frac) <= 0.0 or x.size <= int(k_nearest):
        return np.ones(x.size, dtype=bool)
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=int(k_nearest) + 1)
    dk = distances[:, -1]
    dk_cut = float(np.quantile(dk, 1.0 - float(sparse_trim_frac)))
    keep = dk <= dk_cut
    if not np.any(keep):
        return np.ones(x.size, dtype=bool)
    return keep


def fit_piecewise_by_quantile_grid(
    text_len: np.ndarray,
    edit_distance: np.ndarray,
    l0: float,
    k_nearest: int = K_NEAREST,
    sparse_trim_frac: float = 0.0,
    use_ransac: bool = False,
    ransac_residual_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    ransac_min_samples: float = RANSAC_MIN_SAMPLES,
    min_log_l: Optional[float] = None,
    q_start: float = BREAK_Q_START,
    q_end: float = BREAK_Q_END,
    q_step: float = BREAK_Q_STEP,
) -> Tuple[dict, np.ndarray]:
    """
    Fit a single line in (x,y) where:
      x = log(L)
      y = log(E)
    Only uses points with L > L0 and log(E) >= LOG_E_MIN.

    Returns (best_row, best_coef), where:
      best_row has keys: xb, L_break, b1, b2, r2, n_points
      best_coef is [a, b1] in y = a + b1*x
    """
    L = np.asarray(text_len, dtype=float)
    E = np.asarray(edit_distance, dtype=float)

    keep = (L > float(l0)) & (E > 0)
    if not np.any(keep):
        return {"fit_type": "linear", "xb": float("nan"), "L_break": float("nan"), "b1": float("nan"), "b2": float("nan"), "r2": float("nan"), "n_points": 0}, np.array([])

    L_keep = L[keep]
    E_keep = E[keep]
    log_e_keep = np.log(E_keep)
    keep_log_e = log_e_keep >= float(LOG_E_MIN)
    if not np.any(keep_log_e):
        return {"fit_type": "linear", "xb": float("nan"), "L_break": float("nan"), "b1": float("nan"), "b2": float("nan"), "r2": float("nan"), "n_points": 0}, np.array([])

    L_keep = L_keep[keep_log_e]
    E_keep = E_keep[keep_log_e]
    order = np.argsort(L_keep)
    L_keep = L_keep[order]
    E_keep = E_keep[order]

    x = np.log(L_keep)
    y = np.log(E_keep)

    if min_log_l is not None:
        keep_log_l = x >= float(min_log_l)
        if not np.any(keep_log_l):
            return {"fit_type": "linear", "xb": float("nan"), "L_break": float("nan"), "b1": float("nan"), "b2": float("nan"), "r2": float("nan"), "n_points": 0}, np.array([])
        L_keep = L_keep[keep_log_l]
        E_keep = E_keep[keep_log_l]
        x = x[keep_log_l]
        y = y[keep_log_l]

    keep_sparse = _sparse_trim_mask(x, y, k_nearest, sparse_trim_frac)
    L_keep = L_keep[keep_sparse]
    E_keep = E_keep[keep_sparse]
    x = x[keep_sparse]
    y = y[keep_sparse]

    fit_type = "linear"
    coef = np.array([], dtype=float)
    y_hat = np.array([], dtype=float)
    if use_ransac and x.size >= 2:
        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(fit_intercept=True),
                min_samples=float(ransac_min_samples),
                residual_threshold=float(ransac_residual_threshold),
                max_trials=int(RANSAC_MAX_TRIALS),
                random_state=RANDOM_SEED,
            )
            ransac.fit(x.reshape(-1, 1), y)
            base = ransac.estimator_
            coef = np.array([float(base.intercept_), float(base.coef_[0])], dtype=float)
            y_hat = coef[0] + coef[1] * x
            fit_type = "ransac"
        except Exception:
            coef = np.array([], dtype=float)
            y_hat = np.array([], dtype=float)

    if coef.size != 2 or y_hat.size != y.size:
        coef, y_hat = _fit_linear_ols(x=x, y=y)
        fit_type = "linear"

    linear_r2 = _r2_from_y_hat(y=y, y_hat=y_hat)
    linear_slope = float(coef[1]) if coef.size >= 2 else float("nan")
    best_row = {
        "fit_type": str(fit_type),
        "xb": float("nan"),
        "L_break": float("nan"),
        "b1": float(linear_slope),
        "b2": float(linear_slope),
        "r2": float(linear_r2),
        "n_points": int(x.size),
    }
    return best_row, coef


@dataclass
class CensoredRegResult:
    beta: np.ndarray
    sigma: float
    feat_means: np.ndarray
    feat_stds: np.ndarray


def fit_censored_normal_regression(
    X_feat: np.ndarray,
    y_log: np.ndarray,
    censored: np.ndarray,
    censor_log: np.ndarray,
) -> CensoredRegResult:
    """
    Fit y = Xb + N(0, sigma^2) with right-censoring:
      censored=True means y > censor_log.
    Inputs:
      X_feat: (n, p) raw features (no intercept). Will be standardized internally.
      y_log: (n,) log(L0_hat) for uncensored rows; value can be anything for censored rows (ignored).
      censored: (n,) boolean.
      censor_log: (n,) log(Lmax) for censored rows; ignored for uncensored rows.

    Returns beta including intercept in standardized feature space.
    """
    n, p = X_feat.shape
    feat_means = X_feat.mean(axis=0)
    feat_stds = X_feat.std(axis=0)
    feat_stds = np.where(feat_stds < 1e-12, 1.0, feat_stds)

    Xz = (X_feat - feat_means) / feat_stds
    X = np.column_stack([np.ones(n), Xz])  # add intercept

    unc = ~censored
    X_u = X[unc]
    y_u = y_log[unc]

    # OLS init on uncensored
    if y_u.size >= X_u.shape[1]:
        beta_init, *_ = np.linalg.lstsq(X_u, y_u, rcond=None)
        resid = y_u - X_u @ beta_init
        sigma_init = float(np.std(resid)) if resid.size > 1 else 0.5
    else:
        beta_init = np.zeros(X.shape[1], dtype=float)
        sigma_init = 0.5

    sigma_init = max(sigma_init, 1e-3)
    params_init = np.concatenate([beta_init, np.array([math.log(sigma_init)], dtype=float)])

    def nll(params: np.ndarray) -> float:
        beta = params[:-1]
        log_sigma = params[-1]
        sigma = math.exp(log_sigma) + 1e-12
        mu = X @ beta

        # Uncensored likelihood: normal pdf
        z_u = (y_log[unc] - mu[unc]) / sigma
        ll_u = -0.5 * (z_u ** 2) - log_sigma - 0.5 * math.log(2.0 * math.pi)

        # Censored likelihood: P(Y > c) = sf((c - mu)/sigma)
        if np.any(censored):
            z_c = (censor_log[censored] - mu[censored]) / sigma
            ll_c = norm.logsf(z_c)
        else:
            ll_c = np.array([], dtype=float)

        ll = np.sum(ll_u) + np.sum(ll_c)
        return -float(ll)

    res = minimize(nll, params_init, method="L-BFGS-B")

    beta_hat = res.x[:-1]
    sigma_hat = float(math.exp(res.x[-1]))

    return CensoredRegResult(beta=beta_hat, sigma=sigma_hat, feat_means=feat_means, feat_stds=feat_stds)


def predict_log_l0_from_censored_reg(model: CensoredRegResult, X_feat: np.ndarray) -> np.ndarray:
    Xz = (X_feat - model.feat_means) / model.feat_stds
    X = np.column_stack([np.ones(X_feat.shape[0]), Xz])
    return X @ model.beta


def fit_global_theta_phi_and_group_cg(
    df: pd.DataFrame,
    l0_map: Dict[str, float],
    delta_right: float = DELTA_RIGHT,
    eps_clip: float = EPS_CLIP,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Fit:
      E = 1 - exp( - C(style) * (L - L0)_+^theta(R) * R^{-phi} )
      theta(R) = a + b * log R
    Using transform:
      y = log( -log(1 - E) ) = log C(style) + (a + b*log R)*log(L-L0) - phi*log R
    Solve by least squares with style-specific intercepts (log C(style)).
    """
    df2 = df.copy()
    df2["group_id"] = df2.apply(
        lambda r: make_group_id(r["resolution"], r["font_size"], r["line_spacing"], r["letter_spacing"]), axis=1
    )
    df2["style_id"] = df2.apply(
        lambda r: make_style_id(r["font_size"], r["line_spacing"], r["letter_spacing"]), axis=1
    )
    df2["L0"] = df2["group_id"].map(l0_map).astype(float)
    df2["excess"] = df2["text_len"].astype(float) - df2["L0"]
    df_fit = df2[df2["excess"] > float(delta_right)].copy()

    if df_fit.empty:
        raise RuntimeError("No points found on the right side for fitting. Try lowering DELTA_RIGHT.")

    E = df_fit["edit_distance"].astype(float).to_numpy()
    E = np.clip(E, float(eps_clip), 1.0 - 1e-6)
    y = np.log(-np.log(1.0 - E))

    x1 = np.log(df_fit["excess"].astype(float).to_numpy())
    x2 = np.log(df_fit["resolution"].astype(float).to_numpy())
    x3 = x1 * x2

    style_ids = df_fit["style_id"].astype(str).to_numpy()
    uniq_styles = np.unique(style_ids)
    s_index = {s: i for i, s in enumerate(uniq_styles)}
    m = uniq_styles.size

    # Design matrix: [x1, x1*x2, x2, style_onehots(all styles)]
    n = df_fit.shape[0]
    S = np.zeros((n, m), dtype=float)
    for i, s in enumerate(style_ids):
        S[i, s_index[s]] = 1.0

    X = np.column_stack([x1, x3, x2, S])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta_a = float(coef[0])
    theta_b = float(coef[1])
    phi = -float(coef[2])

    a_style = coef[3:]  # log C(style)
    cg_map = {s: float(math.exp(a_style[s_index[s]])) for s in uniq_styles}

    # For styles without fitted C, use median
    if len(cg_map) > 0:
        cg_median = float(np.median(list(cg_map.values())))
    else:
        cg_median = 1.0

    for sid in df2["style_id"].astype(str).unique():
        if sid not in cg_map:
            cg_map[sid] = cg_median

    return theta_a, theta_b, phi, cg_map


def predict_E(
    L: np.ndarray,
    resolution: float,
    L0: float,
    Cg: float,
    theta_a: float,
    theta_b: float,
    phi: float,
) -> np.ndarray:
    L = L.astype(float)
    excess = np.maximum(L - float(L0), 0.0)
    log_r = math.log(float(resolution))
    theta = float(theta_a) + float(theta_b) * log_r
    S = float(Cg) * np.power(excess, float(theta)) * np.power(float(resolution), -float(phi))
    E_hat = 1.0 - np.exp(-S)
    return np.clip(E_hat, 0.0, 1.0)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_powerlaw_diagnostics(
    df: pd.DataFrame,
    l0_map: Dict[str, float],
    theta_a: float,
    theta_b: float,
    phi: float,
    cg_map: Dict[str, float],
    delta_right: float = DELTA_RIGHT,
    eps_clip: float = EPS_CLIP,
    min_group_points: int = MIN_GROUP_POINTS_DIAG,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Quick, non-visual diagnostics to judge if the power-law model is reasonable.
    Assumes theta(R) = theta_a + theta_b * log R and C(style).
    Metrics are reported both in:
      - transformed space y = log(-log(1-E))
      - original space E
    """
    df2 = df.copy()
    df2["group_id"] = df2.apply(
        lambda r: make_group_id(r["resolution"], r["font_size"], r["line_spacing"], r["letter_spacing"]), axis=1
    )
    df2["style_id"] = df2.apply(
        lambda r: make_style_id(r["font_size"], r["line_spacing"], r["letter_spacing"]), axis=1
    )
    df2["L0"] = df2["group_id"].map(l0_map).astype(float)
    df2["excess"] = df2["text_len"].astype(float) - df2["L0"]

    df_fit = df2[df2["excess"] > float(delta_right)].copy()
    if df_fit.empty:
        raise RuntimeError("No points found on the right side for diagnostics. Try lowering DELTA_RIGHT.")

    cg_values = np.asarray(list(cg_map.values()), dtype=float)
    cg_fallback = float(np.median(cg_values)) if cg_values.size > 0 else 1.0
    df_fit["Cg"] = df_fit["style_id"].map(cg_map).fillna(cg_fallback).astype(float)

    L = df_fit["text_len"].astype(float).to_numpy()
    R = df_fit["resolution"].astype(float).to_numpy()
    excess = df_fit["excess"].astype(float).to_numpy()
    Cg = df_fit["Cg"].astype(float).to_numpy()

    E_raw = df_fit["edit_distance"].astype(float).to_numpy()
    E = np.clip(E_raw, float(eps_clip), 1.0 - 1e-6)

    theta = float(theta_a) + float(theta_b) * np.log(R)
    S = Cg * np.power(excess, theta) * np.power(R, -float(phi))
    E_hat = np.clip(1.0 - np.exp(-S), 0.0, 1.0)

    err_e = E - E_hat
    mae_e = float(np.mean(np.abs(err_e)))
    rmse_e = float(np.sqrt(np.mean(err_e ** 2)))
    r2_e = _safe_r2(E, E_hat)

    y_obs = np.log(-np.log(1.0 - E))
    y_pred = np.log(np.maximum(Cg, 1e-12)) + theta * np.log(excess) - float(phi) * np.log(R)
    resid_y = y_obs - y_pred

    mae_y = float(np.mean(np.abs(resid_y)))
    rmse_y = float(np.sqrt(np.mean(resid_y ** 2)))
    r2_y = _safe_r2(y_obs, y_pred)

    log_excess = np.log(excess)
    log_r = np.log(R)

    metrics = {
        "n_points_fit": int(df_fit.shape[0]),
        "n_groups_total": int(df2["group_id"].nunique()),
        "n_groups_fit": int(df_fit["group_id"].nunique()),
        "theta_a": float(theta_a),
        "theta_b": float(theta_b),
        "phi": float(phi),
        "mae_E": mae_e,
        "rmse_E": rmse_e,
        "r2_E": float(r2_e),
        "mae_y": mae_y,
        "rmse_y": rmse_y,
        "r2_y": float(r2_y),
        "corr_resid_y_log_excess": _safe_corr(resid_y, log_excess),
        "corr_resid_y_log_R": _safe_corr(resid_y, log_r),
        "corr_resid_y_log_excess_log_R": _safe_corr(resid_y, log_excess * log_r),
        "frac_E_le_eps_clip": float(np.mean(E_raw <= float(eps_clip))),
        "frac_E_ge_1_minus_1e6": float(np.mean(E_raw >= 1.0 - 1e-6)),
    }

    # By resolution summary in E-space (easy to interpret)
    by_res_rows = []
    for res, sub in df_fit.groupby("resolution"):
        mask = df_fit["resolution"].to_numpy() == float(res)
        e_sub = E[mask]
        ehat_sub = E_hat[mask]
        err_sub = e_sub - ehat_sub
        by_res_rows.append(
            {
                "resolution": int(res),
                "n_points": int(sub.shape[0]),
                "mae_E": float(np.mean(np.abs(err_sub))),
                "rmse_E": float(np.sqrt(np.mean(err_sub ** 2))),
                "r2_E": float(_safe_r2(e_sub, ehat_sub)),
            }
        )
    df_by_resolution = pd.DataFrame(by_res_rows).sort_values("resolution").reset_index(drop=True)

    # Per-group linearity check in transformed space: y_obs ~ a + b*log(excess)
    group_rows = []
    for gid, sub in df_fit.groupby("group_id"):
        if sub.shape[0] < int(min_group_points):
            continue
        e_sub_raw = sub["edit_distance"].astype(float).to_numpy()
        e_sub = np.clip(e_sub_raw, float(eps_clip), 1.0 - 1e-6)
        y_sub = np.log(-np.log(1.0 - e_sub))
        x_sub = np.log(sub["excess"].astype(float).to_numpy())
        res_sub = float(sub["resolution"].iloc[0])
        theta_sub = float(theta_a) + float(theta_b) * math.log(res_sub)

        X = np.column_stack([np.ones(x_sub.size), x_sub])
        coef, *_ = np.linalg.lstsq(X, y_sub, rcond=None)
        y_hat = X @ coef
        group_rows.append(
            {
                "group_id": str(gid),
                "n_points": int(sub.shape[0]),
                "slope_hat": float(coef[1]),
                "theta_pred": float(theta_sub),
                "slope_error": float(coef[1] - theta_sub),
                "r2_y_vs_log_excess": float(_safe_r2(y_sub, y_hat)),
            }
        )
    df_by_group = pd.DataFrame(group_rows)

    if not df_by_group.empty:
        slope_err = np.abs(df_by_group["slope_error"].astype(float).to_numpy())
        metrics.update(
            {
                "n_groups_diag": int(df_by_group.shape[0]),
                "median_group_r2_y_vs_log_excess": float(df_by_group["r2_y_vs_log_excess"].median()),
                "p10_group_r2_y_vs_log_excess": float(df_by_group["r2_y_vs_log_excess"].quantile(0.10)),
                "p90_group_r2_y_vs_log_excess": float(df_by_group["r2_y_vs_log_excess"].quantile(0.90)),
                "median_abs_group_slope_error": float(np.median(slope_err)),
                "frac_groups_r2_gt_0_8": float(np.mean(df_by_group["r2_y_vs_log_excess"] > 0.8)),
            }
        )
    else:
        metrics.update(
            {
                "n_groups_diag": 0,
                "median_group_r2_y_vs_log_excess": float("nan"),
                "p10_group_r2_y_vs_log_excess": float("nan"),
                "p90_group_r2_y_vs_log_excess": float("nan"),
                "median_abs_group_slope_error": float("nan"),
                "frac_groups_r2_gt_0_8": float("nan"),
            }
        )

    return metrics, df_by_resolution, df_by_group


def summarize_powerlaw_reasonableness(metrics: dict) -> Tuple[str, list]:
    """
    Heuristic summary for quick decision-making (not a formal statistical test).
    Returns (verdict, reasons).
    """
    reasons = []
    r2_y = float(metrics.get("r2_y", float("nan")))
    med_r2_group = float(metrics.get("median_group_r2_y_vs_log_excess", float("nan")))
    corr_ex = float(metrics.get("corr_resid_y_log_excess", float("nan")))
    corr_r = float(metrics.get("corr_resid_y_log_R", float("nan")))

    if not np.isfinite(r2_y) or r2_y < 0.5:
        reasons.append(f"low transformed-space R2 (r2_y={r2_y:.3f})")
    if np.isfinite(med_r2_group) and med_r2_group < 0.6:
        reasons.append(f"low median per-group linearity (median_r2_group={med_r2_group:.3f})")
    if np.isfinite(corr_ex) and abs(corr_ex) > 0.2:
        reasons.append(f"residual correlates with log(excess) (corr={corr_ex:.3f})")
    if np.isfinite(corr_r) and abs(corr_r) > 0.2:
        reasons.append(f"residual correlates with log(R) (corr={corr_r:.3f})")

    if len(reasons) == 0:
        return "Power-law form looks broadly reasonable (in transformed space).", reasons
    if np.isfinite(r2_y) and r2_y >= 0.5:
        return "Mixed evidence for the power-law form; may need extensions (better L0, varying exponents, or different scaling).", reasons
    return "Power-law form likely inadequate, or L0/measurement issues dominate the fit.", reasons


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    set_seed(RANDOM_SEED)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(CSV_PATH)
    required_cols = [
        "font_size", "line_spacing", "letter_spacing",
        "text_len", "resolution", "edit_distance"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Normalize dtypes
    df["font_size"] = df["font_size"].astype(float)
    df["line_spacing"] = df["line_spacing"].astype(float)
    df["letter_spacing"] = df["letter_spacing"].astype(float)
    df["text_len"] = df["text_len"].astype(float)
    df["resolution"] = df["resolution"].astype(int)
    df["edit_distance"] = df["edit_distance"].astype(float)

    # Build group_id
    df["group_id"] = df.apply(
        lambda r: make_group_id(r["resolution"], r["font_size"], r["line_spacing"], r["letter_spacing"]), axis=1
    )

    # -------------------------
    # Step 1: Choose L0 per group
    # -------------------------
    group_rows = []
    for (res, fs, ls, cs), gdf in df.groupby(["resolution", "font_size", "line_spacing", "letter_spacing"]):
        k_consecutive = 5 if int(res) in (1024, 1280) else K_CONSECUTIVE
        L0_hat, is_censored, Lmax = estimate_threshold_local(
            text_len=gdf["text_len"].to_numpy(),
            edit_distance=gdf["edit_distance"].to_numpy(),
            bin_width=BIN_WIDTH,
            q=Q_FOR_THRESHOLD,
            eps=EPS_THRESHOLD,
            k_consecutive=k_consecutive,
        )
        L0_hat = 0.0
        group_rows.append(
            {
                "resolution": int(res),
                "font_size": float(fs),
                "line_spacing": float(ls),
                "letter_spacing": float(cs),
                "group_id": make_group_id(res, fs, ls, cs),
                "L0_hat": float(L0_hat),
                "censored": bool(is_censored),
                "Lmax": float(Lmax),
            }
        )

    df_groups = pd.DataFrame(group_rows)

    l0_hat_map = {row["group_id"]: float(row["L0_hat"]) for _, row in df_groups.iterrows()}

    ensure_dir(BIN_PLOT_DIR)
    for res in sorted(df["resolution"].unique()):
        df_r = df[df["resolution"] == res].copy()

        fs_list = sorted(df_r["font_size"].unique())
        ls_list = sorted(df_r["line_spacing"].unique())
        cs_list = sorted(df_r["letter_spacing"].unique())
        x_min = 0.0
        x_max = float(df_r["text_len"].max()) if not df_r.empty else 1.0

        cs_list = sorted(cs_list)

        pairs = []
        i = 0
        while i < len(ls_list):
            ls_a = ls_list[i]
            ls_b = ls_list[i + 1] if (i + 1) < len(ls_list) else None
            pairs.append((ls_a, ls_b))
            i += 2

        ncols = max(1, 2 * len(cs_list))
        nrows = len(fs_list) * len(pairs)
        fig_w = max(6, 3.7 * ncols)
        fig_h = max(6, 3.0 * nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(f"resolution={res} | q={Q_FOR_THRESHOLD:g} | bin_width={BIN_WIDTH}", fontsize=14)

        row_idx = 0
        for fs in fs_list:
            for (ls_a, ls_b) in pairs:
                for j, ls_val in enumerate([ls_a, ls_b]):
                    for k, cs in enumerate(cs_list):
                        col_idx = j * len(cs_list) + k
                        ax = axes[row_idx, col_idx]

                        if ls_val is None:
                            ax.axis("off")
                            continue

                        gid = make_group_id(res, fs, ls_val, cs)
                        gdf = df_r[
                            (df_r["font_size"] == fs) &
                            (df_r["line_spacing"] == ls_val) &
                            (df_r["letter_spacing"] == cs)
                        ]

                        if gdf.empty:
                            ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g}")
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(0.0, 1.0)
                            ax.grid(True, alpha=0.2)
                            continue

                        left_edges, q_values = compute_bin_quantiles(
                            text_len=gdf["text_len"].to_numpy(),
                            edit_distance=gdf["edit_distance"].to_numpy(),
                            bin_width=BIN_WIDTH,
                            q=Q_FOR_THRESHOLD,
                        )

                        if left_edges.size > 0:
                            ax.plot(left_edges, q_values, marker="o", markersize=3, linewidth=1)

                        L0_hat = l0_hat_map.get(gid)
                        if L0_hat is not None:
                            ax.axvline(float(L0_hat), linestyle="--", color="red", linewidth=1)

                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g}")
                        ax.set_xlabel("text_len")
                        ax.set_ylabel(f"q={Q_FOR_THRESHOLD:g}")
                        ax.set_ylim(0.0, 1.0)
                        ax.set_xlim(x_min, x_max)
                        ax.grid(True, alpha=0.2)

                row_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(BIN_PLOT_DIR, f"bins_resolution_{res}.png")
        plt.savefig(out_path, dpi=180)
        plt.close(fig)

    # -------------------------
    # Step 2: Piecewise linear fit + visualization in log-log space
    # Only plot points with L > L0.
    # -------------------------
    ensure_dir(LAW_RELATION_DIR)
    piecewise_rows = []
    piecewise_coef_map = {}
    for (res, fs, ls, cs), gdf in df.groupby(["resolution", "font_size", "line_spacing", "letter_spacing"]):
        gid = make_group_id(res, fs, ls, cs)
        l0 = l0_hat_map.get(gid)
        if l0 is None:
            continue
        sparse_trim_frac = SPARSE_TRIM_FRACTION_BY_RES.get(int(res), 0.0)
        use_ransac = int(res) == 1280
        min_log_l = MIN_LOG_L_BY_RES.get(int(res))
        best_row, best_coef = fit_piecewise_by_quantile_grid(
            text_len=gdf["text_len"].to_numpy(),
            edit_distance=gdf["edit_distance"].to_numpy(),
            l0=float(l0),
            sparse_trim_frac=float(sparse_trim_frac),
            use_ransac=bool(use_ransac),
            min_log_l=min_log_l,
            q_start=BREAK_Q_START,
            q_end=BREAK_Q_END,
            q_step=BREAK_Q_STEP,
        )
        piecewise_rows.append(
            {
                "group_id": gid,
                "resolution": int(res),
                "font_size": float(fs),
                "line_spacing": float(ls),
                "letter_spacing": float(cs),
                "L0": float(l0),
                "fit_type": str(best_row.get("fit_type", "piecewise")),
                "xb": float(best_row["xb"]),
                "L_break": float(best_row["L_break"]),
                "b1": float(best_row["b1"]),
                "b2": float(best_row["b2"]),
                "r2": float(best_row["r2"]),
            }
        )
        fit_type = str(best_row.get("fit_type", "linear"))
        if fit_type in ("linear", "ransac") and best_coef.size == 2:
            piecewise_coef_map[gid] = {"fit_type": fit_type, "coef": best_coef.astype(float)}

    piecewise_path = os.path.join(LAW_RELATION_DIR, "piecewise_fit_params.csv")
    df_piecewise = pd.DataFrame(piecewise_rows)
    if not df_piecewise.empty:
        df_piecewise = df_piecewise.sort_values(
            by=["resolution", "font_size", "line_spacing", "letter_spacing"],
            ascending=True,
            kind="mergesort",
        )
    df_piecewise.to_csv(piecewise_path, index=False)

    for res in sorted(df["resolution"].unique()):
        df_r = df[df["resolution"] == res].copy()

        fs_list = sorted(df_r["font_size"].unique())
        ls_list = sorted(df_r["line_spacing"].unique())
        cs_list = sorted(df_r["letter_spacing"].unique())

        cs_list = sorted(cs_list)

        pairs = []
        i = 0
        while i < len(ls_list):
            ls_a = ls_list[i]
            ls_b = ls_list[i + 1] if (i + 1) < len(ls_list) else None
            pairs.append((ls_a, ls_b))
            i += 2

        ncols = max(1, 2 * len(cs_list))
        nrows = len(fs_list) * len(pairs)
        fig_w = max(6, 3.7 * ncols)
        fig_h = max(6, 3.0 * nrows)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.suptitle(f"resolution={res} | y=log(E) vs x=log(L) (L>L0 only)", fontsize=14)

        row_idx = 0
        for fs in fs_list:
            for (ls_a, ls_b) in pairs:
                for j, ls_val in enumerate([ls_a, ls_b]):
                    for k, cs in enumerate(cs_list):
                        col_idx = j * len(cs_list) + k
                        ax = axes[row_idx, col_idx]

                        if ls_val is None:
                            ax.axis("off")
                            continue

                        gid = make_group_id(res, fs, ls_val, cs)
                        L0 = l0_hat_map.get(gid)
                        if L0 is None:
                            ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g}")
                            ax.set_xlim(0, 1)
                            ax.grid(True, alpha=0.2)
                            continue

                        gdf = df_r[
                            (df_r["font_size"] == fs) &
                            (df_r["line_spacing"] == ls_val) &
                            (df_r["letter_spacing"] == cs)
                        ]

                        if not gdf.empty:
                            L = gdf["text_len"].astype(float).to_numpy()
                            E = gdf["edit_distance"].astype(float).to_numpy()
                            keep = (L > float(L0)) & (E > 0)
                            if np.any(keep):
                                L_keep = L[keep]
                                E_keep = E[keep]
                                log_e_keep = np.log(E_keep)
                                keep_log_e = log_e_keep >= float(LOG_E_MIN)
                                if np.any(keep_log_e):
                                    L_keep = L_keep[keep_log_e]
                                    E_keep = E_keep[keep_log_e]
                                    x = np.log(L_keep)
                                    y = np.log(E_keep)
                                    sparse_trim_frac = SPARSE_TRIM_FRACTION_BY_RES.get(int(res), 0.0)
                                    keep_sparse = _sparse_trim_mask(x, y, K_NEAREST, sparse_trim_frac)
                                    x = x[keep_sparse]
                                    y = y[keep_sparse]
                                    ax.scatter(x, y, s=8, alpha=0.55)
                                    pw = piecewise_coef_map.get(gid)
                                    if pw is not None and x.size >= 2:
                                        fit_type = str(pw.get("fit_type", "piecewise"))
                                        coef = np.asarray(pw.get("coef", []), dtype=float)
                                        x_min = float(np.min(x))
                                        x_max = float(np.max(x))
                                        if fit_type == "piecewise" and coef.size == 3:
                                            xb = float(pw["xb"])
                                            x_left = np.linspace(x_min, min(xb, x_max), 80)
                                            x_right = np.linspace(max(xb, x_min), x_max, 80)
                                            y_left = float(coef[0]) + float(coef[1]) * x_left
                                            y_right = float(coef[0]) + float(coef[1]) * x_right + float(coef[2]) * np.maximum(
                                                x_right - float(xb), 0.0
                                            )
                                            ax.plot(x_left, y_left, color="red", linewidth=2)
                                            ax.plot(x_right, y_right, color="red", linewidth=2)
                                            ax.axvline(float(xb), color="red", linestyle="--", linewidth=1)
                                        elif fit_type in ("linear", "ransac") and coef.size == 2:
                                            x_line = np.linspace(x_min, x_max, 160)
                                            y_line = float(coef[0]) + float(coef[1]) * x_line
                                            ax.plot(x_line, y_line, color="red", linewidth=2)

                        ax.set_title(f"fs={fs:g}, ls={ls_val:g}, cs={cs:g}")
                        ax.set_xlabel("log(L)")
                        ax.set_ylabel("log(E)")
                        ax.grid(True, alpha=0.2)

                row_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(LAW_RELATION_DIR, f"law_relation_resolution_{res}.png")
        plt.savefig(out_path, dpi=180)
        plt.close(fig)

    print("Done (fit+diagnostics temporarily disabled).")
    print(f"- piecewise fit saved to: {piecewise_path}")
    print(f"- law_relation plots saved to: {LAW_RELATION_DIR}")


if __name__ == "__main__":
    main()

"""Fit w0 for new model data with fixed mixture parameters."""

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.optimize import minimize
from scipy.special import expit, gammaln

DEFAULT_JSON_PATH_A = Path("eval_results/pride_square/Qwen/Qwen2.5-VL-7B-Instruct-560/dpsk_eval_metric.json")
DEFAULT_JSON_PATH_B = Path("eval_results/pride_square/OpenGVLab/InternVL3_5-8B-896/dpsk_eval_metric.json")
DEFAULT_SUMMARY_PATH = Path("data_analysis/em_fiting_res/shared_exceptw/640/em_fit_summary.json")
DEFAULT_OUTPUT_PATH = Path("eval_results/other_vlm.pdf")
DEFAULT_TITLE_A = "Qwen2.5-VL-7B"
DEFAULT_TITLE_B = "InternVL3.5-8B"
DEFAULT_MAX_ITER = 200
DEFAULT_EPS = 1e-6
DEFAULT_FONT_SIZE = 28
DEFAULT_LINE_SPACING = 6
DEFAULT_LETTER_SPACING = 0
DEFAULT_CHARACTER_WIDTH = 14.0533
DEFAULT_CHARACTER_HEIGHT = 26.0
DEFAULT_VISION_TOKEN_NUM_A = 324.0
DEFAULT_VISION_TOKEN_NUM_B = 1280.0
FIG_WIDTH = 3.03
FIG_HEIGHT = 1.8
FONT_SIZE = 8
POINT_COLOR = "#86CAD2"
CURVE_COLOR = "#D62728"

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def log1m_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, z)


def beta_logpdf(log_y: np.ndarray, log1m_y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    return (alpha - 1.0) * log_y + (beta - 1.0) * log1m_y + norm


def load_metrics(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected list of metric entries in JSON.")
    text_len: list[float] = []
    edit_dist: list[float] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        t_len = entry.get("text_len")
        e_dist = entry.get("edit_dist")
        if t_len is None or e_dist is None:
            continue
        try:
            text_len.append(float(t_len))
            edit_dist.append(float(e_dist))
        except (TypeError, ValueError):
            continue
    if not text_len:
        raise ValueError("No valid text_len/edit_dist values found in JSON.")
    return np.asarray(text_len, dtype=float), np.asarray(edit_dist, dtype=float)


def format_k_ticks(value: float, _pos: int) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        scaled = value / 1000
        if isinstance(scaled, float) and scaled.is_integer():
            return f"{int(scaled)}k"
        return f"{scaled:g}k"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def resolve_summary_path(summary_path: Path, vision_token_num: float, logger: logging.Logger) -> Path:
    path = summary_path
    if path.is_dir():
        path = path / "em_fit_summary.json"
    if path.exists():
        return path
    resolution_guess = int(round(64.0 * math.sqrt(float(vision_token_num))))
    candidate = Path("data_analysis/em_fiting_res/shared_exceptw") / str(resolution_guess) / "em_fit_summary.json"
    if candidate.exists():
        logger.info("Summary file not found; falling back to %s", candidate)
        return candidate
    raise FileNotFoundError(f"Summary file not found: {path}")


def load_summary_params(summary_path: Path) -> tuple[float, float, float, float, float, float]:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    gating_params = summary.get("gating_params", {})
    beta_params = summary.get("beta_params", {})
    low_params = beta_params.get("low_ed", {})
    high_params = beta_params.get("high_ed", {})
    try:
        a_val = float(gating_params["a"])
        alpha_val = float(gating_params["alpha"])
        alpha0 = float(low_params["alpha"])
        beta0 = float(low_params["beta"])
        alpha1 = float(high_params["alpha"])
        beta1 = float(high_params["beta"])
    except KeyError as exc:
        raise KeyError(f"Missing parameter in summary JSON: {exc}") from exc
    return a_val, alpha_val, alpha0, beta0, alpha1, beta1


def fit_w0(
    log_v: np.ndarray,
    log_g: np.ndarray,
    y: np.ndarray,
    a_val: float,
    alpha_val: float,
    alpha0: float,
    beta0: float,
    alpha1: float,
    beta1: float,
    w0_init: float,
    max_iter: int,
) -> tuple[float, bool]:
    log_y = np.log(y)
    log1m_y = np.log1p(-y)

    def nll(params: np.ndarray) -> float:
        w0 = float(params[0])
        z = w0 + a_val * log_v + alpha_val * log_g
        log_pi = log_sigmoid(z)
        log_p1 = beta_logpdf(log_y, log1m_y, alpha1, beta1)
        log_p0 = beta_logpdf(log_y, log1m_y, alpha0, beta0)
        log_mix = np.logaddexp(log_pi + log_p1, log1m_sigmoid(z) + log_p0)
        return -float(np.sum(log_mix))

    res = minimize(
        nll,
        np.array([w0_init], dtype=float),
        method="L-BFGS-B",
        options={"maxiter": int(max_iter)},
    )
    if not res.success or not np.isfinite(res.x[0]):
        return float(w0_init), False
    return float(res.x[0]), True


def compute_expected_curve(
    input_json: Path,
    summary_json: Path,
    vision_token_num: float,
    args: argparse.Namespace,
    logger: logging.Logger,
    title: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_len, edit_dist = load_metrics(input_json)
    edit_dist = np.clip(edit_dist.astype(float), float(args.eps), 1.0 - float(args.eps))

    denominator = float(args.character_height) + float(args.line_spacing)
    if denominator == 0.0:
        raise ValueError("character_height + line_spacing must be non-zero.")
    v_vals = (float(args.character_width) + float(args.letter_spacing)) * text_len / denominator
    g_vals = text_len / float(vision_token_num)
    v_vals = np.maximum(v_vals, float(args.eps))
    g_vals = np.maximum(g_vals, float(args.eps))
    log_v = np.log(v_vals)
    log_g = np.log(g_vals)

    summary_path = resolve_summary_path(summary_json, float(vision_token_num), logger)
    a_val, alpha_val, alpha0, beta0, alpha1, beta1 = load_summary_params(summary_path)

    w0, converged = fit_w0(
        log_v,
        log_g,
        edit_dist,
        a_val,
        alpha_val,
        alpha0,
        beta0,
        alpha1,
        beta1,
        float(args.init_w0),
        int(args.max_iter),
    )
    logger.info("Fit w0 for %s = %.6f (converged=%s)", title, w0, converged)

    mean0 = alpha0 / (alpha0 + beta0)
    mean1 = alpha1 / (alpha1 + beta1)
    pi = expit(w0 + a_val * log_v + alpha_val * log_g)
    expected = (1.0 - pi) * mean0 + pi * mean1
    return text_len, edit_dist, expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit w0 for two models and plot scaling-law curves.")
    parser.add_argument("--input_json_a", type=Path, default=DEFAULT_JSON_PATH_A, help="Metrics JSON for the first model.")
    parser.add_argument("--input_json_b", type=Path, default=DEFAULT_JSON_PATH_B, help="Metrics JSON for the second model.")
    parser.add_argument("--title_a", type=str, default=DEFAULT_TITLE_A, help="Title for the first subplot.")
    parser.add_argument("--title_b", type=str, default=DEFAULT_TITLE_B, help="Title for the second subplot.")
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_PATH, help="Path to em_fit_summary.json with fixed params.")
    parser.add_argument("--output_plot", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to save the scatter/curve plot.")
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER, help="Maximum iterations for w0 fitting.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Clipping epsilon for edit_dist.")
    parser.add_argument("--font_size", type=float, default=DEFAULT_FONT_SIZE, help="Font size for computing V.")
    parser.add_argument("--line_spacing", type=float, default=DEFAULT_LINE_SPACING, help="Line spacing for computing V.")
    parser.add_argument("--letter_spacing", type=float, default=DEFAULT_LETTER_SPACING, help="Letter spacing for computing V.")
    parser.add_argument("--character_width", type=float, default=DEFAULT_CHARACTER_WIDTH, help="Character width for computing V.")
    parser.add_argument("--character_height", type=float, default=DEFAULT_CHARACTER_HEIGHT, help="Character height for computing V.")
    parser.add_argument("--vision_token_num_a", type=float, default=DEFAULT_VISION_TOKEN_NUM_A, help="Vision token number for the first model.")
    parser.add_argument("--vision_token_num_b", type=float, default=DEFAULT_VISION_TOKEN_NUM_B, help="Vision token number for the second model.")
    parser.add_argument("--init_w0", type=float, default=0.0, help="Initial w0 for optimization.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("fit_scaling_law_other_models")

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    configs = [
        {
            "title": args.title_a,
            "input_json": args.input_json_a,
            "vision_token_num": float(args.vision_token_num_a),
        },
        {
            "title": args.title_b,
            "input_json": args.input_json_b,
            "vision_token_num": float(args.vision_token_num_b),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.75, wspace=0.08)

    for idx, config in enumerate(configs):
        text_len, edit_dist, expected = compute_expected_curve(
            config["input_json"],
            args.summary_json,
            config["vision_token_num"],
            args,
            logger,
            config["title"],
        )
        order = np.argsort(text_len)

        ax = axes[idx]
        ax.scatter(text_len, edit_dist, s=1.5, alpha=0.6, color=POINT_COLOR, label="Data Point")
        ax.plot(text_len[order], expected[order], color=CURVE_COLOR, linewidth=1.2, label="E[ED|Z]")

        ax.xaxis.set_major_formatter(FuncFormatter(format_k_ticks))
        ax.tick_params(axis="x", length=2)
        if idx == 0:
            ax.text(
                -0.06,
                1.04,
                "ED",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                clip_on=False,
            )
            ax.tick_params(axis="y", length=2, pad=1)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        if idx == len(configs) - 1:
            ax.set_xlabel("TL")
            ax.xaxis.set_label_coords(0.96, -0.02)
        else:
            ax.set_xlabel("")

        ax.text(
            0.5,
            1.04,
            config["title"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            clip_on=False,
        )
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor=POINT_COLOR,
            markeredgecolor=POINT_COLOR,
            label="Data Point",
        ),
        Line2D(
            [0],
            [0],
            color=CURVE_COLOR,
            linewidth=1.2,
            label="E[ED|Z]",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    output_plot = args.output_plot
    if output_plot.suffix.lower() != ".pdf":
        output_plot = output_plot.with_suffix(".pdf")
    ensure_dir(output_plot)
    fig.savefig(output_plot)
    plt.close(fig)
    logger.info("Saved plot to %s", output_plot)


if __name__ == "__main__":
    main()

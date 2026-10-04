import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


NOMINAL_TO_EFFECTIVE = {512: 504, 640: 644, 1024: 1036, 1280: 1288}
FIG_WIDTH = 7.2
FIG_HEIGHT = 2.15
FONT_SIZE = 8
POINT_COLOR = "#1F77B4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen2.5-VL OCR edit distance for layout 28_6_0 at four resolutions.")
    parser.add_argument("--input_dir", type=Path, default=Path("VQAExp/eval_results/ocr/qwen25vl7b/28_6_0"), help="Directory containing the four OCR edit-distance JSONL files.")
    parser.add_argument("--output", type=Path, default=Path("aaai_resub/qwen_instability_intervention/figures/qwen_ocr_28_6_0_four_resolutions.jpg"), help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in dots per inch.")
    return parser.parse_args()


def load_points(path: Path, expected_resolution: int) -> tuple[list[int], list[float]]:
    text_lengths = []
    edit_distances = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["resolution"]) != expected_resolution:
                raise ValueError(f"Unexpected resolution in {path}:{line_number}")
            sample_id = str(record["id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id in {path}: {sample_id}")
            seen_ids.add(sample_id)
            text_lengths.append(int(record["text_len"]))
            edit_distances.append(float(record["edit_dist"]))
    if not text_lengths:
        raise ValueError(f"No points found in {path}")
    return text_lengths, edit_distances


def format_k_ticks(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    loaded = {}
    reference_ids = None
    for nominal_resolution in NOMINAL_TO_EFFECTIVE:
        input_path = args.input_dir / f"ocr_edit_distance_res_{nominal_resolution}.jsonl"
        text_lengths, edit_distances = load_points(input_path, nominal_resolution)
        ids = {json.loads(line)["id"] for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Sample IDs differ across resolutions: {input_path}")
        loaded[nominal_resolution] = (text_lengths, edit_distances)

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=args.dpi, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.982, bottom=0.25, top=0.82, wspace=0.08)

    for index, (axis, nominal_resolution) in enumerate(zip(axes, NOMINAL_TO_EFFECTIVE)):
        effective_resolution = NOMINAL_TO_EFFECTIVE[nominal_resolution]
        vision_tokens = (effective_resolution // 28) ** 2
        text_lengths, edit_distances = loaded[nominal_resolution]
        mean_edit_distance = sum(edit_distances) / len(edit_distances)
        axis.scatter(text_lengths, edit_distances, s=3, alpha=0.55, edgecolors="none", color=POINT_COLOR, rasterized=True)
        axis.set_xlim(0, 25000)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position, hide_zero=index > 0: "" if hide_zero and value == 0 else format_k_ticks(value, position)))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        axis.set_title(f"R={effective_resolution}, VT={vision_tokens}", pad=3)
        axis.text(0.97, 0.96, f"mean={mean_edit_distance:.3f}", transform=axis.transAxes, ha="right", va="top", fontsize=7)
        axis.tick_params(axis="both", length=2, pad=1)
        if index == 0:
            axis.set_ylabel("Normalized edit distance")
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)

    fig.supxlabel("Number of characters", x=0.53, y=0.04, fontsize=FONT_SIZE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.lstrip("."), dpi=args.dpi)
    plt.close(fig)
    logging.info("Saved four-panel plot with %d samples per panel to %s", len(reference_ids or []), args.output)


if __name__ == "__main__":
    main()

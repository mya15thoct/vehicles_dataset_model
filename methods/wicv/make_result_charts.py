#!/usr/bin/env python3
"""Charts for the two highest-value result findings: loss-weight sensitivity
(resolves the FCA framing question) and cross-condition generalization
(supports the weather-invariance claim in the title).

Reads the summary CSVs already produced by run_sensitivity.py and
run_cross_condition.py -- no retraining, just visualization. Style matches
conference/make_result_chart.py (flat white background, rounded bars).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (25, 31, 40)
MUTED = (91, 99, 112)
GRID = (222, 226, 232)
BG = (255, 255, 255)
ACCENT = (36, 95, 145)
LINE_COLOR = (50, 116, 184)
SELECTED_COLOR = (220, 132, 59)
REFERENCE_COLOR = (160, 166, 176)
CE_COLOR = (160, 166, 176)
WICV_COLOR = (50, 116, 184)

PROTOCOL_ORDER = ["norain2rain", "rain2norain", "morning2evening", "evening2morning"]
PROTOCOL_LABELS = {
    "norain2rain": "No-rain -> Rain",
    "rain2norain": "Rain -> No-rain",
    "morning2evening": "Morning -> Evening",
    "evening2morning": "Evening -> Morning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-csv", default="results/wicv_sensitivity/summary.csv")
    parser.add_argument("--cross-condition-csv", default="results/wicv_cross_condition/summary.csv")
    parser.add_argument("--ablation-csv", default="results/wicv_ablation/summary.csv")
    parser.add_argument("--selected-w-adv", type=float, default=0.1)
    parser.add_argument("--output-root", default="docs/figures")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, entries: list[tuple[str, tuple]], font) -> None:
    swatch = 16
    for label, color in entries:
        draw.rounded_rectangle((x, y, x + swatch, y + swatch), radius=4, fill=color)
        text_w, _ = text_size(draw, label, font)
        draw.text((x + swatch + 8, y - 1), label, fill=INK, font=font)
        x += swatch + 8 + text_w + 26


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    panel_w: int,
    panel_h: int,
    title: str,
    points: list[tuple[float, float]],
    selected_x: float | None,
    reference_y: float | None,
    reference_label: str,
    x_label_fmt: str = "{:.2f}",
) -> None:
    label_font = load_font(16, bold=True)
    small_font = load_font(13)
    value_font = load_font(12)

    draw.text((x0, y0), title, fill=INK, font=label_font)

    plot_left = x0 + 40
    plot_right = x0 + panel_w - 20
    plot_top = y0 + 40
    plot_bottom = y0 + panel_h - 40

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    x_span = (x_max - x_min) or 1.0
    y_min = min(ys + ([reference_y] if reference_y is not None else [])) - 3
    y_max = max(ys + ([reference_y] if reference_y is not None else [])) + 3

    def px(x: float) -> float:
        return plot_left + (plot_right - plot_left) * (x - x_min) / x_span

    def py(y: float) -> float:
        return plot_bottom - (plot_bottom - plot_top) * (y - y_min) / (y_max - y_min)

    # axes
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRID, width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRID, width=1)

    if reference_y is not None:
        ry = py(reference_y)
        draw.line((plot_left, ry, plot_right, ry), fill=REFERENCE_COLOR, width=1)
        draw.text((plot_right - 150, ry - 16), f"{reference_label} ({reference_y:.2f})", fill=MUTED, font=small_font)

    # line + markers
    coords = [(px(x), py(y)) for x, y in points]
    for i in range(len(coords) - 1):
        draw.line((coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]), fill=LINE_COLOR, width=2)
    for (x, y), (px_, py_) in zip(points, coords):
        is_selected = selected_x is not None and abs(x - selected_x) < 1e-9
        radius = 6 if is_selected else 4
        color = SELECTED_COLOR if is_selected else LINE_COLOR
        draw.ellipse((px_ - radius, py_ - radius, px_ + radius, py_ + radius), fill=color, outline=BG, width=2)
        label = f"{y:.1f}"
        draw.text((px_ - 12, py_ - radius - 18), label, fill=INK, font=value_font)
        draw.text((px_ - 14, plot_bottom + 6), x_label_fmt.format(x), fill=MUTED, font=small_font)

    if selected_x is not None:
        draw.text((plot_left, y0 + panel_h - 18), "orange marker = tuned/selected value", fill=MUTED, font=small_font)


def make_sensitivity_figure(sensitivity_rows: list[dict], no_adv_map: float | None, selected_w_adv: float, output_root: Path) -> Path:
    width = 1000
    height = 380
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(14)

    draw.text((24, 20), "Loss-weight sensitivity: resolving the adversarial-weight finding", fill=INK, font=title_font)
    draw.text(
        (24, 52),
        "Naive w_adv hurt performance; tuning it recovers and exceeds the no-adversarial ablation (dashed reference).",
        fill=MUTED,
        font=subtitle_font,
    )

    w_adv_points = sorted(
        ((float(row["value"]), float(row["mAP"]) * 100) for row in sensitivity_rows if row["param"] == "w_adv"),
        key=lambda p: p[0],
    )
    w_cvpa_points = sorted(
        ((float(row["value"]), float(row["mAP"]) * 100) for row in sensitivity_rows if row["param"] == "w_cvpa"),
        key=lambda p: p[0],
    )

    panel_w = (width - 24 * 3) // 2
    panel_h = height - 110
    draw_line_panel(
        draw, 24, 90, panel_w, panel_h,
        "w_adv (mAP %, w_cvpa fixed at default)",
        w_adv_points,
        selected_x=selected_w_adv,
        reference_y=no_adv_map,
        reference_label="no_adv ablation",
    )
    draw_line_panel(
        draw, 24 * 2 + panel_w, 90, panel_w, panel_h,
        "w_cvpa (mAP %, w_adv fixed at default)",
        w_cvpa_points,
        selected_x=None,
        reference_y=None,
        reference_label="",
    )

    path = output_root / "figure_07_sensitivity.png"
    save_image(image, path)
    print(f"Saved {path}")
    return path


def make_cross_condition_figure(cc_rows: list[dict], output_root: Path) -> Path:
    width = 980
    height = 450
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(14)
    label_font = load_font(15, bold=True)
    value_font = load_font(13)

    draw.text((24, 20), "Cross-condition generalization: WICV-Net vs. CE baseline", fill=INK, font=title_font)
    draw.text(
        (24, 52),
        "Train on one domain value, test on the unseen other value. WICV-Net wins all four protocols.",
        fill=MUTED,
        font=subtitle_font,
    )
    draw_legend(draw, 24, 78, [("CE baseline (ce_only)", CE_COLOR), ("WICV-Net (full)", WICV_COLOR)], value_font)

    by_protocol: dict[str, dict[str, float]] = {}
    for row in cc_rows:
        by_protocol.setdefault(row["protocol"], {})[row["variant"]] = float(row["mAP"]) * 100

    chart_left = 220
    chart_right = width - 90
    chart_top = 130
    chart_bottom = height - 30
    max_val = max((v for d in by_protocol.values() for v in d.values()), default=1.0) * 1.15
    row_h = (chart_bottom - chart_top) / len(PROTOCOL_ORDER)
    bar_h = row_h * 0.32
    bar_gap = row_h * 0.06

    for idx, protocol in enumerate(PROTOCOL_ORDER):
        values = by_protocol.get(protocol, {})
        group_y = chart_top + idx * row_h
        draw.text((24, group_y + row_h / 2 - 9), PROTOCOL_LABELS[protocol], fill=INK, font=label_font)
        for offset, (variant, color) in enumerate([("ce_only", CE_COLOR), ("full", WICV_COLOR)]):
            value = values.get(variant, 0.0)
            bar_y = group_y + bar_gap + offset * (bar_h + bar_gap)
            bar_w = (chart_right - chart_left) * value / max_val if max_val else 0
            draw.rounded_rectangle((chart_left, bar_y, chart_left + bar_w, bar_y + bar_h), radius=6, fill=color)
            draw.text((chart_left + bar_w + 10, bar_y + bar_h / 2 - 7), f"{value:.1f}", fill=INK, font=value_font)

    path = output_root / "figure_08_cross_condition.png"
    save_image(image, path)
    print(f"Saved {path}")
    return path


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sensitivity_path = Path(args.sensitivity_csv)
    if sensitivity_path.exists():
        sensitivity_rows = read_csv(sensitivity_path)
        no_adv_map = None
        ablation_path = Path(args.ablation_csv)
        if ablation_path.exists():
            for row in read_csv(ablation_path):
                if row.get("variant") == "no_adv":
                    no_adv_map = float(row["mAP"]) * 100
                    break
        make_sensitivity_figure(sensitivity_rows, no_adv_map, args.selected_w_adv, output_root)
    else:
        print(f"Skipping sensitivity figure: {sensitivity_path} not found")

    cc_path = Path(args.cross_condition_csv)
    if cc_path.exists():
        make_cross_condition_figure(read_csv(cc_path), output_root)
    else:
        print(f"Skipping cross-condition figure: {cc_path} not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

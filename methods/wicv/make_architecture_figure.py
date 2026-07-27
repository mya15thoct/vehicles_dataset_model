#!/usr/bin/env python3
"""Render the WICV-Net architecture diagram for the method section.

Two panels, because the point the figure has to make is that one component
survives past training:

  (a) Training  -- shared backbone, condition-adaptive neck, and where each of
      the four losses attaches. CV-Tri and CVPA act on the raw backbone
      feature; the identity head, the adversarial heads, and CVT act on the
      neck output, which is the space retrieval actually runs in.
  (b) Inference -- the gallery is mapped through the learned before->after
      transition so matching happens inside a single view subspace. Nothing
      else from training is present here.

Pure PIL so it needs no LaTeX toolchain; SCALE keeps it print-resolution.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCALE = 2  # render at 2x for print, then keep the large canvas

BG = (255, 255, 255)
INK = (25, 31, 40)
MUTED = (91, 99, 112)
GRID = (208, 214, 222)
ACCENT = (36, 95, 145)
BEFORE = (62, 126, 184)
AFTER = (220, 132, 59)
NEW_FILL = (232, 241, 249)      # v2 modules: tinted so they read as the new part
NEW_EDGE = (36, 95, 145)
BASE_FILL = (245, 246, 248)     # inherited machinery
LOSS_FILL = (252, 246, 236)
LOSS_EDGE = (196, 138, 62)


def s(value: float) -> int:
    return int(round(value * SCALE))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=s(size))
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def centered_text(draw, cx: float, cy: float, text: str, font, fill=INK) -> None:
    width, height = text_size(draw, text, font)
    draw.text((s(cx) - width / 2, s(cy) - height / 2), text, fill=fill, font=font)


def box(
    draw,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[str],
    fonts: list,
    fill=BASE_FILL,
    edge=GRID,
    edge_width: int = 1,
    dashed: bool = False,
) -> tuple[float, float, float, float]:
    """Draw a rounded block with centered, vertically stacked label lines."""
    rect = (s(x), s(y), s(x + w), s(y + h))
    if dashed:  # visual cue for "training only": dashed outline all round
        draw.rounded_rectangle(rect, radius=s(7), fill=fill)
        dash, gap = s(6), s(4)
        for x0 in range(rect[0], rect[2], dash + gap):
            x1 = min(x0 + dash, rect[2])
            draw.line((x0, rect[1], x1, rect[1]), fill=edge, width=s(1))
            draw.line((x0, rect[3], x1, rect[3]), fill=edge, width=s(1))
        for y0 in range(rect[1], rect[3], dash + gap):
            y1 = min(y0 + dash, rect[3])
            draw.line((rect[0], y0, rect[0], y1), fill=edge, width=s(1))
            draw.line((rect[2], y0, rect[2], y1), fill=edge, width=s(1))
    else:
        draw.rounded_rectangle(rect, radius=s(7), fill=fill, outline=edge, width=s(edge_width))

    total = sum(text_size(draw, line, font)[1] + s(4) for line, font in zip(lines, fonts)) - s(4)
    cursor = s(y + h / 2) - total / 2
    for line, font in zip(lines, fonts):
        tw, th = text_size(draw, line, font)
        draw.text((s(x + w / 2) - tw / 2, cursor), line, fill=INK, font=font)
        cursor += th + s(4)
    return x, y, x + w, y + h


def arrow(draw, x1: float, y1: float, x2: float, y2: float, color=MUTED, width: int = 2, head: float = 7) -> None:
    draw.line((s(x1), s(y1), s(x2), s(y2)), fill=color, width=s(width))
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(angle - math.pi / 7), y2 - head * math.sin(angle - math.pi / 7))
    right = (x2 - head * math.cos(angle + math.pi / 7), y2 - head * math.sin(angle + math.pi / 7))
    draw.polygon([(s(x2), s(y2)), (s(left[0]), s(left[1])), (s(right[0]), s(right[1]))], fill=color)


def elbow(draw, x1: float, y1: float, x2: float, y2: float, color=MUTED, width: int = 2) -> None:
    """Right-angled connector: out horizontally, then vertically into the target."""
    mid = x1 + (x2 - x1) * 0.45
    draw.line((s(x1), s(y1), s(mid), s(y1)), fill=color, width=s(width))
    draw.line((s(mid), s(y1), s(mid), s(y2)), fill=color, width=s(width))
    arrow(draw, mid, y2, x2, y2, color=color, width=width)


def draw_training_panel(draw, top: float, width: float, fonts: dict) -> None:
    y_mid = top + 150

    centered_text(draw, 68, top + 8, "(a) Training", fonts["panel"], INK)

    # --- input: view-balanced PK batch -------------------------------------
    box(draw, 24, y_mid - 62, 118, 46,
        ["before view", "(front)"], [fonts["b"], fonts["s"]],
        fill=(238, 245, 251), edge=BEFORE)
    box(draw, 24, y_mid + 16, 118, 46,
        ["after view", "(rear)"], [fonts["b"], fonts["s"]],
        fill=(253, 243, 234), edge=AFTER)
    centered_text(draw, 83, y_mid - 82, "view-balanced PK batch", fonts["s"], MUTED)
    centered_text(draw, 83, y_mid + 78, "+ time / weather label", fonts["s"], MUTED)

    # --- backbone + neck ----------------------------------------------------
    box(draw, 178, y_mid - 40, 104, 80,
        ["Backbone", "(shared)"], [fonts["b"], fonts["s"]])
    arrow(draw, 142, y_mid - 39, 178, y_mid - 12)
    arrow(draw, 142, y_mid + 39, 178, y_mid + 12)

    arrow(draw, 282, y_mid, 318, y_mid)
    centered_text(draw, 300, y_mid + 16, "f", fonts["b"], ACCENT)

    box(draw, 318, y_mid - 46, 116, 92,
        ["CAN", "condition-adaptive", "normalization"],
        [fonts["b"], fonts["s"], fonts["s"]],
        fill=NEW_FILL, edge=NEW_EDGE, edge_width=2)
    centered_text(draw, 376, y_mid + 60, "4 condition branches", fonts["s"], MUTED)
    centered_text(draw, 376, y_mid + 74, "+ shared fallback", fonts["s"], MUTED)

    arrow(draw, 434, y_mid, 470, y_mid)
    centered_text(draw, 452, y_mid - 14, "z", fonts["b"], ACCENT)

    # --- loss branches taken from the raw backbone feature f ---------------
    branch_x = 300
    draw.line((s(branch_x), s(y_mid), s(branch_x), s(top + 46)), fill=MUTED, width=s(2))
    elbow(draw, branch_x, top + 46, 560, top + 46)
    box(draw, 560, top + 28, 128, 36, ["CV-Tri loss"], [fonts["b"]], fill=LOSS_FILL, edge=LOSS_EDGE)

    draw.line((s(branch_x), s(y_mid), s(branch_x), s(top + 96)), fill=MUTED, width=s(2))
    elbow(draw, branch_x, top + 96, 470, top + 96)
    box(draw, 470, top + 78, 74, 36, ["memory"], [fonts["s"]], fill=BASE_FILL, edge=GRID, dashed=True)
    arrow(draw, 544, top + 96, 560, top + 96)
    box(draw, 560, top + 78, 128, 36, ["CVPA loss"], [fonts["b"]], fill=LOSS_FILL, edge=LOSS_EDGE)

    # --- branches taken from the neck output z ------------------------------
    rail_x = 452          # vertical rail carrying z down to the lower branches
    module_x = 476        # left edge of the GRL / CVT blocks
    box(draw, 560, y_mid - 18, 128, 36, ["identity loss"], [fonts["b"]], fill=LOSS_FILL, edge=LOSS_EDGE)
    arrow(draw, 470, y_mid, 560, y_mid)

    draw.line((s(rail_x), s(y_mid), s(rail_x), s(y_mid + 62)), fill=MUTED, width=s(2))
    arrow(draw, rail_x, y_mid + 62, module_x, y_mid + 62)
    box(draw, module_x, y_mid + 44, 68, 36, ["GRL"], [fonts["s"]], fill=BASE_FILL, edge=MUTED, dashed=True)
    arrow(draw, module_x + 68, y_mid + 62, 560, y_mid + 62)
    box(draw, 560, y_mid + 44, 128, 36, ["FCA loss (v1)"], [fonts["b"]], fill=LOSS_FILL, edge=LOSS_EDGE)

    draw.line((s(rail_x), s(y_mid), s(rail_x), s(y_mid + 122)), fill=NEW_EDGE, width=s(2))
    arrow(draw, rail_x, y_mid + 122, module_x, y_mid + 122, color=NEW_EDGE)
    box(draw, module_x, y_mid + 104, 68, 36, ["CVT"], [fonts["b"]],
        fill=NEW_FILL, edge=NEW_EDGE, edge_width=2)
    arrow(draw, module_x + 68, y_mid + 122, 560, y_mid + 122, color=NEW_EDGE)
    box(draw, 560, y_mid + 104, 128, 36, ["CVT loss"], [fonts["b"]], fill=LOSS_FILL, edge=LOSS_EDGE)
    centered_text(draw, 624, y_mid + 152, "targets detached", fonts["s"], MUTED)


def draw_inference_panel(draw, top: float, width: float, fonts: dict) -> None:
    y_q = top + 44
    y_g = top + 116

    centered_text(draw, 74, top + 4, "(b) Inference", fonts["panel"], INK)

    box(draw, 24, y_q - 20, 118, 40, ["query (after)"], [fonts["b"]],
        fill=(253, 243, 234), edge=AFTER)
    box(draw, 24, y_g - 20, 118, 40, ["gallery (before)"], [fonts["b"]],
        fill=(238, 245, 251), edge=BEFORE)

    box(draw, 178, y_q - 46, 104, 132, ["Backbone", "+ CAN"], [fonts["b"], fonts["s"]])
    arrow(draw, 142, y_q, 178, y_q)
    arrow(draw, 142, y_g, 178, y_g)

    box(draw, 470, y_q - 8, 118, 96, ["cosine", "matching"], [fonts["b"], fonts["s"]],
        fill=BASE_FILL, edge=ACCENT)

    # query path: straight out of the neck, then down into the matcher
    draw.line((s(282), s(y_q), s(433), s(y_q)), fill=MUTED, width=s(2))
    draw.line((s(433), s(y_q), s(433), s(y_q + 20)), fill=MUTED, width=s(2))
    arrow(draw, 433, y_q + 20, 470, y_q + 20)
    centered_text(draw, 350, y_q - 14, "q", fonts["b"], ACCENT)

    # gallery path: through the learned transition first
    arrow(draw, 282, y_g, 318, y_g)
    box(draw, 318, y_g - 20, 96, 40, ["T (b to a)"], [fonts["b"]],
        fill=NEW_FILL, edge=NEW_EDGE, edge_width=2)
    draw.line((s(414), s(y_g), s(447), s(y_g)), fill=NEW_EDGE, width=s(2))
    draw.line((s(447), s(y_g), s(447), s(y_q + 60)), fill=NEW_EDGE, width=s(2))
    arrow(draw, 447, y_q + 60, 470, y_q + 60, color=NEW_EDGE)
    centered_text(draw, 366, y_g + 34, "learned view transition", fonts["s"], MUTED)

    arrow(draw, 588, y_q + 40, 624, y_q + 40)
    box(draw, 624, y_q + 22, 96, 36, ["ranking"], [fonts["b"]], fill=LOSS_FILL, edge=LOSS_EDGE)
    centered_text(draw, 672, y_q + 72, "both views compared", fonts["s"], MUTED)
    centered_text(draw, 672, y_q + 86, "in one subspace", fonts["s"], MUTED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="docs/figures")
    args = parser.parse_args()

    width, height = 760, 620
    image = Image.new("RGB", (s(width), s(height)), BG)
    draw = ImageDraw.Draw(image)

    fonts = {
        "title": load_font(15, bold=True),
        "panel": load_font(11, bold=True),
        "b": load_font(9, bold=True),
        "s": load_font(7.5),
    }

    draw.text((s(24), s(18)), "WICV-Net", fill=INK, font=fonts["title"])
    draw.text(
        (s(24), s(40)),
        "Shaded blocks are the proposed structural modules; dashed blocks exist only during training.",
        fill=MUTED,
        font=fonts["s"],
    )

    draw_training_panel(draw, 66, width, fonts)
    draw.line((s(24), s(392), s(width - 24), s(392)), fill=GRID, width=s(1))
    draw_inference_panel(draw, 408, width, fonts)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "figure_09_architecture.png"
    image.save(path, dpi=(300, 300))
    print(f"Saved {path} ({image.width}x{image.height}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

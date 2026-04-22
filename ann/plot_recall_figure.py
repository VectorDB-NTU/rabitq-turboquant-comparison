from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import FIGURES_DIR, RECALL_KS, RESULTS_DIR, dataset_specs


FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
LINE_WIDTH = 2
MARKER_SIZE = 4
BAND_ALPHA = 48
BAND_WHITE_MIX = 0.82


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_rotated_text(
    image: Image.Image,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    angle: int,
) -> None:
    scratch = Image.new("RGBA", (900, 220), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    width, height = text_size(scratch_draw, text, font)
    scratch = scratch.crop((0, 0, width + 8, height + 8))
    scratch_draw = ImageDraw.Draw(scratch)
    scratch_draw.text((4, 4), text, font=font, fill=fill)
    rotated = scratch.rotate(angle, expand=True)
    image.alpha_composite(rotated, dest=position)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    marker: str,
    color: str,
    size: int,
) -> None:
    if marker == "s":
        draw.rectangle(
            [(x - size, y - size), (x + size, y + size)],
            outline=color,
            fill="white",
            width=2,
        )
    elif marker == "o":
        draw.ellipse(
            [(x - size, y - size), (x + size, y + size)],
            outline=color,
            fill="white",
            width=2,
        )
    elif marker == "^":
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        draw.polygon(points, outline=color, fill="white")
    elif marker == "d":
        points = [(x, y - size), (x - size, y), (x, y + size), (x + size, y)]
        draw.polygon(points, outline=color, fill="white")
    elif marker == "*":
        points = []
        for idx in range(10):
            radius = size if idx % 2 == 0 else size * 0.45
            theta = math.pi / 2 + idx * math.pi / 5
            points.append((x + radius * math.cos(theta), y - radius * math.sin(theta)))
        draw.polygon(points, outline=color, fill="white")


def parse_hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def mix_with_white(rgb: tuple[int, int, int], white_mix: float) -> tuple[int, int, int]:
    return tuple(
        round(channel * (1.0 - white_mix) + 255 * white_mix) for channel in rgb
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def draw_band(
    draw: ImageDraw.ImageDraw,
    x_to_px,
    y_to_px,
    mean_values: list[float],
    std_values: list[float],
    color: str,
) -> None:
    upper = [m + s for m, s in zip(mean_values, std_values)]
    lower = [m - s for m, s in zip(mean_values, std_values)]
    polygon = [
        (x_to_px(idx), y_to_px(value)) for idx, value in enumerate(upper)
    ] + [
        (x_to_px(idx), y_to_px(value)) for idx, value in reversed(list(enumerate(lower)))
    ]
    r, g, b = mix_with_white(parse_hex_rgb(color), BAND_WHITE_MIX)
    draw.polygon(polygon, fill=(r, g, b, BAND_ALPHA))


def legend_label_width(
    draw: ImageDraw.ImageDraw,
    style: dict,
    font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
) -> int:
    if "subscript" not in style:
        return text_size(draw, style["label"], font)[0]
    base_w, _ = text_size(draw, style["base_label"], font)
    sub_w, _ = text_size(draw, style["subscript"], sub_font)
    suffix_w, _ = text_size(draw, style["suffix"], font)
    return base_w + sub_w + suffix_w


def draw_legend_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    style: dict,
    font: ImageFont.ImageFont,
    sub_font: ImageFont.ImageFont,
) -> None:
    if "subscript" not in style:
        draw.text((x, y), style["label"], fill="black", font=font)
        return

    base_w, _ = text_size(draw, style["base_label"], font)
    sub_w, _ = text_size(draw, style["subscript"], sub_font)

    draw.text((x, y), style["base_label"], fill="black", font=font)
    draw.text((x + base_w + 1, y + 8), style["subscript"], fill="black", font=sub_font)
    draw.text((x + base_w + sub_w + 3, y), style["suffix"], fill="black", font=font)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    style_map: dict,
    font: ImageFont.ImageFont,
) -> None:
    keys = [
        "turbo_mse_2bit",
        "turbo_mse_4bit",
        "turbo_prod_2bit",
        "turbo_prod_4bit",
        "rabitq_2bit",
        "rabitq_4bit",
    ]
    sub_font = load_font(14)
    row_height = 25
    text_widths = [legend_label_width(draw, style_map[key], font, sub_font) for key in keys]
    width = max(350, 62 + max(text_widths) + 12)
    height = 16 + len(keys) * row_height + 10

    draw.rectangle([(left, top), (left + width, top + height)], fill="white", outline="black", width=1)

    for idx, key in enumerate(keys):
        style = style_map[key]
        cy = top + 15 + idx * row_height
        draw.line([(left + 14, cy), (left + 46, cy)], fill=style["color"], width=2)
        draw_marker(draw, left + 30, cy, style["marker"], style["color"], size=4)
        draw_legend_label(draw, left + 58, cy - 11, style, font, sub_font)


def draw_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    panel_box: tuple[int, int, int, int],
    title: str,
    series: dict,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    style_map: dict,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
    tick_font: ImageFont.ImageFont,
    legend_font: ImageFont.ImageFont,
    plot_left_padding: int,
) -> None:
    left, top, right, bottom = panel_box
    plot_left = left + plot_left_padding
    plot_top = top + 40
    plot_right = right - 18
    plot_bottom = bottom - 62
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    title_w, _ = text_size(draw, title, title_font)
    draw.text((left + (right - left - title_w) / 2, top), title, fill="black", font=title_font)

    def x_to_px(idx: int) -> float:
        return plot_left + idx * (plot_width / (len(RECALL_KS) - 1))

    def y_to_px(value: float) -> float:
        ratio = (clamp(value, y_min, y_max) - y_min) / (y_max - y_min)
        return plot_top + plot_height * (1.0 - ratio)

    grid_color = "#C8C8C8"
    for tick in y_ticks:
        y = y_to_px(tick)
        draw.line([(plot_left, y), (plot_right, y)], fill=grid_color, width=1)
        label = f"{tick:.3f}" if y_min >= 0.8 else f"{tick:.1f}"
        lw, lh = text_size(draw, label, tick_font)
        draw.text((plot_left - 12 - lw, y - lh / 2), label, fill="black", font=tick_font)

    for idx, k in enumerate(RECALL_KS):
        x = x_to_px(idx)
        draw.line([(x, plot_top), (x, plot_bottom)], fill=grid_color, width=1)
        label = str(k)
        lw, lh = text_size(draw, label, tick_font)
        draw.text((x - lw / 2, plot_bottom + 8), label, fill="black", font=tick_font)

    draw.rectangle([(plot_left, plot_top), (plot_right, plot_bottom)], outline="black", width=2)

    for key in [
        "turbo_mse_2bit",
        "turbo_mse_4bit",
        "turbo_prod_2bit",
        "turbo_prod_4bit",
        "rabitq_2bit",
        "rabitq_4bit",
    ]:
        style = style_map[key]
        draw_band(draw, x_to_px, y_to_px, series[key]["mean"], series[key]["std"], style["color"])

    for key in [
        "turbo_mse_2bit",
        "turbo_mse_4bit",
        "turbo_prod_2bit",
        "turbo_prod_4bit",
        "rabitq_2bit",
        "rabitq_4bit",
    ]:
        style = style_map[key]
        points = [(x_to_px(idx), y_to_px(value)) for idx, value in enumerate(series[key]["mean"])]
        draw.line(points, fill=style["color"], width=LINE_WIDTH)
        for x, y in points:
            draw_marker(draw, x, y, style["marker"], style["color"], size=MARKER_SIZE)

    xlabel = "Top-k"
    ylabel = "Recall@1@k"
    xw, _ = text_size(draw, xlabel, label_font)
    draw.text((plot_left + (plot_width - xw) / 2, bottom - 35), xlabel, fill="black", font=label_font)
    draw_rotated_text(image, (left - 8, plot_top + plot_height // 2 - 95), ylabel, label_font, "black", 90)

    legend_width = 372
    draw_legend(draw, plot_right - legend_width, plot_bottom - 184, style_map, legend_font)


def export_plot_data(results: dict, output_json: Path, output_csv: Path) -> None:
    method_map = {
        "turbo_mse_2bit": "TurboQuant_mse 2 bits",
        "turbo_mse_4bit": "TurboQuant_mse 4 bits",
        "turbo_prod_2bit": "TurboQuant_prod 2 bits",
        "turbo_prod_4bit": "TurboQuant_prod 4 bits",
        "rabitq_2bit": "RaBitQ 2 bits",
        "rabitq_4bit": "RaBitQ 4 bits",
    }
    payload: dict[str, dict[str, dict[str, list[float]]]] = {}
    rows: list[dict[str, object]] = []
    for dataset_key, panel in results.items():
        payload[dataset_key] = {}
        for key, label in method_map.items():
            payload[dataset_key][label] = {
                "mean": panel[key]["mean"],
                "std": panel[key]["std"],
            }
            for idx, topk in enumerate(RECALL_KS):
                rows.append(
                    {
                        "dataset": dataset_key,
                        "method": label,
                        "topk": topk,
                        "mean": panel[key]["mean"][idx],
                        "std": panel[key]["std"][idx],
                    }
                )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "topk", "mean", "std"])
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results: dict, output_png: Path) -> None:
    image = Image.new("RGBA", (2260, 650), "white")
    draw = ImageDraw.Draw(image)

    title_font = load_font(30)
    label_font = load_font(26)
    tick_font = load_font(18)
    legend_font = load_font(18)

    style_map = {
        "turbo_mse_2bit": {
            "base_label": "TurboQuant",
            "subscript": "mse",
            "suffix": " 2 bits",
            "color": "#21B5D9",
            "marker": "s",
        },
        "turbo_mse_4bit": {
            "base_label": "TurboQuant",
            "subscript": "mse",
            "suffix": " 4 bits",
            "color": "#0B3C6F",
            "marker": "o",
        },
        "turbo_prod_2bit": {
            "base_label": "TurboQuant",
            "subscript": "prod",
            "suffix": " 2 bits",
            "color": "#2A9D8F",
            "marker": "^",
        },
        "turbo_prod_4bit": {
            "base_label": "TurboQuant",
            "subscript": "prod",
            "suffix": " 4 bits",
            "color": "#1D6F62",
            "marker": "d",
        },
        "rabitq_2bit": {"label": "RaBitQ 2 bits", "color": "#6B3F1D", "marker": "*"},
        "rabitq_4bit": {"label": "RaBitQ 4 bits", "color": "#D81B8A", "marker": "*"},
    }

    panel_width = 710
    panel_height = 590
    gap = 28
    left_margin = 18
    top_margin = 18
    panel_boxes = []
    for idx in range(3):
        left = left_margin + idx * (panel_width + gap)
        panel_boxes.append((left, top_margin, left + panel_width, top_margin + panel_height))

    specs = dataset_specs()
    for panel_box, dataset_key in zip(panel_boxes, ["glove_200", "openai_1536", "openai_3072"]):
        spec = specs[dataset_key]
        draw_panel(
            image=image,
            draw=draw,
            panel_box=panel_box,
            title=spec.title,
            series=results[dataset_key],
            y_min=spec.y_min,
            y_max=spec.y_max,
            y_ticks=spec.y_ticks,
            style_map=style_map,
            title_font=title_font,
            label_font=label_font,
            tick_font=tick_font,
            legend_font=legend_font,
            plot_left_padding=spec.plot_left_padding,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_png)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the combined three-panel RaBitQ vs TurboQuant figure from saved experiment results."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=RESULTS_DIR / "recall_at1_three_panel.json",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURES_DIR / "recall_at1_three_panel.png",
    )
    parser.add_argument(
        "--plot-data-json",
        type=Path,
        default=FIGURES_DIR / "recall_at1_three_panel.json",
    )
    parser.add_argument(
        "--plot-data-csv",
        type=Path,
        default=FIGURES_DIR / "recall_at1_three_panel.csv",
    )
    args = parser.parse_args()

    with args.results_json.resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = {key: value for key, value in payload.items() if key in {"glove_200", "openai_1536", "openai_3072"}}
    plot_results(results, args.output_png.resolve())
    export_plot_data(results, args.plot_data_json.resolve(), args.plot_data_csv.resolve())
    print(f"Saved figure to {args.output_png.resolve()}")
    print(f"Saved plot data to {args.plot_data_json.resolve()}")
    print(f"Saved plot data CSV to {args.plot_data_csv.resolve()}")


if __name__ == "__main__":
    main()

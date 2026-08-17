#!/usr/bin/env python3
"""Build the story-aligned CROME CS figure package with Python/Matplotlib.

Figure contract
---------------
pipeline: declared evidence is verified and composed into a three-axis product.
rq1: certificate optimization reduces valid radius and raises point yield without
     reducing coverage on the predeclared ill-conditioned recovery slice.
rq2: matching point-output count does not make uncertainty gating structurally safe.
rq3: verified single-component removals lose point yield or structural recovery
     without leaking points; an unsafe forced-point control demonstrates why the
     verified public path cannot be bypassed.

All quantitative panels are computed from the 2026-08-12 revision summaries.  Exports are
double-column vector figures with editable text plus 600-dpi TIFF and PNG previews.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "results" / "revision_20260812" / "summaries"
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "cs"

# One restrained family for the whole package. CROME is navy in every figure;
# legal controls stay slate; the unsafe bypass is neutral grey, not a fourth hue.
COLORS = {
    "crome": "#0F4D92",
    "crome_mid": "#3D6FA8",
    "crome_soft": "#6D8FB5",
    "baseline": "#6B7C93",
    "scope": "#B7A48C",
    "unsafe": "#4D4D4D",
    "threshold": "#555555",
    "black": "#272727",
    "muted": "#5A5A5A",
    "hair": "#C8C8C8",
    "band": "#F1F3F5",
    "tray": "#F6F7F8",
    "white": "#FFFFFF",
}


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "regular",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _load_summary(stem: str) -> dict[str, Any]:
    with (SUMMARY_DIR / f"{stem}_main.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metadata = {
        "Title": stem,
        "Author": "CROME figure pipeline",
        "Creator": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    for suffix in ("pdf", "svg", "png", "tiff"):
        path = output_dir / f"{stem}.{suffix}"
        options: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == "pdf":
            options["metadata"] = metadata
        elif suffix == "svg":
            options["metadata"] = {"Title": stem}
        elif suffix == "png":
            options["dpi"] = 300
            options["metadata"] = {"Software": "Matplotlib", "Title": stem}
        else:
            options["dpi"] = 600
        fig.savefig(path, format=suffix, **options)
        paths.append(path)
    plt.close(fig)
    return paths


def _asymmetric_error(rate: float, interval: dict[str, Any]) -> np.ndarray:
    return np.array(
        [[rate - float(interval["lower"])], [float(interval["upper"]) - rate]],
        dtype=float,
    )


def _hex_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def _tint(color: str, amount: float = 0.88) -> tuple[float, float, float]:
    red, green, blue = _hex_rgb(color)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )


def _luminance(color: str) -> float:
    red, green, blue = _hex_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _on_fill(color: str) -> str:
    return COLORS["white"] if _luminance(color) < 0.45 else COLORS["black"]


def _panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=6, fontweight="regular")


def _clean_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.4, pad=1.6)


def _horizontal_arrow(ax: plt.Axes, x0: float, x1: float, y: float, *, color: str) -> None:
    """Draw a filled arrow long enough to survive PDF downsampling."""

    span = max(x1 - x0, 1e-3)
    head_len = min(0.013, 0.30 * span)
    head_half = 0.016
    shaft_half = 0.0046
    neck = x1 - head_len
    ax.add_patch(
        Polygon(
            [
                (x0, y - shaft_half),
                (neck, y - shaft_half),
                (neck, y - head_half),
                (x1, y),
                (neck, y + head_half),
                (neck, y + shaft_half),
                (x0, y + shaft_half),
            ],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0.2,
            joinstyle="miter",
            zorder=5,
            clip_on=False,
        )
    )


def _pipeline_figure() -> tuple[plt.Figure, list[dict[str, Any]]]:
    fig, ax = plt.subplots(figsize=(7.15, 2.62))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    nodes = [
        ("01", COLORS["baseline"], "producer", "Evidence", "typed payload\n+ provenance"),
        ("02", COLORS["crome"], "verifier", "Verify", "exact predicates\n+ bound recomputation"),
        ("03", COLORS["crome"], "inner layer", "Optimize", r"$w^{\star}=\arg\min_w\,\mathcal{R}(w)$"),
        ("04", COLORS["baseline"], "outer layer", "Compose", "scope + joint\nfailure ledger"),
        ("05", COLORS["scope"], "public claim", "Route", "three-axis\nproduct"),
    ]
    count = len(nodes)
    left, right, gap = 0.014, 0.986, 0.062
    width = (right - left - (count - 1) * gap) / count
    box_y, box_h = 0.448, 0.430
    mid_y = box_y + 0.50 * box_h
    arrow_inset = 0.008
    arrow_color = COLORS["black"]

    for index, (number, color, role, title, subtitle) in enumerate(nodes):
        x = left + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, box_y),
                width,
                box_h,
                boxstyle="round,pad=0.0035,rounding_size=0.016",
                facecolor=_tint(color, 0.88),
                edgecolor=color,
                linewidth=1.05,
                zorder=2,
                clip_on=False,
            )
        )
        ax.text(
            x + width / 2.0,
            0.948,
            role,
            ha="center",
            va="center",
            fontsize=6.1,
            color=COLORS["muted"],
            zorder=4,
        )
        ax.text(
            x + width / 2.0,
            box_y + 0.338,
            number,
            ha="center",
            va="center",
            fontsize=5.8,
            color=color,
            weight="bold",
            zorder=4,
        )
        ax.text(
            x + width / 2.0,
            box_y + 0.250,
            title,
            ha="center",
            va="center",
            fontsize=8.3,
            weight="bold",
            color=COLORS["black"],
            zorder=4,
        )
        ax.text(
            x + width / 2.0,
            box_y + 0.108,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.9,
            color="#444444",
            linespacing=1.22,
            zorder=4,
        )
        if index < count - 1:
            _horizontal_arrow(
                ax,
                x + width + arrow_inset,
                x + width + gap - arrow_inset,
                mid_y,
                color=arrow_color,
            )

    tray_x, tray_w, tray_y, tray_h = left, right - left, 0.126, 0.168
    rule_y = box_y - 0.022
    ax.plot(
        [left + 0.004, right - 0.004],
        [rule_y, rule_y],
        color=COLORS["hair"],
        linewidth=0.7,
        solid_capstyle="round",
        clip_on=False,
        zorder=4,
    )
    ax.plot(
        [0.5, 0.5],
        [rule_y, tray_y + tray_h + 0.006],
        color=COLORS["hair"],
        linewidth=0.7,
        solid_capstyle="round",
        clip_on=False,
        zorder=4,
    )
    ax.add_patch(
        FancyBboxPatch(
            (tray_x, tray_y),
            tray_w,
            tray_h,
            boxstyle="round,pad=0.003,rounding_size=0.024",
            facecolor=COLORS["tray"],
            edgecolor="#D0D2D4",
            linewidth=0.7,
            zorder=2,
            clip_on=False,
        )
    )
    axes_spec = (
        (COLORS["crome"], "Structural"),
        (COLORS["baseline"], "Operational"),
        (COLORS["scope"], "Scope"),
    )
    for index, (color, label) in enumerate(axes_spec):
        x = tray_x + (index + 0.5) * tray_w / 3.0
        ax.scatter(
            [x - 0.058],
            [tray_y + tray_h / 2.0],
            s=16,
            color=color,
            zorder=4,
            linewidths=0,
            clip_on=False,
        )
        ax.text(
            x - 0.046,
            tray_y + tray_h / 2.0,
            label,
            ha="left",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=color,
            zorder=4,
        )

    ax.text(
        0.5,
        0.050,
        "Structural · operational · scope remain separate; unverified evidence and missing ledger entries cannot escalate a claim.",
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["muted"],
        zorder=4,
    )
    rows = [
        {"figure": "pipeline", "panel": "workflow", "metric": "nodes", "value": 5, "n": ""}
    ]
    return fig, rows


def _rq1_figure(cs03: dict[str, Any]) -> tuple[plt.Figure, list[dict[str, Any]]]:
    metric = cs03["story_metrics"]["rq1"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 2.72), gridspec_kw={"wspace": 0.32}
    )
    labels = ["LS dual", "Certificate-\noptimized"]
    colors = [COLORS["baseline"], COLORS["crome"]]

    radii = [
        float(metric["median_radius_current"]),
        float(metric["median_radius_optimal"]),
    ]
    bars = axes[0].bar(
        labels,
        radii,
        width=0.62,
        color=colors,
        edgecolor=COLORS["black"],
        linewidth=0.55,
        zorder=3,
    )
    axes[0].axhline(
        1.10,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=0.8,
        zorder=2,
    )
    for bar, value, color in zip(bars, radii, colors):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.09,
            f"{value:.3f}",
            ha="center",
            va="top",
            color=_on_fill(color),
            fontsize=6.8,
            zorder=6,
        )
    axes[0].text(
        1.38,
        1.18,
        r"$\tau=1.10$",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=COLORS["threshold"],
        clip_on=False,
        zorder=7,
    )
    axes[0].set_xlim(-0.55, 1.72)
    axes[0].set_ylim(0, 2.28)
    axes[0].set_ylabel("Certified radius")
    _panel_title(axes[0], "a", "Certified radius")
    _clean_axes(axes[0])

    x = np.arange(2)
    point_yield = [
        100 * float(metric["current"]["point_yield"]),
        100 * float(metric["optimal"]["point_yield"]),
    ]
    coverage = [
        100 * float(metric["current"]["marginal_coverage"]),
        100 * float(metric["optimal"]["marginal_coverage"]),
    ]
    width = 0.34
    yield_bars = axes[1].bar(
        x - width / 2,
        point_yield,
        width,
        color=colors,
        edgecolor=COLORS["black"],
        linewidth=0.55,
        zorder=3,
    )
    coverage_colors = [_tint(color, 0.72) for color in colors]
    coverage_bars = axes[1].bar(
        x + width / 2,
        coverage,
        width,
        color=coverage_colors,
        edgecolor=colors,
        linewidth=0.7,
        zorder=3,
    )
    for bar, value, color in zip(yield_bars, point_yield, colors):
        if value < 4:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                2.4,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                color=COLORS["black"],
                fontsize=6.4,
                zorder=6,
            )
        else:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                50.0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=_on_fill(color),
                fontsize=6.4,
                zorder=6,
            )
    for bar, value, face in zip(coverage_bars, coverage, coverage_colors):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            50.0,
            f"{value:.1f}%",
            ha="center",
            va="center",
            color=_on_fill(face if isinstance(face, str) else _rgb_to_hex(face)),
            fontsize=6.4,
            zorder=6,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Trials (%)")
    _panel_title(axes[1], "b", "Yield and coverage")
    _clean_axes(axes[1])
    axes[1].legend(
        handles=[
            Patch(
                facecolor=COLORS["crome"],
                edgecolor=COLORS["black"],
                linewidth=0.5,
                label="POINT yield",
            ),
            Patch(
                facecolor=_tint(COLORS["crome"], 0.72),
                edgecolor=COLORS["crome"],
                linewidth=0.7,
                label="Interval coverage",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.2,
        handleheight=0.72,
        borderaxespad=0.0,
        frameon=False,
    )

    rows = []
    for method, radius, yield_value, coverage_value in zip(
        labels, radii, point_yield, coverage
    ):
        rows.extend(
            [
                {
                    "figure": "rq1",
                    "panel": "a",
                    "metric": f"{method} median radius",
                    "value": radius,
                    "n": 200,
                },
                {
                    "figure": "rq1",
                    "panel": "b",
                    "metric": f"{method} point yield percent",
                    "value": yield_value,
                    "n": 200,
                },
                {
                    "figure": "rq1",
                    "panel": "b",
                    "metric": f"{method} interval coverage percent",
                    "value": coverage_value,
                    "n": 200,
                },
            ]
        )
    return fig, rows


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )


def _rq2_figure(cs03: dict[str, Any]) -> tuple[plt.Figure, list[dict[str, Any]]]:
    story = cs03["story_metrics"]["rq2"]
    methods = cs03["methods"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.72),
        gridspec_kw={"width_ratios": [0.86, 1.14], "wspace": 0.30},
    )
    labels = ["CROME", "Matched\nuncertainty"]
    colors = [COLORS["crome"], COLORS["baseline"]]
    point_counts = [int(story["point_outputs"]), int(story["point_outputs"])]
    bars = axes[0].bar(
        labels,
        point_counts,
        width=0.58,
        color=colors,
        edgecolor=COLORS["black"],
        linewidth=0.55,
        zorder=3,
    )
    for bar, value in zip(bars, point_counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 48,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            zorder=6,
        )
    axes[0].set_ylim(0, max(point_counts) * 1.20)
    axes[0].set_ylabel("POINT outputs")
    _panel_title(axes[0], "a", "Matched output count")
    _clean_axes(axes[0])

    rates = [
        100 * float(methods["crome_optimal"]["false_point_rate"]),
        100 * float(methods["matched_uncertainty"]["false_point_rate"]),
    ]
    intervals = [
        methods["crome_optimal"]["false_point_interval"],
        methods["matched_uncertainty"]["false_point_interval"],
    ]
    yerr = (
        np.hstack(
            [
                _asymmetric_error(rate / 100, interval)
                for rate, interval in zip(rates, intervals)
            ]
        )
        * 100
    )
    bars = axes[1].bar(
        labels,
        rates,
        width=0.58,
        yerr=yerr,
        capsize=2.4,
        color=colors,
        edgecolor=COLORS["black"],
        linewidth=0.55,
        error_kw={
            "elinewidth": 0.85,
            "capthick": 0.85,
            "ecolor": COLORS["black"],
            "zorder": 5,
        },
        zorder=3,
    )
    uppers = [100 * float(interval["upper"]) for interval in intervals]
    for x_pos, rate, upper in zip(np.arange(len(rates)), rates, uppers):
        axes[1].text(
            x_pos,
            max(rate, upper) + 4.4,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.7,
            zorder=6,
        )
    axes[1].set_ylim(0, max(uppers) + 14)
    axes[1].set_ylabel("False POINTs among non-point cases (%)")
    _panel_title(axes[1], "b", "Structural leakage after matching")
    _clean_axes(axes[1])

    rows = [
        {
            "figure": "rq2",
            "panel": "a",
            "metric": f"{label} point outputs",
            "value": value,
            "n": 3840,
        }
        for label, value in zip(labels, point_counts)
    ]
    rows.extend(
        {
            "figure": "rq2",
            "panel": "b",
            "metric": f"{label} false point percent",
            "value": value,
            "n": 640,
        }
        for label, value in zip(labels, rates)
    )
    return fig, rows


def _rq3_figure(cs06: dict[str, Any]) -> tuple[plt.Figure, list[dict[str, Any]]]:
    panels = (
        (
            "a",
            "Support producer",
            "POINT yield\namong point cases",
            "point_oracle_yield",
            (("full_crome", "Full CROME"), ("no_support_certifier", "No support certifier")),
        ),
        (
            "b",
            "Exact-null payload",
            "Structural-null\nrecovery",
            "structural_null_recovery",
            (("full_crome", "Full CROME"), ("no_target_null_check", "No null payload")),
        ),
        (
            "c",
            "Public router",
            "False POINTs\namong non-point cases",
            "false_point",
            (("full_crome", "Full CROME"), ("force_point", "Unsafe forced POINT")),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), sharex=True)
    rows: list[dict[str, Any]] = []
    for ax, (panel, title, xlabel, metric, methods) in zip(axes, panels, strict=True):
        for yi, (key, label) in zip((1, 0), methods, strict=True):
            summary = cs06["variants"][key][metric]
            rate = 100.0 * float(summary["rate"])
            interval = summary["interval"]
            lower = 100.0 * float(interval["lower"])
            upper = 100.0 * float(interval["upper"])
            unsafe = key == "force_point"
            color = COLORS["unsafe"] if unsafe else COLORS["crome"] if yi == 1 else COLORS["crome_mid"]
            marker = "D" if unsafe else "o"
            face = COLORS["white"] if unsafe else color
            ax.plot([lower, upper], [yi, yi], color=color, linewidth=1.5, solid_capstyle="round")
            ax.plot(rate, yi, marker=marker, markersize=5.5, markerfacecolor=face,
                    markeredgecolor=color, markeredgewidth=1.0, zorder=3)
            count = int(summary.get("count", cs06["variants"][key].get("false_point_count", 0)))
            total = int(summary["total"])
            ax.text(rate, yi + 0.25, f"{count}/{total}", ha="center", va="bottom",
                    fontsize=6.2, color=COLORS["black"])
            rows.append({
                "figure": "rq3", "panel": panel,
                "metric": f"{key} {metric} percent", "value": rate, "n": total,
            })
        ax.set_yticks([0, 1])
        ax.set_yticklabels([methods[1][1], methods[0][1]])
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(-0.45, 1.55)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=8)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.spines["left"].set_visible(False)
        _clean_axes(ax)
    fig.suptitle(
        "Single-component verified ablations lose utility or structural recovery; only router bypass leaks POINT",
        x=0.01, y=1.02, ha="left", fontsize=8.0,
    )
    fig.subplots_adjust(wspace=0.48)
    return fig, rows


def build_story_figures(output_dir: Path | str | None = None) -> dict[str, list[Path]]:
    """Generate the schematic and one quantitative figure for each main RQ."""

    _style()
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    cs03 = _load_summary("cs03")
    cs06 = _load_summary("cs06")
    builders = {
        "pipeline": ("figure_00_evidence_certificate_route", _pipeline_figure()),
        "rq1": ("figure_01_certificate_optimization", _rq1_figure(cs03)),
        "rq2": ("figure_02_matched_structural_safety", _rq2_figure(cs03)),
        "rq3": ("figure_03_routing_ablation", _rq3_figure(cs06)),
    }
    exported: dict[str, list[Path]] = {}
    source_rows: list[dict[str, Any]] = []
    for key, (stem, (figure, rows)) in builders.items():
        exported[key] = _save(figure, destination, stem)
        source_rows.extend(rows)

    source_path = destination / "source_data.csv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["figure", "panel", "metric", "value", "n"]
        )
        writer.writeheader()
        writer.writerows(source_rows)

    manifest = {
        "backend": "Python / Matplotlib",
        "source_summaries": {
            "cs03_main.json": _sha256(SUMMARY_DIR / "cs03_main.json"),
            "cs06_main.json": _sha256(SUMMARY_DIR / "cs06_main.json"),
        },
        "figures": {
            key: {
                path.suffix.lstrip("."): {"name": path.name, "sha256": _sha256(path)}
                for path in paths
            }
            for key, paths in exported.items()
        },
        "source_data": {"name": source_path.name, "sha256": _sha256(source_path)},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return exported


def main() -> None:
    exported = build_story_figures()
    print(
        json.dumps(
            {key: [path.name for path in paths] for key, paths in exported.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

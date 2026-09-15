#!/usr/bin/env python3
"""Build the illustrative multiscale-geometry figure from three ArtBench images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from painting_geometry.curvature import level_set_curvature_dog
from painting_geometry.preprocessing import preprocess


PAINTINGS = (
    {
        "role": "Low geometry",
        "artist": "Anita Malfatti",
        "title": "Fernanda de Castro",
        "year": "1922",
        "filename": "anita-malfatti_fernanda-de-castro-1922.jpg",
        "value": 0.252172,
    },
    {
        "role": "Intermediate geometry",
        "artist": "Amrita Sher-Gil",
        "title": "Tribal Women",
        "year": "1938",
        "filename": "amrita-sher-gil_tribal-women-1938.jpg",
        "value": 0.324556,
    },
    {
        "role": "High geometry",
        "artist": "Abraham Manievich",
        "title": "The Yellow House",
        "year": "n.d.",
        "filename": "abraham-manievich_the-yellow-house.jpg",
        "value": 0.403162,
    },
)


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.06,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=20,
        fontweight="bold",
    )


def clean_axis(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_figure(figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure1_multiscale_luminance_geometry"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def main(image_dir: Path, output_dir: Path) -> None:
    records = []
    for metadata in PAINTINGS:
        path = image_dir / metadata["filename"]
        rgb, luminance = preprocess(path, long_side=256)
        records.append({**metadata, "rgb": rgb, "luminance": luminance})

    middle = records[1]
    maps = {}
    masks = {}
    absolute_values = []
    for sigma_ref in (1, 2, 4, 8):
        result = level_set_curvature_dog(
            middle["luminance"],
            sigma_px=sigma_ref * 256 / 512,
            sigma_ref=sigma_ref,
            grad_quantile=0.20,
        )
        maps[sigma_ref] = result.curvature_scale_normalized
        masks[sigma_ref] = result.valid_mask
        absolute_values.append(np.abs(maps[sigma_ref][masks[sigma_ref]]))
    shared_limit = float(np.percentile(np.concatenate(absolute_values), 99))

    figure = plt.figure(figsize=(14.4, 10.2))
    layout = gridspec.GridSpec(
        3, 12, figure=figure, height_ratios=(1.08, 1.0, 1.12), hspace=0.34, wspace=0.18
    )
    figure.suptitle(
        "From painting to multiscale luminance geometry",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )

    for column, record in enumerate(records):
        axis = figure.add_subplot(layout[0, 4 * column : 4 * (column + 1)])
        if column == 0:
            panel_label(axis, "a")
        axis.imshow(record["rgb"])
        clean_axis(axis)
        axis.set_title(
            f"{record['role']}\n{record['artist']}, {record['title']} ({record['year']})\n"
            rf"$G_{{\sigma=2}}={record['value']:.3f}$",
            fontsize=9.8,
        )

    for column, record in enumerate(records):
        axis = figure.add_subplot(layout[1, 4 * column : 4 * (column + 1)])
        if column == 0:
            panel_label(axis, "b")
        smoothed = gaussian_filter(
            record["luminance"], sigma=1.0, mode="reflect", truncate=3.0
        )
        levels = np.unique(np.quantile(smoothed, (0.15, 0.30, 0.50, 0.70, 0.85)))
        axis.imshow(smoothed, cmap="gray", vmin=0, vmax=1)
        axis.contour(smoothed, levels=levels, colors="white", linewidths=1.05)
        clean_axis(axis)
        axis.set_title("Iso-luminance contours", fontsize=9.8)

    lower = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=layout[2, :], wspace=0.06)
    axes = []
    image_artist = None
    for column, sigma_ref in enumerate((1, 2, 4, 8)):
        axis = figure.add_subplot(lower[0, column])
        if column == 0:
            panel_label(axis, "c")
        axis.imshow(np.clip(0.86 * middle["luminance"] + 0.14, 0, 1), cmap="gray")
        overlay = np.ma.masked_where(~masks[sigma_ref], maps[sigma_ref])
        image_artist = axis.imshow(
            overlay,
            cmap="coolwarm",
            vmin=-shared_limit,
            vmax=shared_limit,
            alpha=0.78,
        )
        clean_axis(axis)
        axis.set_title(rf"$\sigma_{{\mathrm{{ref}}}}={sigma_ref}$", fontsize=10.5)
        axes.append(axis)

    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    bottom = min(axis.get_position().y0 for axis in axes) - 0.045
    color_axis = figure.add_axes([left + 0.08, bottom, right - left - 0.16, 0.016])
    colorbar = figure.colorbar(image_artist, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks((-shared_limit, 0, shared_limit))
    colorbar.set_label(r"scale-normalised curvature $\widetilde{\kappa}_{\sigma}$")

    figure.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.10)
    save_figure(figure, output_dir)
    plt.close(figure)
    print(f"Figure 1 written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "build" / "figures")
    arguments = parser.parse_args()
    main(arguments.image_dir, arguments.output_dir)

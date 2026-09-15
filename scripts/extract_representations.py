#!/usr/bin/env python3
"""Extract B90, K40, and the four additional G44 descriptors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from painting_geometry.appearance import (
    lbp_features,
    multidistance_glcm_features,
    multiscale_gradient_features,
    orientation_histogram_features,
)
from painting_geometry.curvature import relative_scale_curvature_features
from painting_geometry.orientation import structure_tensor_features
from painting_geometry.preprocessing import preprocess


def extract_one(path: Path, long_side: int = 256) -> dict[str, float]:
    _, luminance = preprocess(path, long_side=long_side)
    reference_long_side = 512
    sigma_refs = (1.0, 2.0, 4.0, 8.0)
    sigma_pixels = tuple(
        sigma * long_side / reference_long_side for sigma in sigma_refs
    )
    orientation_sigma = 2.0 * long_side / reference_long_side

    curvature = relative_scale_curvature_features(
        luminance,
        long_side=long_side,
        sigma_refs=sigma_refs,
        reference_long_side=reference_long_side,
    )
    tensor = structure_tensor_features(luminance, sigma=orientation_sigma)
    appearance: dict[str, float] = {}
    appearance.update(multiscale_gradient_features(luminance, sigmas=sigma_pixels))
    appearance.update(
        orientation_histogram_features(luminance, sigma=orientation_sigma)
    )
    appearance.update(multidistance_glcm_features(luminance, distances=(1, 2, 4)))
    appearance.update(lbp_features(luminance))

    features = {f"geom__curv__{key}": value for key, value in curvature.items()}
    features.update({f"geom__orient__{key}": value for key, value in tensor.items()})
    features.update({f"base__{key}": value for key, value in appearance.items()})
    return features


def write_checkpoint(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main(manifest_path: Path, output_path: Path, checkpoint_every: int) -> None:
    manifest = pd.read_csv(manifest_path)
    required = {"split", "style", "artist", "filename", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")

    checkpoint = output_path.with_suffix(".checkpoint.csv")
    failures_path = output_path.with_suffix(".failures.csv")
    rows: list[dict] = []
    completed: set[str] = set()
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        rows = prior.to_dict("records")
        completed = set(prior["record_id"].astype(str))

    failures: list[dict] = []
    for index, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Image representations"
    ):
        record_id = f"{row['split']}/{row['style']}/{row['filename']}"
        if record_id in completed:
            continue
        metadata = {
            "record_id": record_id,
            "split": row["split"],
            "style": row["style"],
            "artist": row["artist"],
            "source": row.get("source", ""),
            "filename": row["filename"],
            "path": row["path"],
            "long_side": 256,
        }
        try:
            metadata.update(extract_one(Path(row["path"])))
            rows.append(metadata)
            completed.add(record_id)
        except Exception as exc:
            failures.append({**metadata, "error": repr(exc)})

        if checkpoint_every > 0 and (index + 1) % checkpoint_every == 0:
            write_checkpoint(rows, checkpoint)
            if failures:
                write_checkpoint(failures, failures_path)

    write_checkpoint(rows, checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_csv(output_path, index=False)
    if failures:
        write_checkpoint(failures, failures_path)

    counts = {
        "B90": sum(column.startswith("base__") for column in output.columns),
        "K40": sum(column.startswith("geom__curv__") for column in output.columns),
        "G44 additional": sum(
            column.startswith("geom__orient__") for column in output.columns
        ),
    }
    expected = {"B90": 90, "K40": 40, "G44 additional": 4}
    if counts != expected:
        raise RuntimeError(f"Unexpected feature dimensions: {counts}")
    print(f"Wrote {len(output):,} rows to {output_path}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    arguments = parser.parse_args()
    main(arguments.manifest, arguments.output, arguments.checkpoint_every)

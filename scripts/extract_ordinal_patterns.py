#!/usr/bin/env python3
"""Append the 75 tie-aware ordinal-pattern probabilities to a feature matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from painting_geometry.ordinal import (
    OP75_COLUMNS,
    load_ordinal_grayscale,
    ordinal_pattern_probabilities,
    validate_against_ordpy,
)


def main(features_path: Path, output_path: Path, checkpoint_every: int) -> None:
    validate_against_ordpy()
    features = pd.read_csv(features_path)
    if "path" not in features:
        raise KeyError("The feature matrix must contain an image path column")

    checkpoint = output_path.with_suffix(".ordinal_checkpoint.csv")
    completed: dict[int, dict] = {}
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        completed = {
            int(row["row_index"]): row for row in prior.to_dict("records")
        }

    failures: list[dict] = []
    pending = [index for index in features.index if index not in completed]
    for position, index in enumerate(
        tqdm(pending, desc="Tie-aware ordinal patterns"), start=1
    ):
        try:
            gray = load_ordinal_grayscale(str(features.at[index, "path"]))
            completed[index] = {
                "row_index": int(index),
                **ordinal_pattern_probabilities(gray),
            }
        except Exception as exc:
            failures.append({"row_index": int(index), "error": repr(exc)})
        if checkpoint_every > 0 and position % checkpoint_every == 0:
            pd.DataFrame([completed[key] for key in sorted(completed)]).to_csv(
                checkpoint, index=False
            )

    ordinal = pd.DataFrame([completed[key] for key in sorted(completed)])
    ordinal.to_csv(checkpoint, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_path.with_suffix(".failures.csv"), index=False)
        raise RuntimeError(
            f"Ordinal extraction failed for {len(failures)} images; see the failures file"
        )
    if len(ordinal) != len(features) or any(column not in ordinal for column in OP75_COLUMNS):
        raise RuntimeError("The ordinal checkpoint is incomplete")

    ordinal = ordinal.set_index("row_index").reindex(features.index)
    enriched = pd.concat([features.reset_index(drop=True), ordinal.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    print(f"Wrote {len(enriched):,} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    arguments = parser.parse_args()
    main(arguments.features, arguments.output, arguments.checkpoint_every)

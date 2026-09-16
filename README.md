# The geometry of artistic style across spatial scales

Code and compact numerical results accompanying the manuscript *The geometry of artistic style across spatial scales*.

The study represents each digitised painting as a luminance field and summarises the curvature of its equal-intensity contours at four spatial scales. The repository contains the final analysis used in the manuscript: artist-disjoint linear evaluation, artist-cluster bootstrap contrasts, corpus-level geometric organisation, source-composition sensitivity, and paper-output generation.

## Repository contents

```text
painting_geometry/   Descriptor implementations
scripts/             Extraction, evaluation, and figure-building programs
results/             Compact numerical results reported in the manuscript
figures/             Figures 2--4 in PDF, PNG, and SVG formats
tables/              Tables 1--2 in CSV and LaTeX formats
tests/               Numerical and structural tests
data/                Required input schema; source images are not redistributed
docs/                Detailed reproduction guide
```

The repository exposes only the fixed final protocol. Exploratory analyses and superseded model-selection pipelines are intentionally excluded.

## Representations

| Label | Dimensions | Description |
|---|---:|---|
| B90 | 90 | Gradients, edge densities, orientation, co-occurrence, and local binary patterns |
| K10 | 10 | Curvature summaries at one reference scale |
| K40 | 40 | Four K10 blocks at reference scales 1, 2, 4, and 8 |
| G44 | 44 | K40 plus four structure-tensor summaries |
| OP75 | 75 | Probabilities of all tie-aware weak orderings in overlapping 2 x 2 windows |

## Quick reproduction of paper outputs

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_paper_outputs.py
```

The command rebuilds Figures 2–4 and Tables 1–2 in `build/` from the compact CSV files in `results/`. It does not require access to the source images.

Run the validation suite with:

```bash
python -m pytest
```

## Full analysis from images

ArtBench-10 images are not redistributed. Arrange the images and artist metadata as described in [`data/README.md`](data/README.md), then run:

```bash
python scripts/prepare_artbench_manifest.py \
  --dataset-root /path/to/artbench \
  --metadata-csv /path/to/artist_metadata.csv \
  --output data/derived/artbench_manifest.csv

python scripts/extract_representations.py \
  --manifest data/derived/artbench_manifest.csv \
  --output data/derived/appearance_geometry.csv

python scripts/extract_ordinal_patterns.py \
  --features data/derived/appearance_geometry.csv \
  --output data/derived/all_representations.csv

python scripts/run_artist_disjoint_evaluation.py \
  --features data/derived/all_representations.csv \
  --output-dir build/evaluation \
  --outer-folds 5 \
  --n-boot 5000

python scripts/run_geometric_organisation.py \
  --features data/derived/all_representations.csv \
  --output-dir build/geometric_organisation \
  --n-permutations 4999

python scripts/run_source_sensitivity.py \
  --features data/derived/all_representations.csv \
  --output-dir build/source_sensitivity \
  --n-permutations 4999
```

Extraction and evaluation use checkpoints because the full analysis is computationally intensive. The fixed numerical settings are recorded in [`analysis_specification.json`](analysis_specification.json), and the expected output relationships are described in [`docs/reproducibility.md`](docs/reproducibility.md).

## Figure 1

Figure 1 uses three ArtBench images that cannot be redistributed here. Place the files listed in `data/README.md` in one directory and run:

```bash
python scripts/build_figure1.py --image-dir /path/to/figure1_images
```

## Data and licensing

The source images remain subject to the terms of their original collections and ArtBench-10. This repository does not grant rights to those images. Code in this repository is released under the MIT License; numerical result tables and generated plots are provided for scholarly verification.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). 

## Contact

Andy Domínguez-Monterroza 
Department of Mathematics, Faculty of Sciences  
Pontificia Universidad Javeriana, Bogotá, Colombia

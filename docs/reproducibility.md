# Reproducibility guide

## Two reproduction levels

The compact route rebuilds the displayed outputs from the numerical results committed in `results/`. It is intended for checking figures, labels, and tabulated values without distributing the source corpus.

The full route starts from ArtBench-10 images and artist metadata. It reconstructs the manifest, extracts every representation, obtains one held-out prediction per artist-linked image, estimates paired artist-bootstrap intervals, and performs the artist-centroid analyses.

## Analysis sequence

1. `prepare_artbench_manifest.py` resolves the ImageFolder structure and attaches artist metadata.
2. `extract_representations.py` resizes each image to a longest side of 256 pixels and extracts B90, K40, and the additional structure-tensor features in G44.
3. `extract_ordinal_patterns.py` appends OP75 and validates the vectorised implementation against `ordpy` on tie-rich synthetic fields.
4. `run_artist_disjoint_evaluation.py` applies the common five-fold artist partition and fixed linear support-vector classifier to every representation.
5. `run_geometric_organisation.py` analyses style separation among artist centroids and partitions descriptor-wise variation into style, artist-within-style, and painting-level components.
6. `run_source_sensitivity.py` repeats the centroid analysis after source- and style-specific exclusions.
7. `build_paper_outputs.py` converts the compact result files into publication figures and tables.

## Expected compact outputs

| File | Content |
|---|---|
| `results/classification_metrics.csv` | Macro-F1 and accuracy for each representation and corpus |
| `results/classification_contrasts.csv` | Paired macro-F1 increments, bootstrap intervals, and adjusted probabilities |
| `results/geometric_organisation.csv` | Scale-specific centroid separation and variance decomposition |

The tests verify descriptor dimensionality, ordinal-pattern normalisation, key manuscript values, and the absence of internal development terminology in public-facing text and code.

## Numerical provenance

The compact CSV files were copied from the final full-corpus run without rounding. Displayed values are rounded only by the plotting and LaTeX-table code. Random seeds, fold count, bootstrap count, permutation count, smoothing scales, and classifier settings are recorded in `analysis_specification.json`.

One B90+G44 fold reached the default iteration limit. `check_convergence.py` reproduces the longer-fit diagnostic from the saved prediction checkpoint. The diagnostic changed two of 11,692 predictions and changed fold macro-F1 by approximately -0.000115.

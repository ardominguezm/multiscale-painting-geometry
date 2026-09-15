# Input data

The original ArtBench-10 images are not included in this repository. Full reproduction requires an ImageFolder-style directory with `train/` and `test/` subdirectories, each containing one directory per style.

```text
artbench/
  train/
    art_nouveau/
    baroque/
    ...
  test/
    art_nouveau/
    baroque/
    ...
```

The metadata CSV supplied to `prepare_artbench_manifest.py` must contain an image filename and artist identifier. Columns for style, split, and source are accepted when available. The script recognises common column names such as `filename`, `artist`, `style`, `split`, and `source` and writes a resolved manifest containing:

| Column | Meaning |
|---|---|
| `split` | Distributed train/test directory |
| `style` | Normalised style label |
| `artist` | Artist identifier used to define disjoint folds |
| `source` | Source collection when available |
| `filename` | Image basename |
| `path` | Local path to the image |

The reported classification excluded 38 images without usable artist identifiers, leaving 59,962 paintings by 2,126 artists. The WikiArt-8 control excludes `surrealism` and `ukiyo_e`, leaving 48,000 paintings by 1,701 artists. These counts are checked during comparison with the included results.

Figure 1 additionally requires these three files in a common directory:

- `anita-malfatti_fernanda-de-castro-1922.jpg`
- `amrita-sher-gil_tribal-women-1938.jpg`
- `abraham-manievich_the-yellow-house.jpg`

No source image should be committed unless its redistribution rights have been verified.

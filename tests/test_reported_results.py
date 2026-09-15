from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_key_classification_results():
    metrics = pd.read_csv(ROOT / "results" / "classification_metrics.csv")
    lookup = metrics.set_index(["dataset", "representation"])["macro_f1"]
    assert np.isclose(lookup["artbench10_all", "B90"], 0.35039378353696515)
    assert np.isclose(lookup["artbench10_all", "B90_OP75_K40"], 0.4005180259506086)
    assert np.isclose(lookup["artbench10_wikiart8", "B90_OP75_K40"], 0.3795471571142722)


def test_directional_intervals_exclude_zero():
    contrasts = pd.read_csv(ROOT / "results" / "classification_contrasts.csv")
    assert len(contrasts) == 8
    assert (contrasts["ci_low"] > 0).all()
    assert (contrasts["ci_high"] > contrasts["ci_low"]).all()
    assert np.isclose(contrasts["q_bh"], 1 / 5001).all()


def test_intermediate_centroid_peak():
    organisation = pd.read_csv(ROOT / "results" / "geometric_organisation.csv")
    for _, subset in organisation.groupby("dataset"):
        peak = subset.loc[subset["eta2_style_artist_centroids"].idxmax(), "sigma_ref"]
        assert peak == 2.0

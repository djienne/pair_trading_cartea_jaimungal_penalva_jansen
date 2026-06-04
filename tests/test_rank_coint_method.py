"""C8: the ADF ranker uses the OLS calibrator (Bayesian leaves adf_pvalue NaN)."""
import numpy as np
import pandas as pd

import rank_coint_pairs
import coint_calibrate
from rank_coint_pairs import score_coint


def test_ranker_uses_ols_calibrator():
    # rank_coint_pairs must bind calibrate_pair to the OLS implementation, not the Bayesian one.
    assert rank_coint_pairs.calibrate_pair is coint_calibrate.calibrate_pair


def test_score_coint_nonempty_with_real_pvalues():
    df = pd.DataFrame(
        {
            "adf_pvalue": [0.01, 0.2, np.nan, 0.03],
            "adf_pass": [True, False, False, True],
        }
    )
    metrics = score_coint(df)
    assert metrics  # non-empty
    assert metrics["n_windows"] == 3
    assert metrics["min_p"] == 0.01


def test_score_coint_empty_when_all_nan():
    # Mirrors the Bayesian output (adf_pvalue all NaN) -> would yield no metrics.
    df = pd.DataFrame({"adf_pvalue": [np.nan, np.nan], "adf_pass": [False, False]})
    assert score_coint(df) == {}

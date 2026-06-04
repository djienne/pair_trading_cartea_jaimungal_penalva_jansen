"""A2: the Bayesian coint output keeps ALL rows (leading NaN) for parity with the OLS path.

Stubs out the PyMC fit so the test is fast and deterministic.
"""
import numpy as np
import pandas as pd

import bayesian_coint
from rolling_bayesian_hedge import RollingBayesHedge
from conftest import make_config, write_symbol_data


def test_bayesian_output_aligns_with_price_frame(tmp_path, pair, monkeypatch):
    n = 20
    window = 5
    config = make_config(tmp_path, rolling_window_days=window)
    # Cointegration-method-agnostic synthetic prices.
    write_symbol_data(config, "AAA", 100.0 + np.arange(n))
    write_symbol_data(config, "BBB", 50.0 + 0.5 * np.arange(n))

    def fake_hedge(s1, s2, window, **kwargs):
        idx = s1.index
        a = pd.Series(np.nan, index=idx, name="alpha_hat")
        a.iloc[window:] = 1.0
        b = pd.Series(np.nan, index=idx, name="beta_hat")
        b.iloc[window:] = 2.0
        s = pd.Series(np.nan, index=idx, name="sigma_obs_hat")
        s.iloc[window:] = 0.1
        # Mirror the real return: ffill().dropna() drops the leading NaN block.
        return RollingBayesHedge(
            alpha_hat=a.dropna(), beta_hat=b.dropna(), sigma_obs_hat=s.dropna()
        )

    monkeypatch.setattr(bayesian_coint, "rolling_bayesian_rw_hedge_ratio", fake_hedge)

    results = bayesian_coint.calibrate_pair(pair, config, show_progress=False)

    # Row count matches the full merged price frame (no rows dropped, unlike the old code).
    assert len(results) == n
    # The first `window` rows are NaN (parity with the OLS calibrator's leading block).
    assert results["alpha"].iloc[:window].isna().all()
    assert results["beta"].iloc[:window].isna().all()
    # Everything from `window` onward is populated.
    assert results["alpha"].iloc[window:].notna().all()
    assert results["beta"].iloc[window:].notna().all()

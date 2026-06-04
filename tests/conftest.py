"""Shared test fixtures/helpers.

These build tiny synthetic feather files on disk so the backtest/calibration code can be
exercised without network access or PyMC.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils  # noqa: E402


def make_config(tmp_path, **overrides):
    """A minimal config pointing all data dirs at a temp directory."""
    cfg = {
        "data_dir": str(tmp_path),
        "feather_dir": str(tmp_path / "feather"),
        "intermediate_dir": str(tmp_path / "intermediate"),
        "output_dir": str(tmp_path / "output"),
        "candle_interval": "1d",
        "rolling_window_days": 5,
        "start_equity": 1000.0,
        "fee_rate": 0.001,
        "sizing_mode": "gross",
        "target_gross_leverage": 1.0,
        "capital_per_trade_frac": 0.5,
        "execution_lag_bars": 0,
        "flip_signals": False,
    }
    cfg.update(overrides)
    return cfg


def _dt_index(n):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    idx.name = utils.TIME_COLUMN
    return idx


def write_pair_data(config, pair, y, x, beta, epsilon, lower, upper, mu):
    """Write synthetic coint + bands feathers for `pair` into the config's dirs."""
    n = len(y)
    idx = _dt_index(n)
    coint = pd.DataFrame(
        {
            "y_close": np.asarray(y, float),
            "x_close": np.asarray(x, float),
            "alpha": np.zeros(n),
            "beta": np.asarray(beta, float),
            "epsilon": np.asarray(epsilon, float),
        },
        index=idx,
    )
    bands = pd.DataFrame(
        {
            "lower": np.asarray(lower, float),
            "upper": np.asarray(upper, float),
            "mu": np.asarray(mu, float),
        },
        index=idx,
    )
    utils.save_pair_data(coint, pair, config, "coint", verbose=False)
    utils.save_pair_data(bands, pair, config, "bands", verbose=False)


def write_symbol_data(config, symbol, closes):
    """Write a synthetic per-symbol kline feather (used by the Bayesian calibrator path)."""
    n = len(closes)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    open_time = (idx.asi8 // 10**6).astype("int64")  # ns -> ms epoch
    df = pd.DataFrame(
        {
            "open_time": open_time,
            "open_time_dt": idx,
            "close": np.asarray(closes, float),
        }
    )
    feather_dir = config["feather_dir"]
    os.makedirs(feather_dir, exist_ok=True)
    interval = config.get("candle_interval", "1d")
    df.to_feather(os.path.join(feather_dir, f"{symbol}_{interval}.feather"))


@pytest.fixture
def pair():
    return {"y_symbol": "AAA", "x_symbol": "BBB", "name": "AAA-BBB"}

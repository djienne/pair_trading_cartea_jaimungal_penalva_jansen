"""A1: strategy_return is guarded against a non-positive prior equity."""
import numpy as np

from backtest import backtest_pair
from conftest import make_config, write_pair_data


def test_returns_finite_when_equity_goes_negative(tmp_path, pair):
    n = 8
    # Enter long heavily at i=2, then Y price crashes at i=3 while still holding -> equity < 0.
    y = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    x = np.full(n, 50.0)
    beta = np.full(n, 2.0)
    epsilon = np.array([0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])  # stays below mu -> holds
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    mu = np.zeros(n)

    config = make_config(tmp_path, sizing_mode="gross", target_gross_leverage=50.0)
    write_pair_data(config, pair, y, x, beta, epsilon, lower, upper, mu)

    df = backtest_pair(pair, config)
    eq = df["equity"].values
    ret = df["strategy_return"].values

    # The scenario really does drive equity through zero.
    assert eq.min() < 0
    # Returns are always finite (no inf/NaN) despite the sign flip.
    assert np.isfinite(ret).all()
    # The bar following a non-positive prior equity yields a guarded 0.0 return.
    neg_prev = np.where(eq[:-1] <= 1e-9)[0]
    assert len(neg_prev) > 0
    for i in neg_prev:
        assert ret[i + 1] == 0.0

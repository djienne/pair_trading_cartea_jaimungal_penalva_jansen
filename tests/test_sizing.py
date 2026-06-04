"""A1: capital-normalized sizing in backtest_pair."""
import numpy as np

from backtest import backtest_pair
from conftest import make_config, write_pair_data


def _one_round_trip_inputs(n=8):
    # Constant prices; epsilon dips below the lower band (enter long at i=2) then crosses mu (exit at i=5).
    y = np.full(n, 100.0)
    x = np.full(n, 50.0)
    beta = np.full(n, 2.0)
    epsilon = np.array([0.0, 0.0, -2.0, -2.0, -2.0, 0.5, 0.0, 0.0])
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    mu = np.zeros(n)
    return y, x, beta, epsilon, lower, upper, mu


def test_gross_sizing_and_hedge(tmp_path, pair):
    y, x, beta, epsilon, lower, upper, mu = _one_round_trip_inputs()
    config = make_config(tmp_path, sizing_mode="gross", target_gross_leverage=1.0)
    write_pair_data(config, pair, y, x, beta, epsilon, lower, upper, mu)

    df = backtest_pair(pair, config)
    m1 = df["m1"].values
    m2 = df["m2"].values
    eq = df["equity"].values

    # Entry happens at index 2 (epsilon -2 <= lower -1).
    assert m1[2] > 0 and m2[2] < 0

    # Gross notional == target_gross_leverage * prior-bar equity.
    gross = abs(m1[2] * y[2]) + abs(m2[2] * x[2])
    assert abs(gross - 1.0 * eq[1]) <= 1e-6 * eq[1]

    # Legs respect the hedge ratio: m2 == -beta * m1.
    assert abs(m2[2] + beta[2] * m1[2]) < 1e-9

    # At leverage 1 with constant prices equity stays positive and returns are finite.
    assert (eq > 0).all()
    assert np.isfinite(df["strategy_return"].values).all()


def test_legacy_unit_sizing_reproduces_one_unit(tmp_path, pair):
    y, x, beta, epsilon, lower, upper, mu = _one_round_trip_inputs()
    config = make_config(tmp_path, sizing_mode="legacy_unit")
    write_pair_data(config, pair, y, x, beta, epsilon, lower, upper, mu)

    df = backtest_pair(pair, config)
    # Old behaviour: exactly 1 unit of Y, -beta units of X.
    assert abs(df["m1"].values[2] - 1.0) < 1e-12
    assert abs(df["m2"].values[2] - (-2.0)) < 1e-12


def test_gross_scales_with_leverage(tmp_path, pair):
    y, x, beta, epsilon, lower, upper, mu = _one_round_trip_inputs()
    config = make_config(tmp_path, sizing_mode="gross", target_gross_leverage=2.0)
    write_pair_data(config, pair, y, x, beta, epsilon, lower, upper, mu)

    df = backtest_pair(pair, config)
    m1 = df["m1"].values
    m2 = df["m2"].values
    eq = df["equity"].values
    gross = abs(m1[2] * y[2]) + abs(m2[2] * x[2])
    assert abs(gross - 2.0 * eq[1]) <= 1e-6 * eq[1]

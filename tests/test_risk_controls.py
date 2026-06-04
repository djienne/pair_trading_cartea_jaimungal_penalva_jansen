"""Opt-in risk controls: equity-drawdown stop-loss and daily rehedge."""
import numpy as np

from backtest import backtest_pair
from conftest import make_config, write_pair_data


def _crash_inputs(n=8):
    # Enter long at i=2; the short X leg then blows up as x ramps up while the position is held.
    y = np.full(n, 100.0)
    x = np.array([50, 50, 50, 80, 120, 160, 200, 240], float)
    beta = np.full(n, 2.0)
    epsilon = np.array([0, 0, -2, -2, -2, -2, -2, -2], float)  # stays below mu -> holds
    lower = np.full(n, -1.0)
    upper = np.full(n, 1.0)
    mu = np.zeros(n)
    return y, x, beta, epsilon, lower, upper, mu


def test_stop_loss_force_flats_and_preserves_capital(tmp_path, pair):
    y, x, beta, eps, lo, up, mu = _crash_inputs()

    cfg = make_config(tmp_path / "sl", sizing_mode="gross", target_gross_leverage=1.0, stop_loss_frac=0.2)
    write_pair_data(cfg, pair, y, x, beta, eps, lo, up, mu)
    df = backtest_pair(pair, cfg)
    pos = df["position"].values
    eq_stop = df["equity"].values

    assert pos[2] == 1   # entered long
    assert pos[3] == 0   # stop-loss force-flats at i=3 (x 50->80, >20% loss)
    # (The persistent signal re-enters on later bars -- a stop has no cooldown -- which is fine;
    # what matters is that each trade's loss is bounded, so the drawdown is far shallower.)

    # Without the stop, the held short leg drives equity deeply negative.
    cfg2 = make_config(tmp_path / "nosl", sizing_mode="gross", target_gross_leverage=1.0)
    write_pair_data(cfg2, pair, y, x, beta, eps, lo, up, mu)
    eq_nostop = backtest_pair(pair, cfg2)["equity"].values

    assert eq_nostop[-1] < 0                     # un-stopped position blows up
    assert eq_stop.min() > eq_nostop.min()       # stop bounds the drawdown
    assert eq_stop[-1] > eq_nostop[-1]
    assert eq_stop[-1] > 0                        # capital preserved


def test_rehedge_tracks_current_beta(tmp_path, pair):
    n = 8
    y = np.full(n, 100.0)
    x = np.full(n, 50.0)
    beta = np.array([2, 2, 2, 3, 4, 2, 2, 2], float)  # hedge ratio drifts while holding
    eps = np.array([0, 0, -2, -2, -2, -2, -2, -2], float)
    lo = np.full(n, -1.0)
    up = np.full(n, 1.0)
    mu = np.zeros(n)

    cfg = make_config(tmp_path / "rh", target_gross_leverage=1.0, rehedge_daily=True)
    write_pair_data(cfg, pair, y, x, beta, eps, lo, up, mu)
    df = backtest_pair(pair, cfg)
    m1 = df["m1"].values
    m2 = df["m2"].values
    turn = df["turnover"].values

    assert m1[2] == 5.0  # entry sizing (gross 1.0x: q = 1000 / (100 + 2*50))
    # The X leg is refreshed to -beta_t * m1 each bar.
    assert abs(m2[3] - (-beta[3] * m1[3])) < 1e-9
    assert abs(m2[4] - (-beta[4] * m1[4])) < 1e-9
    assert turn[3] > 0  # rehedging incurs turnover/fees

    # With rehedge off, the X leg is held constant while in the position.
    cfg2 = make_config(tmp_path / "norh", target_gross_leverage=1.0, rehedge_daily=False)
    write_pair_data(cfg2, pair, y, x, beta, eps, lo, up, mu)
    df2 = backtest_pair(pair, cfg2)
    assert df2["m2"].values[3] == df2["m2"].values[2]
    assert df2["turnover"].values[3] == 0.0

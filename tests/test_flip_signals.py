"""flip_signals swaps the entry bands (and inverts exits), so it changes behaviour."""
import numpy as np

from backtest import backtest_pair
from conftest import make_config, write_pair_data


def _inputs(n=6):
    # Asymmetric bands so the two configs diverge clearly.
    # epsilon = 1 sits inside [lower=-1, upper=3] (no normal entry) but below the flipped
    # long-entry threshold (z <= upper=3), so flip enters long while non-flip stays flat.
    y = np.full(n, 100.0)
    x = np.full(n, 50.0)
    beta = np.full(n, 2.0)
    epsilon = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lower = np.full(n, -1.0)
    upper = np.full(n, 3.0)
    mu = np.zeros(n)
    return y, x, beta, epsilon, lower, upper, mu


def test_flip_changes_positions(tmp_path, pair):
    y, x, beta, epsilon, lower, upper, mu = _inputs()

    cfg_off = make_config(tmp_path / "off", flip_signals=False)
    write_pair_data(cfg_off, pair, y, x, beta, epsilon, lower, upper, mu)
    pos_off = backtest_pair(pair, cfg_off)["position"].values

    cfg_on = make_config(tmp_path / "on", flip_signals=True)
    write_pair_data(cfg_on, pair, y, x, beta, epsilon, lower, upper, mu)
    pos_on = backtest_pair(pair, cfg_on)["position"].values

    # Non-flip never enters (epsilon stays strictly inside the bands).
    assert np.all(pos_off == 0)
    # Flip enters long from i=1 onward (swapped bands make z <= upper a long entry).
    assert pos_on[1] == 1
    assert np.any(pos_off != pos_on)

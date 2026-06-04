"""B7: shared periods_per_year / annualized_sharpe helpers."""
import numpy as np

from utils import periods_per_year, annualized_sharpe


def test_periods_per_year():
    assert periods_per_year("1d") == 365
    assert periods_per_year("1h") == 8760
    assert periods_per_year("3d") == 121  # 365 // 3
    assert periods_per_year("1d", override=252) == 252
    # Unknown strings fall back to the daily-crypto default.
    assert periods_per_year("weird") == 365


def test_annualized_sharpe_edge_cases():
    assert annualized_sharpe([], 365) == 0.0
    assert annualized_sharpe([0.01, 0.01, 0.01], 365) == 0.0  # zero variance


def test_annualized_sharpe_value():
    r = np.array([0.01, -0.005, 0.012, 0.003, -0.002])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(365)
    assert abs(annualized_sharpe(r, 365) - expected) < 1e-9

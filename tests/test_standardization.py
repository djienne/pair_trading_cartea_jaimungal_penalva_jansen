"""A3: per-window standardization is a faithful (causal) reparametrization.

Standardizing a window with its own mean/std, fitting OLS in standardized space, then
de-standardizing must recover the raw-slice OLS coefficients exactly.
"""
import numpy as np

from rolling_bayesian_hedge import _standardize_window, _to_orig_coeffs, _to_std_coeffs


def _ols(y, x):
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def test_destandardize_recovers_raw_ols():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 3.0, 200)
    y = 2.5 + 1.7 * x + rng.normal(0.0, 0.5, 200)

    a_raw, b_raw = _ols(y, x)

    ym, ys, xm, xs = _standardize_window(y, x)
    y_s = (y - ym) / ys
    x_s = (x - xm) / xs
    a_s, b_s = _ols(y_s, x_s)

    a_o, b_o = _to_orig_coeffs(a_s, b_s, ym, ys, xm, xs)
    assert abs(a_o - a_raw) < 1e-6
    assert abs(b_o - b_raw) < 1e-6


def test_std_orig_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(-2.0, 4.0, 50)
    y = -1.0 + 0.3 * x + rng.normal(0.0, 1.0, 50)
    ym, ys, xm, xs = _standardize_window(y, x)

    a, b = 3.3, -0.7
    a_s, b_s = _to_std_coeffs(a, b, ym, ys, xm, xs)
    a_back, b_back = _to_orig_coeffs(a_s, b_s, ym, ys, xm, xs)
    assert abs(a_back - a) < 1e-9
    assert abs(b_back - b) < 1e-9

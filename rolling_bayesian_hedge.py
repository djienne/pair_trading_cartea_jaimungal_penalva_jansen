from __future__ import annotations

import multiprocessing as mp
import warnings
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd

# Suppress PyTensor and PyMC noise
warnings.filterwarnings("ignore", module="pytensor.link.c.cmodule")

# Configure logging to be quiet
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)


# ============================================================
# Rolling Bayesian Random-Walk regression (daily update)
# ============================================================

def _import_pymc():
    try:
        import pymc as pm
        return pm
    except Exception:
        import pymc3 as pm
        return pm

Inference = Literal["advi", "nuts"]

@dataclass
class RollingBayesHedge:
    alpha_hat: pd.Series
    beta_hat: pd.Series
    sigma_obs_hat: pd.Series

def _fit_rw_window_pymc(
    y: np.ndarray,
    x: np.ndarray,
    inference: Inference = "advi",
    advi_steps: int = 1500,
    draws: int = 300,
    tune: int = 300,
    target_accept: float = 0.9,
    warm_start: bool = True,
    prev_alpha0: Optional[float] = None,
    prev_beta0: Optional[float] = None,
    random_seed: int = 7,
    use_ols_init: bool = True,
) -> Tuple[float, float, float]:
    pm = _import_pymc()
    T = len(y)

    initvals = None
    if use_ols_init:
        # OLS init values for alpha0/beta0 (init only; priors unchanged)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_centered = x - x_mean
        denom = float(np.sum(x_centered ** 2))
        if denom > 1e-12 and np.isfinite(denom):
            beta_ols = float(np.sum(x_centered * (y - y_mean)) / denom)
        else:
            beta_ols = 0.0
        alpha_ols = float(y_mean - beta_ols * x_mean)
        if np.isfinite(alpha_ols) and np.isfinite(beta_ols):
            initvals = {"alpha0": alpha_ols, "beta0": beta_ols}

    mu_a0 = float(prev_alpha0) if (warm_start and prev_alpha0 is not None) else 0.0
    mu_b0 = float(prev_beta0) if (warm_start and prev_beta0 is not None) else 0.0
    s0 = 1.0 

    with pm.Model() as model:
        sigma_alpha = pm.Exponential("sigma_alpha", 50.0)
        sigma_beta = pm.Exponential("sigma_beta", 50.0)

        alpha0 = pm.Normal("alpha0", mu=mu_a0, sigma=s0)
        beta0 = pm.Normal("beta0", mu=mu_b0, sigma=s0)

        eta_a = pm.Normal("eta_a", mu=0.0, sigma=sigma_alpha, shape=T-1)
        eta_b = pm.Normal("eta_b", mu=0.0, sigma=sigma_beta, shape=T-1)

        alpha_path = pm.Deterministic("alpha", pm.math.concatenate([[alpha0], alpha0 + pm.math.cumsum(eta_a)]))
        beta_path  = pm.Deterministic("beta",  pm.math.concatenate([[beta0],  beta0  + pm.math.cumsum(eta_b)]))

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)

        mu = alpha_path + beta_path * x
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y)

        if inference == "nuts":
            sample_kwargs = dict(
                draws=draws, tune=tune, target_accept=target_accept,
                chains=2, cores=1, random_seed=random_seed, progressbar=False,
            )
            if initvals is not None:
                try:
                    trace = pm.sample(**sample_kwargs, initvals=initvals)
                except TypeError:
                    trace = pm.sample(**sample_kwargs, start=initvals)
            else:
                trace = pm.sample(**sample_kwargs)
        else:
            fit_kwargs = dict(n=advi_steps, method="advi", random_seed=random_seed, progressbar=False)
            if initvals is not None:
                try:
                    approx = pm.fit(**fit_kwargs, start=initvals)
                except TypeError:
                    approx = pm.fit(**fit_kwargs)
            else:
                approx = pm.fit(**fit_kwargs)
            trace = approx.sample(draws=draws, random_seed=random_seed)

    try:
        alpha_last = float(trace.posterior["alpha"].sel(alpha_dim_0=T-1).mean(("chain", "draw")).values)
        beta_last  = float(trace.posterior["beta"].sel(beta_dim_0=T-1).mean(("chain", "draw")).values)
        sig_hat    = float(trace.posterior["sigma_obs"].mean(("chain", "draw")).values)
    except Exception:
        alpha_last = float(np.mean(trace["alpha"][:, T-1]))
        beta_last  = float(np.mean(trace["beta"][:, T-1]))
        sig_hat    = float(np.mean(trace["sigma_obs"]))

    return alpha_last, beta_last, sig_hat

# ------------------------------------------------------------
# Causal (per-window) standardization helpers
# ------------------------------------------------------------
# The model priors are fixed in standardized space, so the *scale* used to standardize must not
# leak information from outside the current window. We therefore standardize each window with its
# OWN mean/std (not the full series), fit, then de-standardize the coefficients back to raw units.

def _standardize_window(y_win: np.ndarray, x_win: np.ndarray) -> Tuple[float, float, float, float]:
    ym = float(np.mean(y_win))
    ys = float(np.std(y_win, ddof=1) + 1e-12)
    xm = float(np.mean(x_win))
    xs = float(np.std(x_win, ddof=1) + 1e-12)
    return ym, ys, xm, xs


def _to_orig_coeffs(alpha_s: float, beta_s: float, ym: float, ys: float, xm: float, xs: float) -> Tuple[float, float]:
    beta_orig = (ys / xs) * beta_s
    alpha_orig = ym + ys * alpha_s - beta_orig * xm
    return alpha_orig, beta_orig


def _to_std_coeffs(alpha_orig: float, beta_orig: float, ym: float, ys: float, xm: float, xs: float) -> Tuple[float, float]:
    # Inverse of _to_orig_coeffs: express raw-unit coefficients in this window's standardized space.
    beta_s = beta_orig * xs / ys
    alpha_s = (alpha_orig + beta_orig * xm - ym) / ys
    return alpha_s, beta_s


def _fit_window_destd(
    y_win: np.ndarray,
    x_win: np.ndarray,
    inference: "Inference" = "advi",
    advi_steps: int = 1500,
    draws: int = 300,
    tune: int = 300,
    target_accept: float = 0.9,
    warm_start: bool = False,
    prev_alpha_orig: Optional[float] = None,
    prev_beta_orig: Optional[float] = None,
    random_seed: int = 7,
    use_ols_init: bool = True,
) -> Tuple[float, float, float]:
    """Standardize one RAW window causally, fit, and return de-standardized (alpha, beta, sigma)."""
    ym, ys, xm, xs = _standardize_window(y_win, x_win)
    y_s = (y_win - ym) / ys
    x_s = (x_win - xm) / xs

    if warm_start and prev_alpha_orig is not None and prev_beta_orig is not None:
        prev_a_s, prev_b_s = _to_std_coeffs(prev_alpha_orig, prev_beta_orig, ym, ys, xm, xs)
    else:
        prev_a_s, prev_b_s = None, None

    a_s, b_s, sig_s = _fit_rw_window_pymc(
        y_s, x_s, inference=inference, advi_steps=advi_steps, draws=draws, tune=tune,
        target_accept=target_accept, warm_start=warm_start, prev_alpha0=prev_a_s,
        prev_beta0=prev_b_s, random_seed=random_seed, use_ols_init=use_ols_init,
    )
    alpha_orig, beta_orig = _to_orig_coeffs(a_s, b_s, ym, ys, xm, xs)
    sigma_orig = ys * sig_s  # de-standardize observation noise back to raw y units
    return alpha_orig, beta_orig, sigma_orig


def _fit_single_window_job(args) -> Tuple[int, float, float, float]:
    # Receives a RAW (unstandardized) window; standardization happens causally inside the fit.
    # warm_start is always False here so parallel results are independent of process scheduling.
    (i, y_win, x_win, inference, advi_steps, draws, tune, target_accept, random_seed, use_ols_init) = args
    try:
        a, b, s = _fit_window_destd(
            y_win, x_win, inference=inference, advi_steps=advi_steps,
            draws=draws, tune=tune, target_accept=target_accept, warm_start=False,
            random_seed=random_seed, use_ols_init=use_ols_init,
        )
        return i, a, b, s
    except Exception:
        return i, np.nan, np.nan, np.nan

def rolling_bayesian_rw_hedge_ratio(
    s1: pd.Series,
    s2: pd.Series,
    window: int = 252,
    inference: Inference = "advi",
    advi_steps: int = 1500,
    draws: int = 300,
    tune: int = 300,
    target_accept: float = 0.9,
    update_every: int = 1,
    warm_start: bool = True,
    random_seed: int = 7,
    use_ols_init: bool = True,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> RollingBayesHedge:
    s1 = s1.dropna()
    s2 = s2.dropna()
    idx = s1.index.intersection(s2.index)
    s1 = s1.loc[idx].astype(float)
    s2 = s2.loc[idx].astype(float)

    y, x = s1.values, s2.values
    T = len(idx)
    if T <= window + 2:
        raise ValueError("Not enough data.")

    # NOTE: no full-series standardization here. Each window is standardized causally inside
    # _fit_window_destd using only that window's own statistics (see helpers above).

    alpha_out = pd.Series(index=idx, dtype=float, name="alpha_hat")
    beta_out  = pd.Series(index=idx, dtype=float, name="beta_hat")
    sig_out   = pd.Series(index=idx, dtype=float, name="sigma_obs_hat")

    indices = [i for i in range(window, T) if (i - window) % update_every == 0]

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Warm start chains each window's prior to the previous window's posterior, which is inherently
    # sequential. Parallelism is only safe (and deterministic) when warm_start is off; otherwise
    # results would depend on process scheduling. So disable parallelism when warm_start is on.
    use_parallel = (n_jobs > 1) and (not warm_start)
    if (n_jobs > 1) and warm_start and show_progress:
        print("[Bayesian] warm_start=True forces sequential fitting (n_jobs ignored) for determinism.")

    if use_parallel:
        tasks = [(i, y[i-window:i], x[i-window:i], inference, advi_steps, draws, tune, target_accept, random_seed, use_ols_init) for i in indices]
        results = []
        with mp.Pool(processes=n_jobs) as pool:
            if has_tqdm and show_progress:
                it = pool.imap_unordered(_fit_single_window_job, tasks)
                for res in tqdm(it, total=len(tasks), desc=f"Parallel Fit ({n_jobs} jobs)", leave=False):
                    results.append(res)
            else:
                results = pool.map(_fit_single_window_job, tasks)

        for i, alpha_orig, beta_orig, sig_hat in results:
            if not np.isnan(alpha_orig):
                alpha_out.iloc[i], beta_out.iloc[i], sig_out.iloc[i] = alpha_orig, beta_orig, sig_hat
    else:
        prev_a, prev_b = None, None
        it = tqdm(indices, desc="Sequential Fit", leave=False) if has_tqdm and show_progress else indices
        for i in it:
            try:
                alpha_orig, beta_orig, sig_hat = _fit_window_destd(
                    y[i-window:i], x[i-window:i], inference=inference, advi_steps=advi_steps,
                    draws=draws, tune=tune, target_accept=target_accept, warm_start=warm_start,
                    prev_alpha_orig=prev_a, prev_beta_orig=prev_b,
                    random_seed=random_seed, use_ols_init=use_ols_init,
                )
            except Exception:
                alpha_orig, beta_orig, sig_hat = np.nan, np.nan, np.nan
            if not np.isnan(alpha_orig):
                prev_a, prev_b = alpha_orig, beta_orig
                alpha_out.iloc[i], beta_out.iloc[i], sig_out.iloc[i] = alpha_orig, beta_orig, sig_hat

    return RollingBayesHedge(alpha_hat=alpha_out.ffill().dropna(), beta_hat=beta_out.ffill().dropna(), sigma_obs_hat=sig_out.ffill().dropna())

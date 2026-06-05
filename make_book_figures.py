"""Regenerate the 8 figures of the pairs-trading / stat-arb chapter (Figs 11.1-11.8) from this
repo's own models, and write each as a vector PDF into ``figures/`` for inclusion in
``pair_trading.tex``.

Sources:
  * 11.1            - a real co-integrated crypto pair from the repo's own data (OLS hedge).
  * 11.2 / 11.3     - Ornstein-Uhlenbeck simulations + the naive standard-deviation band strategy.
  * 11.4 / 11.5 / 11.6 - the Cartea-Jaimungal-Penalva optimal entry/exit value functions and
                        trigger levels (``book_opt.OUOpt``; same math as ``band_calc.CointOpti``,
                        cross-validated against the trigger table printed in the chapter).
  * 11.7 / 11.8     - the Section-12 three-asset co-integrated *short-term-alpha* model and its
                        explicit optimal investment (eq. 11.16).

Deterministic: every random draw uses a fixed seed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from book_opt import OUOpt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ======================================================================================
# OU simulation + naive band strategy (Figs 11.2, 11.3)
# ======================================================================================
def simulate_ou(kappa, theta, sigma, T, n_steps, rng, eps0=None):
    """Exact OU discretisation. Returns (t, eps) with eps[0]=eps0 (default theta)."""
    dt = T / n_steps
    eps = np.empty(n_steps + 1)
    eps[0] = theta if eps0 is None else eps0
    a = np.exp(-kappa * dt)
    var = sigma**2 / (2.0 * kappa) * (1.0 - a**2)
    sd = np.sqrt(var)
    z = rng.standard_normal(n_steps)
    for i in range(n_steps):
        eps[i + 1] = theta + a * (eps[i] - theta) + sd * z[i]
    t = np.linspace(0.0, T, n_steps + 1)
    return t, eps


def naive_band_strategy(eps, theta, band, exit_thr):
    """Naive strategy: enter at +/- band, exit near the mean. Returns (inventory, book_value).

    inventory in {-1,0,+1}; book value = cash + inventory * eps (one unit of the portfolio).
    """
    n = len(eps)
    inv = np.zeros(n)
    bv = np.zeros(n)
    cash = 0.0
    pos = 0
    for i in range(n):
        e = eps[i]
        if pos == 0:
            if e <= theta - band:
                pos = 1
                cash -= e          # buy 1 unit of the portfolio
            elif e >= theta + band:
                pos = -1
                cash += e          # sell 1 unit of the portfolio
        elif pos == 1:
            if e >= theta - exit_thr:   # reverted up to near the mean
                cash += e
                pos = 0
        elif pos == -1:
            if e <= theta + exit_thr:   # reverted down to near the mean
                cash -= e
                pos = 0
        inv[i] = pos
        bv[i] = cash + pos * e
    return inv, bv


# ======================================================================================
# Fig 11.1 - a real co-integrated crypto pair (substitute for INTC / SMH)
# ======================================================================================
def fig_11_1():
    y = pd.read_feather("data/feather/LTCUSDT_1d.feather")[["open_time_dt", "close"]]
    x = pd.read_feather("data/feather/UNIUSDT_1d.feather")[["open_time_dt", "close"]]
    m = y.merge(x, on="open_time_dt", suffixes=("_y", "_x")).dropna()
    # a representative ~10-month window
    m = m.iloc[-300:-20].reset_index(drop=True)
    py = m["close_y"].values
    px = m["close_x"].values
    dt = m["open_time_dt"].values

    # OLS hedge: py = alpha + beta * px ; residual = co-integration factor.
    beta, alpha = np.polyfit(px, py, 1)
    eps = py - (alpha + beta * px)
    mu, sd = eps.mean(), eps.std()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(dt, py / py.mean(), label="LTC (scaled)", color="#1f77b4")
    ax[0].plot(dt, px / px.mean(), label="UNI (scaled)", color="#d62728")
    ax[0].set_title("(a) Midprices relative to their mean")
    ax[0].set_ylabel("price / mean price")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].tick_params(axis="x", labelrotation=30, labelsize=7)

    ax[1].plot(dt, eps, color="#2ca02c", label=r"$\varepsilon_t$")
    ax[1].axhline(mu, ls="--", color="k", lw=0.9, label="mean")
    ax[1].axhline(mu + 2 * sd, ls="-.", color="grey", lw=0.9, label=r"$\pm 2\sigma$")
    ax[1].axhline(mu - 2 * sd, ls="-.", color="grey", lw=0.9)
    ax[1].set_title("(b) Co-integration factor")
    ax[1].set_ylabel(r"$\varepsilon_t = P^{LTC}_t - (\alpha + \beta\,P^{UNI}_t)$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[1].tick_params(axis="x", labelrotation=30, labelsize=7)

    fig.suptitle(f"LTC & UNI (daily)  |  hedge  $\\beta$={beta:.2f},  $\\alpha$={alpha:.2f}",
                 fontsize=10)
    _save(fig, "fig_11_1.pdf")


# ======================================================================================
# Fig 11.2 - naive-band sample path: factor, inventory, book value
# ======================================================================================
def fig_11_2():
    kappa, theta, sigma, T = 6.0, 0.0, 0.5, 1.0
    n = 1000
    std_stat = sigma / np.sqrt(2 * kappa)
    band = 1.3 * std_stat
    exit_thr = 0.15 * std_stat
    # Pick a seed whose path triggers BOTH a long and a short, for a clearer illustration.
    for seed in range(80):
        rng = np.random.default_rng(seed)
        t, eps = simulate_ou(kappa, theta, sigma, T, n, rng)
        inv, bv = naive_band_strategy(eps, theta, band, exit_thr)
        if (inv == 1).any() and (inv == -1).any():
            break

    fig, ax = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True)
    ax[0].plot(t, eps, color="#1f77b4", lw=1.0)
    ax[0].axhline(theta, ls="--", color="k", lw=0.8)
    ax[0].axhline(theta + band, ls="-.", color="grey", lw=0.8)
    ax[0].axhline(theta - band, ls="-.", color="grey", lw=0.8)
    ax[0].set_ylabel(r"factor $\varepsilon_t$")
    ax[0].set_title(r"Naive band strategy (entry at $\pm 1.3\,\sigma_{stat}$, exit near the mean)")
    ax[0].grid(alpha=0.3)

    ax[1].step(t, inv, where="post", color="#d62728", lw=1.2)
    ax[1].set_ylabel("inventory")
    ax[1].set_yticks([-1, 0, 1])
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, bv, color="#2ca02c", lw=1.2)
    ax[2].set_ylabel("book value")
    ax[2].set_xlabel("t")
    ax[2].grid(alpha=0.3)
    _save(fig, "fig_11_2.pdf")


# ======================================================================================
# Fig 11.3 - naive-strategy P&L histograms across band sizes (Monte Carlo)
# ======================================================================================
def fig_11_3():
    kappa, theta, sigma, T = 8.0, 0.0, 0.5, 1.0
    n_steps, n_paths = 500, 10000
    std_stat = sigma / np.sqrt(2 * kappa)
    exit_thr = 0.1 * std_stat
    bands = [0.25, 0.5, 1.0, 2.0]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for ax, mult in zip(axes.ravel(), bands):
        band = mult * std_stat
        pnl = np.empty(n_paths)
        for p in range(n_paths):
            _, eps = simulate_ou(kappa, theta, sigma, T, n_steps, rng)
            _, bv = naive_band_strategy(eps, theta, band, exit_thr)
            pnl[p] = bv[-1]
        sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
        ax.hist(pnl, bins=60, color="#4c72b0", edgecolor="white", linewidth=0.2)
        ax.set_title(f"band = {mult:g} $\\times$ std.dev.   "
                     f"(mean {pnl.mean():.2f}, Sharpe {sharpe:.2f})", fontsize=9)
        ax.set_xlabel("P&L")
        ax.grid(alpha=0.3)
    fig.suptitle("Naive strategy: P&L over 10,000 simulated scenarios", fontsize=11)
    _save(fig, "fig_11_3.pdf")


# ======================================================================================
# Fig 11.4 - optimal EXIT value function H+ and triggers (vary kappa, rho)
# ======================================================================================
def fig_11_4():
    theta, sigma, c = 0.0, 0.5, 0.01
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) vary kappa at fixed rho
    rho = 0.05
    for kappa, col in zip([0.5, 1.0, 2.0], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        m = OUOpt(kappa, theta, sigma, rho, c)
        es = m.exit_long()
        grid = np.linspace(theta - 1.5, es + 0.4, 400)
        H = m.H_plus(grid, es)
        cont = grid < es
        ax[0].plot(grid[cont], H[cont], color=col, label=fr"$\kappa={kappa:g}$")
        ax[0].plot(grid[~cont], H[~cont], color=col, ls="--", lw=1.0)
        ax[0].plot(es, m.H_plus(np.array([es]), es)[0], "o", color="k", ms=4, zorder=5)
    ax[0].set_title(fr"(a) $H_+$, varying $\kappa$  ($\rho={rho}$)")
    ax[0].set_xlabel(r"$\varepsilon$")
    ax[0].set_ylabel(r"$H_+(\varepsilon)$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # (b) vary rho at fixed kappa
    kappa = 1.0
    for rho, col in zip([0.01, 0.1, 0.5], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        m = OUOpt(kappa, theta, sigma, rho, c)
        es = m.exit_long()
        grid = np.linspace(theta - 1.5, es + 0.4, 400)
        H = m.H_plus(grid, es)
        cont = grid < es
        ax[1].plot(grid[cont], H[cont], color=col, label=fr"$\rho={rho:g}$")
        ax[1].plot(grid[~cont], H[~cont], color=col, ls="--", lw=1.0)
        ax[1].plot(es, m.H_plus(np.array([es]), es)[0], "o", color="k", ms=4, zorder=5)
    ax[1].set_title(fr"(b) $H_+$, varying $\rho$  ($\kappa={kappa}$)")
    ax[1].set_xlabel(r"$\varepsilon$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Optimal exit: value function $H_+$ (solid=continue, dashed=exercise, "
                 "$\\bullet$=trigger $\\varepsilon^*$)", fontsize=10)
    _save(fig, "fig_11_4.pdf")


# ======================================================================================
# Fig 11.5 - optimal ENTRY value function G+ and triggers (vary kappa, rho)
# ======================================================================================
def fig_11_5():
    theta, sigma, c = 0.0, 0.5, 0.01
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    rho = 0.05
    for kappa, col in zip([0.5, 1.0, 2.0], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        m = OUOpt(kappa, theta, sigma, rho, c)
        es_exit = m.exit_long()
        es_entry = m.entry_long(es_exit)
        grid = np.linspace(es_entry - 0.4, es_exit + 0.2, 400)
        G = m.G_plus(grid, es_exit, es_entry)
        cont = grid > es_entry
        ax[0].plot(grid[cont], G[cont], color=col, label=fr"$\kappa={kappa:g}$")
        ax[0].plot(grid[~cont], G[~cont], color=col, ls="--", lw=1.0)
        ax[0].plot(es_entry, m.G_plus(np.array([es_entry]), es_exit, es_entry)[0],
                   "o", color="k", ms=4, zorder=5)
    ax[0].set_title(fr"(a) $G_+$, varying $\kappa$  ($\rho={rho}$)")
    ax[0].set_xlabel(r"$\varepsilon$")
    ax[0].set_ylabel(r"$G_+(\varepsilon)$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    kappa = 1.0
    for rho, col in zip([0.01, 0.1, 0.5], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        m = OUOpt(kappa, theta, sigma, rho, c)
        es_exit = m.exit_long()
        es_entry = m.entry_long(es_exit)
        grid = np.linspace(es_entry - 0.4, es_exit + 0.2, 400)
        G = m.G_plus(grid, es_exit, es_entry)
        cont = grid > es_entry
        ax[1].plot(grid[cont], G[cont], color=col, label=fr"$\rho={rho:g}$")
        ax[1].plot(grid[~cont], G[~cont], color=col, ls="--", lw=1.0)
        ax[1].plot(es_entry, m.G_plus(np.array([es_entry]), es_exit, es_entry)[0],
                   "o", color="k", ms=4, zorder=5)
    ax[1].set_title(fr"(b) $G_+$, varying $\rho$  ($\kappa={kappa}$)")
    ax[1].set_xlabel(r"$\varepsilon$")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Optimal entry: value function $G_+$ (solid=continue, dashed=exercise, "
                 "$\\bullet$=entry trigger)", fontsize=10)
    _save(fig, "fig_11_5.pdf")


# ======================================================================================
# Fig 11.6 - double-sided entry/exit value functions + trigger table
# ======================================================================================
def fig_11_6():
    theta, sigma, rho, c = 1.0, 0.5, 0.01, 0.01
    kappas = [0.5, 1.0, 2.0, 4.0]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    rows = []
    for kappa, col in zip(kappas, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        m = OUOpt(kappa, theta, sigma, rho, c)
        el, esh = m.exit_long(), m.exit_short()
        rows.append((kappa, el, esh))
        # Plot each value function only over a window around its own trigger, where it is
        # numerically well-conditioned (the far-from-theta tail underflows for small rho/kappa).
        gp = np.linspace(el - 1.2, el + 0.5, 300)
        ax[0].plot(gp, m.H_plus(gp, el), color=col, lw=1.3, label=fr"$\kappa={kappa:g}$")
        ax[0].plot(el, el - c, "o", color="k", ms=4, zorder=5)
        gm = np.linspace(esh - 0.5, esh + 1.2, 300)
        ax[0].plot(gm, m.H_minus(gm, esh), color=col, lw=1.0, ls="--")
        ax[0].plot(esh, -(esh + c), "s", color="k", ms=4, zorder=5)
    ax[0].axvline(theta, color="grey", lw=0.7, ls=":")
    ax[0].set_title(r"$H_+$ (solid) and $H_-$ (dashed),  $\theta=1$")
    ax[0].set_xlabel(r"$\varepsilon$")
    ax[0].set_ylabel("value")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # trigger table panel
    ax[1].axis("off")
    cells = [[f"{k:g}", f"{el:.4f}", f"{esh:.4f}"] for (k, el, esh) in rows]
    tbl = ax[1].table(
        cellText=cells,
        colLabels=[r"$\kappa$", r"exit long $\varepsilon^*_+$", r"exit short $\varepsilon^*_-$"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    ax[1].set_title("Optimal trigger levels", fontsize=10)
    fig.suptitle("Double-sided optimal entry-exit (entry trigger = opposite exit trigger)",
                 fontsize=10)
    _save(fig, "fig_11_6.pdf")
    return rows


# ======================================================================================
# Section-12 three-asset short-term-alpha model (Figs 11.7, 11.8)
# ======================================================================================
SIG = np.array([
    [0.2000, 0.0, 0.0],
    [0.0375, 0.1452, 0.0],
    [0.0250, 0.0039, 0.0967],
])
DELTA = np.array([1.0, 1.0, 0.0])
A_VEC = np.array([-1.0, 0.0, 1.0])
A0 = 0.0
Y0 = np.array([11.10, 12.00, 11.00])
OMEGA = SIG @ SIG.T
OMEGA_INV = np.linalg.inv(OMEGA)
TR_AOMEGA = float(np.sum(A_VEC * np.diag(OMEGA)))   # Tr(A Omega), A=diag(a)
DOD = float(DELTA @ OMEGA @ DELTA)                  # delta' Omega delta
GAMMA = 1.0


def _alpha(logY):
    return A0 + float(A_VEC @ logY)


def _pi_star(alpha, tau):
    """Optimal dollar investment, eq. (11.16)."""
    merton = OMEGA_INV @ DELTA * alpha
    corr = DOD * (0.5 * tau * alpha + 0.25 * TR_AOMEGA * tau**2) * A_VEC
    return (merton + corr) / GAMMA


def simulate_section12(T, n_steps, rng):
    dt = T / n_steps
    sqdt = np.sqrt(dt)
    Y = np.empty((n_steps + 1, 3))
    Y[0] = Y0
    alpha = np.empty(n_steps + 1)
    pis = np.empty((n_steps + 1, 3))
    X = np.empty(n_steps + 1)
    X[0] = 0.0
    logY = np.log(Y0)
    for i in range(n_steps):
        tau = T - i * dt
        a = _alpha(logY)
        alpha[i] = a
        pi = _pi_star(a, tau)
        pis[i] = pi
        dW = sqdt * rng.standard_normal(3)
        # dY_k / Y_k = delta_k a dt + sum_i sig_ki dW_i
        ret = DELTA * a * dt + SIG @ dW
        Yn = Y[i] * (1.0 + ret)
        Yn = np.maximum(Yn, 1e-6)
        Y[i + 1] = Yn
        # dX = sum_k pi_k * (dY_k / Y_k)
        X[i + 1] = X[i] + float(pi @ ret)
        logY = np.log(Yn)
    alpha[-1] = _alpha(logY)
    pis[-1] = _pi_star(alpha[-1], 0.0)
    t = np.linspace(0, T, n_steps + 1)
    return t, Y, alpha, pis, X


def fig_11_7():
    rng = np.random.default_rng(3)
    t, Y, alpha, pis, X = simulate_section12(1.0, 1000, rng)
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    for k, lab in enumerate(["$Y^1$", "$Y^2$", "$Y^3$"]):
        ax[0, 0].plot(t, Y[:, k], label=lab)
    ax[0, 0].set_title("(a) Asset prices")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(t, alpha, color="#9467bd")
    ax[0, 1].axhline(0, ls="--", color="k", lw=0.7)
    ax[0, 1].set_title(r"(b) Co-integration factor $\alpha_t$")
    ax[0, 1].grid(alpha=0.3)

    for k, lab in enumerate(["$\\pi^1$", "$\\pi^2$", "$\\pi^3$"]):
        ax[1, 0].plot(t, pis[:, k], label=lab)
    ax[1, 0].set_title(r"(c) Optimal investment $\pi^*_t$")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(t, X, color="#2ca02c")
    ax[1, 1].set_title("(d) Wealth $X_t$")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle("Section-12 three-asset co-integrated short-term-alpha model: sample path",
                 fontsize=10)
    _save(fig, "fig_11_7.pdf")


def fig_11_8():
    rng = np.random.default_rng(99)
    n_paths = 10000
    pnl = np.empty(n_paths)
    for p in range(n_paths):
        _, _, _, _, X = simulate_section12(1.0, 250, rng)
        pnl[p] = X[-1]
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pnl, bins=80, color="#4c72b0", edgecolor="white", linewidth=0.2)
    ax.axvline(pnl.mean(), color="k", ls="--", lw=1.0, label=f"mean {pnl.mean():.3f}")
    ax.set_title(f"Section-12 optimal strategy: P&L over {n_paths:,} scenarios "
                 f"(Sharpe {sharpe:.2f})", fontsize=10)
    ax.set_xlabel("P&L")
    ax.set_xlim(np.percentile(pnl, 0.2), np.percentile(pnl, 99.8))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "fig_11_8.pdf")


def main():
    print("Generating book figures ->", FIG_DIR)
    fig_11_1()
    fig_11_2()
    fig_11_3()
    fig_11_4()
    fig_11_5()
    rows = fig_11_6()
    fig_11_7()
    fig_11_8()
    print("\nFig 11.6 trigger table (cross-check vs PDF):")
    book = {0.5: (1.9537, -0.4060), 1.0: (1.7460, -0.1815),
            2.0: (1.5744, -0.0740), 4.0: (1.4367, -0.0410)}
    for k, el, es in rows:
        bl, bs = book[k]
        ok = abs(el - bl) < 1e-3 and abs(es - bs) < 1e-3
        print(f"  kappa={k:>3}: exit_long={el:.4f} (book {bl})  "
              f"exit_short={es:.4f} (book {bs})  {'OK' if ok else 'MISMATCH'}")
    print("done.")


if __name__ == "__main__":
    main()

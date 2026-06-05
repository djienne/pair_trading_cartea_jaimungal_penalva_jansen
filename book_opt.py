"""Cartea-Jaimungal-Penalva optimal entry/exit triggers for a mean-reverting (OU) co-integration
factor, used to regenerate Figures 11.4-11.6 of the chapter.

This is the same optimal-stopping problem solved in ``band_calc.py`` (class ``CointOpti``), written
here in a numerically-stabilized form so the value functions H+/H-/G+ and the trigger levels can be
evaluated cleanly over a grid for plotting.

OU factor:  d eps = kappa (theta - eps) dt + sigma dW,   discount rho, transaction cost c.

Fundamental solutions of (L - rho)F = 0 (book, between eq. 11.1 and 11.2):

    F_+/-(eps) = \\int_0^\\infty u^{rho/kappa - 1} exp( -/+ b (theta - eps) u - 1/2 u^2 ) du,
        with  b = sqrt(2 kappa / sigma^2).

Completing the square in u removes the overflow:  with  c1 = -/+ b (theta - eps),
    F(eps) = exp(1/2 c1^2) * J0(eps),   F'(eps) = exp(1/2 c1^2) * c1' * J1(eps),
    J_n(eps) = \\int_0^\\infty u^{p-1+n} exp(-1/2 (u - c1)^2) du,   p = rho/kappa,
    c1' = d c1 / d eps.   The exp(1/2 c1^2) factor cancels in every trigger equation.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq

# The J_n integrand has an integrable u^(p-1) singularity at u=0 when rho/kappa < 1; quad warns but
# returns the correct value (cross-validated against the chapter's trigger table). Silence the noise.
warnings.filterwarnings("ignore", category=IntegrationWarning)


class OUOpt:
    def __init__(self, kappa: float, theta: float, sigma: float, rho: float, c: float):
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.c = float(c)
        self.p = self.rho / self.kappa
        self.b = np.sqrt(2.0 * self.kappa / self.sigma**2)

    # --- stabilized building blocks -------------------------------------------------
    def _c1(self, eps: float, sign: float) -> float:
        # sign = -1 for F_+, +1 for F_-
        return sign * self.b * (self.theta - eps)

    def _Jn(self, eps: float, sign: float, n: int) -> float:
        c1 = self._c1(eps, sign)
        p = self.p
        # integrand peaks near u=c1 (if c1>0) or u=0; integrate over a safe finite window.
        hi = max(c1, 0.0) + 12.0
        val, _ = quad(
            lambda u: u ** (p - 1.0 + n) * np.exp(-0.5 * (u - c1) ** 2),
            0.0, hi, limit=200,
        )
        return val

    def _logpref(self, eps: float, sign: float) -> float:
        c1 = self._c1(eps, sign)
        return 0.5 * c1 * c1

    def F(self, eps: float, sign: float) -> float:
        return np.exp(self._logpref(eps, sign)) * self._Jn(eps, sign, 0)

    def dF(self, eps: float, sign: float) -> float:
        c1p = -sign * self.b  # d c1 / d eps
        return np.exp(self._logpref(eps, sign)) * c1p * self._Jn(eps, sign, 1)

    # Ratio F(eps)/F(eps_ref) for the SAME sign, computed in log space (overflow-safe).
    def F_ratio(self, eps: float, eps_ref: float, sign: float) -> float:
        log_r = self._logpref(eps, sign) - self._logpref(eps_ref, sign)
        return np.exp(log_r) * self._Jn(eps, sign, 0) / self._Jn(eps_ref, sign, 0)

    # --- exit triggers --------------------------------------------------------------
    def exit_long(self) -> float:
        # (eps - c) F_+'(eps) = F_+(eps)   ->   root above theta.
        g = lambda e: (e - self.c) * self.dF(e, -1.0) - self.F(e, -1.0)
        return brentq(g, self.theta + 1e-9, self.theta + 6.0, xtol=1e-12, rtol=1e-12)

    def exit_short(self) -> float:
        # (eps + c) F_-'(eps) = F_-(eps)   ->   root below theta.
        g = lambda e: (e + self.c) * self.dF(e, 1.0) - self.F(e, 1.0)
        return brentq(g, self.theta - 6.0, self.theta - 1e-9, xtol=1e-12, rtol=1e-12)

    # --- one-sided long value functions (Fig 11.4 / 11.5) ---------------------------
    def H_plus(self, eps, eps_star=None):
        """Exit-from-long value function H_+(eps)."""
        if eps_star is None:
            eps_star = self.exit_long()
        eps = np.atleast_1d(np.asarray(eps, dtype=float))
        out = np.empty_like(eps)
        for i, e in enumerate(eps):
            if e < eps_star:
                out[i] = (eps_star - self.c) * self.F_ratio(e, eps_star, -1.0)
            else:
                out[i] = e - self.c
        return out

    def dH_plus(self, eps_star: float) -> float:
        # H_+'(eps*) = 1 by smooth pasting; provided for completeness.
        return 1.0

    def entry_long(self, eps_star_exit: float | None = None) -> float:
        """One-sided optimal LONG entry trigger eps_* (book eq. 11.3).

        (H_+(e) - e - c) F_-'(e) = (H_+'(e) - 1) F_-(e).  Below the exit level, H_+(e) = A F_+(e),
        so H_+'(e) follows from differentiating that branch.
        """
        if eps_star_exit is None:
            eps_star_exit = self.exit_long()
        A = (eps_star_exit - self.c) / self.F(eps_star_exit, -1.0)

        def Hp(e):
            return A * self.F(e, -1.0)

        def dHp(e):
            return A * self.dF(e, -1.0)

        def g(e):
            return (Hp(e) - e - self.c) * self.dF(e, 1.0) - (dHp(e) - 1.0) * self.F(e, 1.0)

        # entry sits below the long-run level; bracket on the low side.
        return brentq(g, self.theta - 6.0, eps_star_exit - 1e-6, xtol=1e-12, rtol=1e-12)

    def G_plus(self, eps, eps_star_exit=None, eps_star_entry=None):
        """Entry-into-long value function G_+(eps) (book, Section 11.3.2)."""
        if eps_star_exit is None:
            eps_star_exit = self.exit_long()
        if eps_star_entry is None:
            eps_star_entry = self.entry_long(eps_star_exit)
        A = (eps_star_exit - self.c) / self.F(eps_star_exit, -1.0)
        Hp_at = lambda e: A * self.F(e, -1.0)
        val_at_entry = Hp_at(eps_star_entry) - eps_star_entry - self.c
        eps = np.atleast_1d(np.asarray(eps, dtype=float))
        out = np.empty_like(eps)
        for i, e in enumerate(eps):
            if e > eps_star_entry:
                out[i] = val_at_entry * self.F_ratio(e, eps_star_entry, 1.0)
            else:
                out[i] = Hp_at(e) - e - self.c
        return out

    def H_minus(self, eps, eps_star=None):
        """Exit-from-short value function H_-(eps)."""
        if eps_star is None:
            eps_star = self.exit_short()
        eps = np.atleast_1d(np.asarray(eps, dtype=float))
        out = np.empty_like(eps)
        for i, e in enumerate(eps):
            if e > eps_star:
                out[i] = -(eps_star + self.c) * self.F_ratio(e, eps_star, 1.0)
            else:
                out[i] = -(e + self.c)
        return out


def trigger_table(kappas, theta, sigma, rho, c):
    rows = []
    for k in kappas:
        m = OUOpt(k, theta, sigma, rho, c)
        rows.append((k, m.exit_long(), m.exit_short()))
    return rows


if __name__ == "__main__":
    # Self-test against the table printed in the PDF (theta=1, sigma=0.5, rho=0.01, c=0.01).
    print("kappa   exit_long(e*+)   exit_short(e*-)   [book values]")
    book = {0.5: (1.9537, -0.4060), 1.0: (1.7460, -0.1815),
            2.0: (1.5744, -0.0740), 4.0: (1.4367, -0.0410)}
    for k, el, es in trigger_table([0.5, 1.0, 2.0, 4.0], 1.0, 0.5, 0.01, 0.01):
        bl, bs = book[k]
        print(f"{k:>4}   {el:>12.4f}   {es:>13.4f}   [{bl:.4f}, {bs:.4f}]")

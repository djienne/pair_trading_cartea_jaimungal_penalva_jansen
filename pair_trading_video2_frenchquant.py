import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

class CointOpti:
    """
    Class to calculate optimal entry and exit thresholds for a cointegration 
    strategy modeled by an Ornstein-Uhlenbeck process.
    """
    def __init__(self, c, theta, kappa, sigma, rho):
        self.c = c          # Transaction costs
        self.theta = theta  # Long-term mean of the spread
        self.kappa = kappa  # Mean reversion speed
        self.sigma = sigma  # Volatility
        self.rho = rho      # Discount factor

    def int_plus(self, u, epsilon):
        """Integrand for the F_plus function (decreasing eigenfunction)."""
        # Note: A minus sign is included in the exponential for convergence as u -> infinity
        return np.power(u, self.rho / self.kappa - 1) * np.exp(- (self.kappa * (u - self.theta)**2) / (self.sigma**2))

    def int_minus(self, u, epsilon):
        """Integrand for the F_minus function (increasing eigenfunction)."""
        return np.power(u, self.rho / self.kappa - 1) * np.exp(- (self.kappa * (u - self.theta)**2) / (self.sigma**2))

    def dint_plus(self, epsilon):
        """
        Analytic derivative of F_plus with respect to epsilon.
        Since F_plus integrates from epsilon to infinity, the derivative is -integrand(epsilon).
        """
        return -1 * np.power(epsilon, self.rho / self.kappa - 1) * np.exp(- (self.kappa * (epsilon - self.theta)**2) / (self.sigma**2))

    def dint_minus(self, epsilon):
        """
        Analytic derivative of F_minus with respect to epsilon.
        Since F_minus integrates from 0 to epsilon, the derivative is integrand(epsilon).
        """
        return 1 * np.power(epsilon, self.rho / self.kappa - 1) * np.exp(- (self.kappa * (epsilon - self.theta)**2) / (self.sigma**2))

    def F_plus(self, epsilon):
        """ Calculates the integral F_plus(epsilon) = Integral(epsilon -> inf). """
        # Integration from epsilon to infinity
        res, error = quad(self.int_plus, epsilon, np.inf, args=(epsilon,))
        return res

    def F_minus(self, epsilon):
        """ Calculates the integral F_minus(epsilon) = Integral(0 -> epsilon). """
        # Integration from 0 to epsilon (assuming the process is positive, e.g., price spread around a positive mean)
        res, error = quad(self.int_minus, 0, epsilon, args=(epsilon,))
        return res

    def dF_plus(self, epsilon, analytic=True):
        """ Calculates the derivative of F_plus. """
        if analytic:
            return self.dint_plus(epsilon)
        else:
            # Numerical approximation
            h = 1e-4
            return (self.F_plus(epsilon + h) - self.F_plus(epsilon - h)) / (2 * h)

    def dF_minus(self, epsilon, analytic=True):
        """ Calculates the derivative of F_minus. """
        if analytic:
            return self.dint_minus(epsilon)
        else:
            # Numerical approximation
            h = 1e-4
            return (self.F_minus(epsilon + h) - self.F_minus(epsilon - h)) / (2 * h)

    def H_plus(self, epsilon, analytic=True):
        """ 
        The condition function H(epsilon). 
        The root of this function gives the optimal threshold.
        Corresponds to the condition: F(e) - (e - c) * F'(e) = 0
        """
        return self.F_plus(epsilon) - (epsilon - self.c) * self.dF_plus(epsilon, analytic)

    def solve_optimal_long_short(self, epsilon_star, analytic=True):
        """ Finds the optimal lower threshold for a symmetric Long-Short strategy. """
        solution = root(self.H_plus, epsilon_star, args=(analytic,))
        return solution.x[0]

    def solve_optimal_entry(self, epsilon_botstar, analytic=True):
        """ Finds the optimal entry threshold (lower bound) for a Long-Only strategy. """
        func = lambda epsilon: self.F_plus(epsilon) - (epsilon - self.c) * self.dF_plus(epsilon, analytic)
        solution = root(func, epsilon_botstar)
        return solution.x[0]

    def solve_optimal_stop(self, epsilon_star, analytic=True):
        """ Finds the optimal exit threshold (upper bound) for a Long-Only strategy. """
        func = lambda epsilon: self.F_minus(epsilon) - (epsilon - self.c) * self.dF_minus(epsilon, analytic)
        solution = root(func, epsilon_star)
        return solution.x[0]

    def get_opti_params(self, analytic=True, long_only=True, init_low=0.5, init_high=1.5):
        """
        Main method to get optimal parameters.
        Returns (entry_threshold, exit_threshold).
        """
        if long_only:
            # For long-only: calculate optimal entry (low) and optimal exit (high) independent equations
            epsilon_entry = self.solve_optimal_entry(init_low, analytic)
            epsilon_exit = self.solve_optimal_stop(init_high, analytic)
            return epsilon_entry, epsilon_exit
        else:
            # For long-short: assume symmetry around theta
            epsilon_star = self.solve_optimal_long_short(init_low, analytic)
            # Returns lower bound and symmetric upper bound
            return epsilon_star, -epsilon_star + 2 * self.theta

# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":
    # Define parameters for the OU process
    # c: transaction cost
    # theta: long term mean
    # kappa: speed of mean reversion
    # sigma: volatility
    # rho: discount factor
    c = 0.05
    theta = 1
    kappa = 1
    sigma = 0.5
    rho = 0.05

    # Initialize the optimizer
    test = CointOpti(c, theta, kappa, sigma, rho)

    # 1. Long Only Strategy Optimization
    # Returns (Entry Level, Exit Level)
    print("--- Long Only Strategy ---")
    entry, exit_ = test.get_opti_params(analytic=True, long_only=True, init_low=0.5, init_high=1.5)
    print(f"Optimal Entry: {entry:.4f}")
    print(f"Optimal Exit:  {exit_:.4f}")

    # 2. Long-Short Strategy Optimization
    # Returns (Lower Bound, Upper Bound)
    print("\n--- Long-Short Strategy ---")
    lower, upper = test.get_opti_params(analytic=True, long_only=False, init_low=0.5, init_high=1.5)
    print(f"Optimal Lower Bound (Long Entry): {lower:.4f}")
    print(f"Optimal Upper Bound (Short Entry): {upper:.4f}")
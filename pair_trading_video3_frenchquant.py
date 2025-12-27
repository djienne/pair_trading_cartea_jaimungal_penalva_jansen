import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spop
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. Simulation of the Ornstein-Uhlenbeck Process
# ---------------------------------------------------------

def simulate_ou_process(theta, mu, sigma, x0, T, dt):
    """
    Simulates an Ornstein-Uhlenbeck process using the Euler-Maruyama method.
    dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    """
    N = int(T / dt) # Number of time steps
    x = np.zeros(N)
    x[0] = x0
    
    for t in range(1, N):
        # Discretized equation:
        # X(t) = X(t-1) + theta * (mu - X(t-1)) * dt + sigma * sqrt(dt) * Normal(0,1)
        # Note: np.random.normal(0, np.sqrt(dt)) is equivalent to sqrt(dt) * Normal(0,1)
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
        
    return x

# ---------------------------------------------------------
# 2. Parameter Estimation: Method of Moments (Linear Regression)
# ---------------------------------------------------------

def method_moments(x, dt):
    """
    Estimates parameters using OLS regression on the discretized process.
    Rearranges the equation to: (X_t - X_{t-1})/dt = theta * (mu - X_{t-1}) + error
    """
    # 1. Estimate mu as the mean of the series
    mu = np.mean(x)
    
    # 2. Estimate theta using Linear Regression (OLS)
    # Dependent variable (y): Approximate derivative
    y = (x[1:] - x[:-1]) / dt
    
    # Independent variable (exog): Distance from mean
    exog = mu - x[:-1]
    
    # Fit model (no intercept needed based on the formulation)
    res = sm.OLS(y, exog).fit()
    theta = res.params[0]
    
    # 3. Estimate sigma from the residuals of the regression
    # The residuals correspond to sigma * dW_t / dt ? 
    # Actually residuals ~ sigma * N(0, dt)/dt. 
    # So std(resid) approx sigma * 1/sqrt(dt). Therefore sigma = std(resid) * sqrt(dt)?
    # In the video code: sigma = np.std(res.resid) / np.sqrt(dt) 
    # Let's stick to the video implementation:
    sigma = np.std(res.resid) / np.sqrt(dt)
    
    return theta, mu, sigma

# ---------------------------------------------------------
# 3. Parameter Estimation: Maximum Likelihood Estimation (MLE)
# ---------------------------------------------------------

def ll_ou(params, x, dt):
    """
    Negative Log-Likelihood function for the OU process.
    The conditional distribution of X_t given X_{t-1} is Normal.
    """
    theta, mu, sigma = params
    
    # Constraint checks to prevent mathematical errors during optimization
    if sigma <= 0 or theta < 0:
        return 1e10
    
    # Conditional Mean: X_{t-1} + theta * (mu - X_{t-1}) * dt
    m = x[:-1] + theta * (mu - x[:-1]) * dt
    
    # Conditional Standard Deviation: sigma * sqrt(dt)
    s = sigma * np.sqrt(dt)
    
    # Calculate Negative Log Likelihood
    # PDF of Normal(m, s) at x[1:]
    pdf_values = (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x[1:] - m) / s)**2)
    log_likelihood = np.sum(np.log(pdf_values))
    
    return -log_likelihood

def method_mle(x, dt):
    """
    Estimates parameters using numerical optimization of the Log-Likelihood.
    """
    # Initial guesses
    mu_init = np.mean(x)
    sigma_init = np.std(x) # Simplified guess
    theta_init = 1
    
    x0 = [theta_init, mu_init, sigma_init]
    
    # Minimize the negative log-likelihood
    res = spop.minimize(ll_ou, x0, args=(x, dt), method='Nelder-Mead')
    
    return res.x

# ---------------------------------------------------------
# 4. Execution / Main Script
# ---------------------------------------------------------

if __name__ == "__main__":
    # Parameters definition
    theta = 2      # Speed of mean reversion
    mu = 2         # Long term mean
    sigma = 0.5    # Volatility
    x0 = 2         # Initial value
    
    # Time settings
    T = 10         # Total time in years (increased to 10 for better estimation convergence)
    dt = 1/252     # Time step (daily assuming 252 trading days)
    
    # 1. Run Simulation
    x = simulate_ou_process(theta, mu, sigma, x0, T, dt)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, T, dt)[:len(x)], x)
    plt.title("Simulation du processus d'Ornstein-Uhlenbeck")
    plt.xlabel("Temps")
    plt.ylabel("X(t)")
    plt.show()
    
    # 2. Estimate Parameters
    print(f"True Parameters -> theta: {theta}, mu: {mu}, sigma: {sigma}")
    
    # Method of Moments
    theta_mm, mu_mm, sigma_mm = method_moments(x, dt)
    print(f"Method of Moments -> theta: {theta_mm:.4f}, mu: {mu_mm:.4f}, sigma: {sigma_mm:.4f}")
    
    # Maximum Likelihood Estimation
    params_mle = method_mle(x, dt)
    print(f"MLE -> theta: {params_mle[0]:.4f}, mu: {params_mle[1]:.4f}, sigma: {params_mle[2]:.4f}")
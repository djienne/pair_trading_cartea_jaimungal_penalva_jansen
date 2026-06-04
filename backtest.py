import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for safer plotting in scripts/parallel processes
import matplotlib.pyplot as plt

from utils import (
    load_config,
    pair_id,
    get_dirs,
    load_pair_data,
    save_pair_data,
)


def backtest_pair(pair: Dict, config: Dict) -> pd.DataFrame:
    # Load data using consolidated utilities
    coint_df = load_pair_data(pair, config, "coint")
    bands_df = load_pair_data(pair, config, "bands")

    # Merge coint and bands data
    df = coint_df.join(bands_df, how="left", rsuffix="_bands")
    n = len(df)
    if n == 0:
        raise ValueError("No data available for backtest.")

    # Prepare arrays for fast iteration
    y = df["y_close"].values
    x = df["x_close"].values
    # alpha = df["alpha"].values # Not explicitly needed for trading delta, but part of signal
    beta = df["beta"].values
    epsilon = df["epsilon"].values
    lower = df["lower"].values
    upper = df["upper"].values
    mu = df["mu"].values

    # Simulation State
    pos = 0          # Current position: +1 (Long Portfolio), -1 (Short Portfolio), 0 (Flat)
    m1 = 0.0         # Quantity of Asset 1 (Y)
    m2 = 0.0         # Quantity of Asset 2 (X)
    start_equity = float(config.get("start_equity", 1000))
    cash = start_equity  # Start with all cash

    # We track Book Value (BV) = Cash + Market Value of Positions
    book_value_arr = np.zeros(n)
    pos_arr = np.zeros(n)
    m1_arr = np.zeros(n)
    m2_arr = np.zeros(n)
    cash_arr = np.zeros(n)
    turnover_arr = np.zeros(n)

    fee_rate = float(config.get("fee_rate", 0.001))  # fraction of traded notional
    flip_signals = bool(config.get("flip_signals", False))

    # --- Position sizing ---
    # Scale the unit bundle (1 unit of Y, beta units of X) by a single factor q so that
    # exposure is proportional to current capital rather than to raw asset price.
    #   "gross"       -> total gross notional |m1*py| + |m2*px| == target_gross_leverage * equity
    #   "net_long"    -> the Y leg notional == capital_per_trade_frac * equity
    #   "legacy_unit" -> q = 1.0 (reproduces the old fixed-unit behaviour exactly)
    sizing_mode = str(config.get("sizing_mode", "gross")).lower()
    target_gross_leverage = float(config.get("target_gross_leverage", 1.0))
    capital_per_trade_frac = float(config.get("capital_per_trade_frac", 0.5))
    # Execution lag: the signal/bands from bar (i - lag) drive the trade executed at bar i's
    # price. lag = 0 keeps the original same-bar decide-and-fill behaviour.
    lag = max(0, int(config.get("execution_lag_bars", 0)))

    # --- Risk controls (opt-in, default off) ---
    # Equity-drawdown stop-loss: force-flat an open position once its mark-to-market loss exceeds
    # stop_loss_frac of the equity at entry (e.g. 0.25 = 25%). null/0 disables it.
    _sl = config.get("stop_loss_frac", None)
    stop_loss_frac = float(_sl) if _sl not in (None, "", 0, 0.0) else 0.0
    # Daily rehedge: each bar while holding, refresh the X leg to -beta_t * m1 so the hedge tracks
    # the current beta (pays fees on the delta). Off => hold the original bundle until exit.
    rehedge_daily = bool(config.get("rehedge_daily", False))
    equity_at_entry = 0.0  # book value when the current position was opened (for the stop-loss)

    # Initial state
    book_value_arr[0] = cash
    cash_arr[0] = cash

    for i in range(1, n):
        # Prices for execution / mark-to-market at the current bar.
        py = y[i]
        px = x[i]

        # The decision (signal + bands) comes from bar (i - lag).
        sig_i = i - lag

        # Skip if the decision bar is out of range or any required data is missing (NaN).
        if sig_i < 0 or not (
            np.isfinite(epsilon[sig_i])
            and np.isfinite(lower[sig_i])
            and np.isfinite(upper[sig_i])
        ):
            # Carry forward state, marking to market at current prices.
            book_value_arr[i] = cash + m1 * py + m2 * px
            pos_arr[i] = pos
            m1_arr[i] = m1
            m2_arr[i] = m2
            cash_arr[i] = cash
            continue

        z = epsilon[sig_i]
        curr_lower = lower[sig_i]
        curr_upper = upper[sig_i]
        curr_mu = mu[sig_i]
        curr_beta = beta[sig_i]

        # Flip signals: swap upper/lower bands to reverse entry/exit logic
        if flip_signals:
            curr_lower, curr_upper = curr_upper, curr_lower

        # --- Trading Logic ---
        # 1. Check Entries
        if pos == 0:
            enter_long = z <= curr_lower
            enter_short = z >= curr_upper
            if enter_long or enter_short:
                # Size the unit bundle relative to current book value (see sizing config).
                equity_now = cash + m1 * py + m2 * px  # == cash when flat
                unit_gross = abs(py) + abs(curr_beta * px)
                if equity_now > 0 and unit_gross > 1e-12 and np.isfinite(unit_gross):
                    sign_y = 1.0 if enter_long else -1.0
                    if sizing_mode == "legacy_unit":
                        q = 1.0
                    elif sizing_mode == "net_long":
                        q = (capital_per_trade_frac * equity_now) / abs(py) if abs(py) > 1e-12 else 0.0
                    else:  # "gross"
                        q = (target_gross_leverage * equity_now) / unit_gross
                    # LONG portfolio: long q units of Y, short beta*q units of X (short flips signs).
                    pos = 1 if enter_long else -1
                    target_m1 = sign_y * q
                    target_m2 = -sign_y * curr_beta * q
                    equity_at_entry = equity_now  # baseline for the stop-loss
                else:
                    # Degenerate sizing (non-positive equity or zero gross): stay flat.
                    target_m1 = 0.0
                    target_m2 = 0.0
            else:
                # Stay Flat
                target_m1 = 0.0
                target_m2 = 0.0

        # 2. Check Exits
        elif pos == 1: # Currently Long
            # Mark-to-market before any action this bar (for the stop-loss check).
            cur_bv = cash + m1 * py + m2 * px
            stop_hit = stop_loss_frac > 0.0 and (cur_bv - equity_at_entry) <= -stop_loss_frac * equity_at_entry
            # Exit condition depends on flip_signals
            exit_long = (z <= curr_mu) if flip_signals else (z >= curr_mu)
            if exit_long or stop_hit:
                # Exit to Flat (band exit or stop-loss)
                pos = 0
                target_m1 = 0.0
                target_m2 = 0.0
            elif rehedge_daily:
                # Keep the Y leg; refresh the X leg to the current hedge ratio.
                target_m1 = m1
                target_m2 = -curr_beta * m1
            else:
                # Hold the original bundle until exit.
                target_m1 = m1
                target_m2 = m2

        elif pos == -1: # Currently Short
            cur_bv = cash + m1 * py + m2 * px
            stop_hit = stop_loss_frac > 0.0 and (cur_bv - equity_at_entry) <= -stop_loss_frac * equity_at_entry
            # Exit condition depends on flip_signals
            exit_short = (z >= curr_mu) if flip_signals else (z <= curr_mu)
            if exit_short or stop_hit:
                # Exit to Flat (band exit or stop-loss)
                pos = 0
                target_m1 = 0.0
                target_m2 = 0.0
            elif rehedge_daily:
                # Keep the Y leg; refresh the X leg to the current hedge ratio (m1 is negative here).
                target_m1 = m1
                target_m2 = -curr_beta * m1
            else:
                # Hold the original bundle until exit.
                target_m1 = m1
                target_m2 = m2

        # --- Execution ---
        # Calculate turnover and costs
        dm1 = target_m1 - m1
        dm2 = target_m2 - m2

        trade_value = abs(dm1 * py) + abs(dm2 * px)
        # fee_rate is a fraction of traded notional (distinct from band_calc's transaction_cost,
        # which is in absolute spread units).
        cost = trade_value * fee_rate

        # Update cash: Cash decreases by cost of buying assets, increases by selling
        # Cost of buying dm1 of Y is (dm1 * py)
        cash -= (dm1 * py + dm2 * px)
        cash -= cost

        m1 = target_m1
        m2 = target_m2

        turnover_arr[i] = trade_value
        pos_arr[i] = pos
        m1_arr[i] = m1
        m2_arr[i] = m2
        cash_arr[i] = cash

        # Mark to Market
        book_value_arr[i] = cash + m1 * py + m2 * px

    df["position"] = pos_arr
    df["m1"] = m1_arr
    df["m2"] = m2_arr
    df["cash"] = cash_arr
    df["equity"] = book_value_arr
    df["turnover"] = turnover_arr
    
    # Calculate returns for metrics.
    # PnL over prior equity, guarded against a non-positive denominator so that a (pathological,
    # leveraged) equity dip through zero cannot produce inf/NaN or absurd percentage returns.
    prev_equity = np.empty(n)
    prev_equity[0] = start_equity
    if n > 1:
        prev_equity[1:] = book_value_arr[:-1]
    pnl = np.empty(n)
    pnl[0] = 0.0
    if n > 1:
        pnl[1:] = book_value_arr[1:] - book_value_arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(prev_equity > 1e-9, prev_equity, np.nan)
        strat_ret = pnl / denom
    df["strategy_return"] = pd.Series(strat_ret, index=df.index).fillna(0.0)

    # Save using consolidated utility
    save_pair_data(df, pair, config, "backtest")
    return df


def plot_equity(
    results: pd.DataFrame,
    pair_name: str,
    save_path: str = None,
    show_plot: bool = True,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["equity"], label="Equity", linewidth=1.5)
    plt.title(f"Pair Trading Equity ({pair_name})")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved equity plot to {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured in config.json")
        
    # Prepare plots directory
    data_dir = config.get("data_dir", "data")
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))

    for pair in pairs:
        pair_name = pair.get("name") or pair_id(pair)
        print(f"Running backtest for {pair_name}...")
        results = backtest_pair(pair, config)
        print(f"Final equity for {pair_name}: {results['equity'].iloc[-1]:.2f} USD")
        
        # Construct meaningful filename
        start_date = results.index[0].strftime('%Y%m%d')
        end_date = results.index[-1].strftime('%Y%m%d')
        
        # Format: equity_{PAIR}_{INTERVAL}_w{WINDOW}_{START}-{END}.png
        filename = f"equity_{pair_name}_{interval}_w{window}_{start_date}-{end_date}.png"
        
        # Sanitize filename (replace forbidden characters)
        filename = filename.replace(os.path.sep, "_").replace(":", "")
        
        save_path = os.path.join(plots_dir, filename)
        show_plot = bool(config.get("show_plots", True))
        plot_equity(results, pair_name, save_path, show_plot=show_plot)


if __name__ == "__main__":
    main()

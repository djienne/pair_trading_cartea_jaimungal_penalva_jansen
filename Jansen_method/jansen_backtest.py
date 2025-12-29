import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import load_config, resolve_path, get_data_path, get_output_path

try:
    from pykalman import KalmanFilter
except ImportError as exc:
    raise ImportError(
        "pykalman is required. Please install with: pip install pykalman"
    ) from exc


def load_price_series(symbol: str, config: dict, base_dir: str) -> pd.Series:
    path = get_data_path(base_dir, symbol, config)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    df = df.sort_values("open_time_dt").set_index("open_time_dt")
    return df["close"].rename(symbol)


def kf_smoother(prices: pd.Series) -> pd.Series:
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=np.eye(1),
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.05,
    )
    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(), index=prices.index)


def kf_hedge_ratio(x: pd.Series, y: pd.Series) -> np.ndarray:
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov,
    )
    state_means, _ = kf.filter(y.values)
    return -state_means


def estimate_half_life(spread: pd.Series) -> int:
    spread = spread.dropna()
    if len(spread) < 3:
        return 1
    X = spread.shift().iloc[1:].to_frame().assign(const=1)
    y = spread.diff().iloc[1:]
    beta = (np.linalg.inv(X.T @ X) @ X.T @ y).iloc[0]
    if beta == 0 or not np.isfinite(beta):
        return 1
    halflife = int(round(-np.log(2) / beta, 0))
    return max(halflife, 1)


def get_quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    return index.to_series().resample("QE").last().dropna().index


def build_period_positions(
    z_score: pd.Series,
    entry_z: float,
    entry_window_months: int,
) -> pd.Series:
    """Replicate the notebook entry/exit rules within a trading window."""
    if z_score.empty:
        return pd.Series(dtype=float, index=z_score.index)

    zeros = pd.Series(0.0, index=z_score.index)

    # Identify Entry Signals
    # Short Spread (-1) when Z > entry_z (Spread is too high, expect reversion down)
    # Long Spread (+1) when Z < -entry_z (Spread is too low, expect reversion up)
    is_above = z_score > entry_z
    is_below = z_score < -entry_z

    # Identify Exit Signals (Crossing Zero)
    cross_zero = np.sign(z_score) != np.sign(z_score.shift(1).bfill())

    # Create event series
    # We only care about the first instance of crossing the threshold
    entry_signals = pd.Series(np.nan, index=z_score.index, dtype=float)
    
    # We need to detect the crossing to avoid repetitive signals, 
    # though filtering adjacent duplicates handles that too.
    # Using raw boolean masks for events:
    entry_signals.loc[is_above & ~is_above.shift(1).fillna(False)] = -1.0
    entry_signals.loc[is_below & ~is_below.shift(1).fillna(False)] = 1.0

    exit_signals = pd.Series(np.nan, index=z_score.index, dtype=float)
    exit_signals.loc[cross_zero] = 0.0

    # Combine signals
    trades = pd.concat([entry_signals.dropna(), exit_signals.dropna()])
    if trades.empty:
        return zeros
    
    # Sort by date
    trades = trades.sort_index()
    
    # Remove adjacent duplicates (e.g., 1 -> 1 -> 0 -> 0)
    trades = trades.loc[trades.shift() != trades]

    # Enforce window constraints
    # Entries must happen within the first `entry_window_months`
    first_start = z_score.index.min()
    first_end = (
        first_start
        + pd.DateOffset(months=entry_window_months)
        - pd.DateOffset(days=1)
    )
    
    # Slice events to the entry window
    # Note: We must allow for an exit event that happens *after* the entry window 
    # if we are currently holding a position.
    
    entry_window_events = trades.loc[first_start:first_end]
    
    if entry_window_events.empty:
        return zeros
        
    # If the first event in the window is an exit (0), ignore it (we start flat)
    if entry_window_events.iloc[0] == 0.0:
        entry_window_events = entry_window_events.iloc[1:]
        
    if entry_window_events.empty:
        return zeros

    final_events = entry_window_events.copy()

    # If the last event in the entry window leaves us open (non-zero),
    # we look for the next exit event in the full timeline
    if entry_window_events.iloc[-1] != 0.0:
        # Look for events after the entry window
        future_events = trades.loc[first_end:] 
        # Find the first '0' (exit)
        future_exits = future_events[future_events == 0.0].head(1)
        if not future_exits.empty:
            final_events = pd.concat([final_events, future_exits]).sort_index()
            # Ensure no duplicates at the boundary
            final_events = final_events.loc[final_events.shift() != final_events]

    # Reconstruct positions series
    positions = zeros.copy()
    positions.loc[final_events.index] = final_events.values
    
    # Forward fill positions
    # We mask the area before the first event to 0 (already 0 initialized)
    # Then ffill from there.
    # However, since we initialized with zeros, simple ffill might overwrite 0s with 0s.
    # We want to hold the state.
    
    # Create a sparse series with just the events
    sparse_positions = pd.Series(np.nan, index=z_score.index, dtype=float)
    sparse_positions.loc[final_events.index] = final_events.values
    
    # Forward fill. 
    # Note: We need to ensure pre-first-event is 0. 
    # If the first event is at index 5, index 0-4 should be 0.
    filled_positions = sparse_positions.ffill().fillna(0.0)
    
    return filled_positions


def backtest_pair(
    config: dict,
    save_output: bool = True,
    output_tag: str | None = None,
) -> pd.DataFrame:
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
    os.makedirs(output_dir, exist_ok=True)

    y_symbol = config.get("symbol_y", "BNB")
    x_symbol = config.get("symbol_x", "SOL")

    y = load_price_series(y_symbol, config, BASE_DIR)
    x = load_price_series(x_symbol, config, BASE_DIR)

    df = pd.concat([y, x], axis=1).dropna()
    df = df.sort_index()
    if df.empty:
        raise ValueError("No overlapping data between the two symbols.")

    lookback_days = int(config.get("lookback_days", 730))
    trade_window_months = int(config.get("trade_window_months", 6))
    entry_window_months = int(config.get("entry_window_months", 3))
    entry_z = float(config.get("entry_z", 2.0))

    results = pd.DataFrame(
        {
            "y": df[y_symbol],
            "x": df[x_symbol],
            "y_smooth": np.nan,
            "x_smooth": np.nan,
            "hedge_ratio": np.nan,
            "spread": np.nan,
            "z_score": np.nan,
            "position": 0.0,
        },
        index=df.index,
    )

    for test_end in get_quarter_ends(df.index):
        trading_start = test_end + pd.DateOffset(days=1)
        lookback_start = trading_start - pd.DateOffset(days=lookback_days)
        trading_end = (
            trading_start
            + pd.DateOffset(months=trade_window_months)
            - pd.DateOffset(days=1)
        )
        if lookback_start < df.index.min() or trading_end > df.index.max():
            continue

        period_prices = df.loc[lookback_start:trading_end, [y_symbol, x_symbol]]
        if period_prices.empty:
            continue

        y_smooth = kf_smoother(period_prices[y_symbol])
        x_smooth = kf_smoother(period_prices[x_symbol])

        hedge_states = kf_hedge_ratio(x_smooth, y_smooth)
        hedge_ratio = pd.Series(hedge_states[:, 0], index=period_prices.index)

        spread = period_prices[y_symbol] + period_prices[x_symbol] * hedge_ratio
        lookback_spread = spread.loc[lookback_start:test_end]
        if lookback_spread.empty:
            continue

        half_life = estimate_half_life(lookback_spread)
        max_window = len(lookback_spread)
        if max_window < 1:
            continue
        window = min(2 * half_life, max_window)

        rolling = spread.rolling(window=window, min_periods=window)
        z_score = (spread - rolling.mean()) / rolling.std()

        trade_slice = slice(trading_start, trading_end)
        trade_index = spread.loc[trade_slice].index
        if trade_index.empty:
            continue
        trade_z = z_score.loc[trade_slice]
        position = build_period_positions(
            trade_z, entry_z, entry_window_months
        )

        period_series = {
            "y_smooth": y_smooth.loc[trade_index],
            "x_smooth": x_smooth.loc[trade_index],
            "hedge_ratio": hedge_ratio.loc[trade_index],
            "spread": spread.loc[trade_index],
            "z_score": trade_z,
            "position": position,
        }
        for column, series in period_series.items():
            results.loc[trade_index, column] = series

    returns_y = df[y_symbol].pct_change()
    returns_x = df[x_symbol].pct_change()
    hr_prev = results["hedge_ratio"].shift(1)
    spread_return = (returns_y + hr_prev * returns_x) / (1 + hr_prev.abs())
    strategy_return = results["position"].shift(1) * spread_return
    strategy_return = strategy_return.fillna(0.0)

    fee_rate = float(config.get("fee_rate", 0.0))
    turnover = results["position"].diff().abs().fillna(0.0)
    fee_cost = fee_rate * turnover
    strategy_return = strategy_return - fee_cost

    start_equity = float(config.get("start_equity", 1000.0))
    equity = start_equity * (1 + strategy_return).cumprod()

    results["strategy_return"] = strategy_return
    results["equity"] = equity
    results["turnover"] = turnover

    if save_output:
        out_path = get_output_path(BASE_DIR, config, y_symbol, x_symbol, output_tag)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results.reset_index().to_feather(out_path)

    return results


def plot_equity(results: pd.DataFrame, output_dir: str, name: str, show_plot: bool) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["equity"], label="Equity", linewidth=1.5)
    plt.title(f"Jansen Method Equity ({name})")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"equity_{name}.png")
    plt.savefig(path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jansen method pair trading backtest (BNB/SOL)."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
        help="Path to config.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))

    results = backtest_pair(config)
    name = f"{config.get('symbol_y', 'BNB')}_{config.get('symbol_x', 'SOL')}"
    show_plot = bool(config.get("show_plot", True))
    plot_equity(results, output_dir, name, show_plot)

    final_equity = results["equity"].iloc[-1]
    print(f"Final equity for {name}: {final_equity:.2f} USD")


if __name__ == "__main__":
    main()

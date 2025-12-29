import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import load_config, resolve_path

try:
    from pykalman import KalmanFilter
except ImportError as exc:
    raise ImportError(
        "pykalman is required. Please install with: pip install pykalman"
    ) from exc


def load_price_series(symbol: str, config: dict, base_dir: str) -> pd.Series:
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    data_dir = resolve_path(base_dir, config.get("data_dir", "../data/feather"))
    filename = f"{symbol}{quote}_{interval}.feather"
    path = os.path.join(data_dir, filename)
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


def build_positions(z_score: pd.Series, entry_z: float) -> pd.Series:
    z = z_score.values
    pos = np.zeros(len(z), dtype=float)
    for i in range(1, len(z)):
        if not np.isfinite(z[i]):
            pos[i] = 0.0
            continue
        prev_pos = pos[i - 1]
        new_pos = prev_pos
        if prev_pos == 0:
            if z[i] > entry_z:
                new_pos = -1.0
            elif z[i] < -entry_z:
                new_pos = 1.0
        else:
            if np.isfinite(z[i - 1]) and np.sign(z[i]) != np.sign(z[i - 1]):
                new_pos = 0.0
        pos[i] = new_pos
    return pd.Series(pos, index=z_score.index)


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
    if df.empty:
        raise ValueError("No overlapping data between the two symbols.")

    y_smooth = kf_smoother(df[y_symbol])
    x_smooth = kf_smoother(df[x_symbol])

    hedge_states = kf_hedge_ratio(x_smooth, y_smooth)
    hedge_ratio = pd.Series(hedge_states[:, 0], index=df.index)

    spread = df[y_symbol] + df[x_symbol] * hedge_ratio

    lookback_days = int(config.get("lookback_days", 730))
    lookback_days = min(lookback_days, len(spread))
    half_life = estimate_half_life(spread.iloc[:lookback_days])
    window = max(2, min(2 * half_life, lookback_days))

    rolling = spread.rolling(window=window, min_periods=window)
    z_score = (spread - rolling.mean()) / rolling.std()

    entry_z = float(config.get("entry_z", 2.0))
    position = build_positions(z_score, entry_z)

    returns_y = df[y_symbol].pct_change()
    returns_x = df[x_symbol].pct_change()
    hr_prev = hedge_ratio.shift(1)
    spread_return = (returns_y + hr_prev * returns_x) / (1 + hr_prev.abs())
    strategy_return = position.shift(1) * spread_return
    strategy_return = strategy_return.fillna(0.0)

    fee_rate = float(config.get("fee_rate", 0.0))
    turnover = position.diff().abs().fillna(0.0)
    fee_cost = fee_rate * turnover
    strategy_return = strategy_return - fee_cost

    start_equity = float(config.get("start_equity", 1000.0))
    equity = start_equity * (1 + strategy_return).cumprod()

    results = pd.DataFrame(
        {
            "y": df[y_symbol],
            "x": df[x_symbol],
            "y_smooth": y_smooth,
            "x_smooth": x_smooth,
            "hedge_ratio": hedge_ratio,
            "spread": spread,
            "z_score": z_score,
            "position": position,
            "strategy_return": strategy_return,
            "equity": equity,
            "turnover": turnover,
        }
    )

    if save_output:
        suffix = f"_{output_tag}" if output_tag else ""
        out_path = os.path.join(
            output_dir, f"jansen_backtest_{y_symbol}_{x_symbol}{suffix}.feather"
        )
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

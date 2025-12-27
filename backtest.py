import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pair_id(pair: Dict) -> str:
    return f"{pair['y_symbol']}__{pair['x_symbol']}"


def get_dirs(config: Dict) -> Tuple[str, str]:
    data_dir = config.get("data_dir", "data")
    intermediate_dir = config.get("intermediate_dir") or os.path.join(
        data_dir, "intermediate"
    )
    output_dir = config.get("output_dir") or os.path.join(data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return intermediate_dir, output_dir


def load_coint_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cointegration data: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    return df.sort_values("open_time_dt").set_index("open_time_dt")


def load_bands_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing band data: {path}")
    df = pd.read_feather(path)
    df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
    return df.sort_values("open_time_dt").set_index("open_time_dt")


def backtest_pair(pair: Dict, config: Dict) -> pd.DataFrame:
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 30))
    intermediate_dir, output_dir = get_dirs(config)

    coint_path = os.path.join(
        intermediate_dir,
        f"coint_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    bands_path = os.path.join(
        intermediate_dir,
        f"bands_{pair_id(pair)}_{interval}_w{window}.feather",
    )

    coint_df = load_coint_data(coint_path)
    bands_df = load_bands_data(bands_path)

    df = coint_df.join(bands_df, how="left", rsuffix="_bands")
    n = len(df)
    if n == 0:
        raise ValueError("No data available for backtest.")

    pos_arr = np.zeros(n)
    epsilon = df["epsilon"].values
    lower = df["lower"].values
    upper = df["upper"].values
    mu = df["mu"].values

    for i in range(1, n):
        if not (
            np.isfinite(epsilon[i])
            and np.isfinite(lower[i])
            and np.isfinite(upper[i])
            and np.isfinite(mu[i])
        ):
            pos_arr[i] = 0.0
            continue

        prev_pos = pos_arr[i - 1]
        new_pos = prev_pos

        if prev_pos == 0:
            if epsilon[i] < lower[i]:
                new_pos = 1
            elif epsilon[i] > upper[i]:
                new_pos = -1
        elif prev_pos == 1 and epsilon[i] >= mu[i]:
            new_pos = 0
        elif prev_pos == -1 and epsilon[i] <= mu[i]:
            new_pos = 0

        pos_arr[i] = new_pos

    flip_signals = bool(config.get("flip_signals", False))
    if flip_signals:
        pos_arr = -pos_arr
    df["position"] = pos_arr

    returns_y = df["y_close"].pct_change()
    returns_x = df["x_close"].pct_change()
    beta_series = df["beta"].shift(1)

    spread_return = (returns_y - beta_series * returns_x) / (1 + beta_series.abs())
    strategy_return = df["position"].shift(1) * spread_return
    strategy_return = strategy_return.fillna(0.0)

    fee_rate = float(config.get("fee_rate", 0.001))
    turnover = df["position"].diff().abs().fillna(0.0)
    fee_cost = fee_rate * turnover
    strategy_return = strategy_return - fee_cost

    start_equity = float(config.get("start_equity", 1000))
    equity = start_equity * (1 + strategy_return).cumprod()

    df["strategy_return"] = strategy_return
    df["equity"] = equity
    df["turnover"] = turnover
    df["fee_cost"] = fee_cost

    out_path = os.path.join(
        output_dir,
        f"backtest_{pair_id(pair)}_{interval}_w{window}.feather",
    )
    df.reset_index().to_feather(out_path)
    print(f"Saved backtest results to {out_path}")
    return df


def plot_equity(results: pd.DataFrame, pair_name: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results["equity"], label="Equity")
    plt.title(f"Pair Trading Equity ({pair_name})")
    plt.xlabel("Date")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(config_path: str = "config.json") -> None:
    config = load_config(config_path)
    pairs = config.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs configured in config.json")

    for pair in pairs:
        pair_name = pair.get("name") or pair_id(pair)
        print(f"Running backtest for {pair_name}...")
        results = backtest_pair(pair, config)
        print(f"Final equity for {pair_name}: {results['equity'].iloc[-1]:.2f} USD")
        plot_equity(results, pair_name)


if __name__ == "__main__":
    main()

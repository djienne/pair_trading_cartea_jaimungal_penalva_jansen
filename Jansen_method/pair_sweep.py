import argparse
import os
import sys
import random

import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import backtest_pair, plot_equity
from utils import (
    compute_signature,
    list_symbols,
    load_config,
    resolve_path,
    resolve_thresholds,
    summarize_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank available symbol pairs by best final equity."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated z-score thresholds to test per pair.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    interval = config.get("interval", "1d")
    quote = config.get("quote", "USDT")
    data_dir = resolve_path(BASE_DIR, config.get("data_dir", "../data/feather"))

    min_history_days = int(config.get("min_history_days", 1000))
    symbols = list_symbols(data_dir, interval, quote, min_history_days)
    if len(symbols) < 2:
        raise ValueError("Not enough symbols found to build pairs.")

    thresholds = resolve_thresholds(config, args.thresholds)
    rows = []
    cache_dir = os.path.join(BASE_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    code_paths = [
        os.path.join(BASE_DIR, "jansen_backtest.py"),
        os.path.join(BASE_DIR, "pair_sweep.py"),
        os.path.join(BASE_DIR, "utils.py"),
    ]
    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    processed = 0
    best_so_far = None
    best_pair = None
    print(f"Testing {total_pairs} pair combinations...")

    pairs = []
    for i, y_symbol in enumerate(symbols):
        for x_symbol in symbols[i + 1 :]:
            pairs.append((y_symbol, x_symbol))

    random.shuffle(pairs)

    for y_symbol, x_symbol in pairs:
        print(f"Starting pair {y_symbol}/{x_symbol}...")
        best_equity = None
        best_z = None
        best_avg_log_return = None
        data_paths = [
            os.path.join(
                data_dir,
                f"{y_symbol}{quote}_{interval}.feather",
            ),
            os.path.join(
                data_dir,
                f"{x_symbol}{quote}_{interval}.feather",
            ),
        ]
        for z in thresholds:
            run_config = dict(config)
            run_config["symbol_y"] = y_symbol
            run_config["symbol_x"] = x_symbol
            run_config["entry_z"] = float(z)
            run_config["output_dir"] = cache_dir
            signature = compute_signature(run_config, data_paths + code_paths)
            cache_key = f"{y_symbol}_{x_symbol}_z{z}".replace(".", "p")
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")
            cached = None
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
            if cached and cached.get("signature") == signature:
                summary = cached["summary"]
                avg_log_return = summary["avg_log_return"]
            else:
                results = backtest_pair(
                    run_config, save_output=True, output_tag=f"z{z}"
                )
                summary = summarize_results(results)
                avg_log_return = summary["avg_log_return"]
                output_path = os.path.join(
                    cache_dir,
                    f"jansen_backtest_{y_symbol}_{x_symbol}_z{z}.feather",
                )
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "signature": signature,
                            "summary": summary,
                            "output_path": output_path,
                        },
                        f,
                        indent=2,
                    )
            if (
                best_avg_log_return is None
                or avg_log_return > best_avg_log_return
            ):
                best_avg_log_return = avg_log_return
                best_equity = summary["final_equity"]
                best_z = z
        processed += 1
        rows.append(
            {
                "symbol_y": y_symbol,
                "symbol_x": x_symbol,
                "best_entry_z": best_z,
                "avg_log_return": best_avg_log_return,
                "final_equity": best_equity,
            }
        )
        if best_so_far is None or best_avg_log_return > best_so_far:
            best_so_far = best_avg_log_return
            best_pair = (y_symbol, x_symbol, best_z)
        if processed % 5 == 0 or processed == total_pairs:
            print(
                (
                    f"Processed {processed}/{total_pairs} pairs. "
                    f"Latest: {y_symbol}/{x_symbol} best_z={best_z} "
                    f"avg_log_return={best_avg_log_return:.6f}"
                )
            )
            if best_pair is not None:
                print(
                    (
                        f"Best so far: {best_pair[0]}/{best_pair[1]} "
                        f"best_z={best_pair[2]} avg_log_return={best_so_far:.6f}"
                    )
                )

    table = pd.DataFrame(rows).sort_values("avg_log_return", ascending=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if best_pair is not None:
        best_y, best_x, best_z = best_pair
        output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
        name = f"{best_y}_{best_x}_z{best_z}"
        show_plot = bool(config.get("show_plot", True))
        cache_key = f"{best_y}_{best_x}_z{best_z}".replace(".", "p")
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        best_results = None
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            output_path = cached.get("output_path")
            if output_path and os.path.exists(output_path):
                df = pd.read_feather(output_path)
                df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
                best_results = df.sort_values("open_time_dt").set_index(
                    "open_time_dt"
                )
        if best_results is None:
            best_config = dict(config)
            best_config["symbol_y"] = best_y
            best_config["symbol_x"] = best_x
            best_config["entry_z"] = float(best_z)
            best_config["output_dir"] = cache_dir
            best_results = backtest_pair(
                best_config, save_output=True, output_tag=f"z{best_z}"
            )
        best_results.reset_index().to_feather(
            os.path.join(
                output_dir,
                f"jansen_backtest_{best_y}_{best_x}_z{best_z}.feather",
            )
        )
        plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()

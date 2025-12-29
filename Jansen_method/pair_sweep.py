import argparse
import os
import sys
import random

import json
import numpy as np
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
    get_data_path,
    get_output_path,
    get_cache_path,
    ResultTracker,
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
    
    pairs = []
    for i, y_symbol in enumerate(symbols):
        for x_symbol in symbols[i + 1 :]:
            pairs.append((y_symbol, x_symbol))

    random.shuffle(pairs)
    total_pairs = len(pairs)
    
    min_trades = int(config.get("min_trades", 20))
    print(f"Testing {total_pairs} pair combinations (min_trades={min_trades})...")

    processed = 0
    # Track the global best pair
    global_tracker = ResultTracker(min_trades=min_trades)
    best_pair_info = None

    for y_symbol, x_symbol in pairs:
        print(f"Starting pair {y_symbol}/{x_symbol}...")
        
        # Track best z for this specific pair
        pair_tracker = ResultTracker(min_trades=min_trades)
        
        data_paths = [
            get_data_path(BASE_DIR, y_symbol, config),
            get_data_path(BASE_DIR, x_symbol, config),
        ]
        
        for z in thresholds:
            run_config = dict(config)
            run_config["symbol_y"] = y_symbol
            run_config["symbol_x"] = x_symbol
            run_config["entry_z"] = float(z)
            run_config["output_dir"] = cache_dir
            
            signature = compute_signature(run_config, data_paths + code_paths)
            cache_path = get_cache_path(cache_dir, y_symbol, x_symbol, float(z))
            
            summary = None
            cached = None
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
            
            if cached and cached.get("signature") == signature:
                summary = cached["summary"]
            else:
                results = backtest_pair(
                    run_config, save_output=True, output_tag=f"z{z}"
                )
                summary = summarize_results(results)
                # Output path logic is now handled inside backtest_pair, but we store the expected path in cache
                output_path = get_output_path(BASE_DIR, run_config, y_symbol, x_symbol, f"z{z}")
                
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
            
            # Update pair-level best
            pair_tracker.update(summary, float(z))

        # End of thresholds loop for this pair
        processed += 1
        best_of_pair = pair_tracker.get_best()
        
        if best_of_pair:
            # Check if this pair beats the global best
            # We reconstruct a summary dict to pass to the global tracker
            # (ResultTracker expects a summary dict with 'sharpe', 'trades' etc)
            # The get_best() returns exactly those keys + best_entry_z.
            
            # Update global tracker
            # We use a slightly hacky way: pass the best_of_pair dict as 'summary' 
            # since it has 'sharpe' and 'trades' keys.
            if global_tracker.update(best_of_pair, best_of_pair["best_entry_z"]):
                best_pair_info = (y_symbol, x_symbol, best_of_pair["best_entry_z"])

            rows.append({
                "symbol_y": y_symbol,
                "symbol_x": x_symbol,
                **best_of_pair
            })
        else:
            # No valid config found for this pair (e.g. not enough trades)
            rows.append({
                "symbol_y": y_symbol,
                "symbol_x": x_symbol,
                "best_entry_z": None,
                "sharpe": None,
                "avg_log_return": None,
                "final_equity": None,
            })

        if processed % 5 == 0 or processed == total_pairs:
            current_best_sharpe = global_tracker.best_sharpe if global_tracker.best_sharpe is not None else 0.0
            pair_sharpe = best_of_pair['sharpe'] if best_of_pair else 0.0
            print(
                (
                    f"Processed {processed}/{total_pairs} pairs. "
                    f"Latest: {y_symbol}/{x_symbol} sharpe={pair_sharpe:.4f} "
                    f"| Global Best Sharpe: {current_best_sharpe:.4f}"
                )
            )

    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if best_pair_info:
        best_y, best_x, best_z = best_pair_info
        output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
        name = f"{best_y}_{best_x}_z{best_z}"
        show_plot = bool(config.get("show_plot", True))
        
        # Load best results to plot
        # Try loading from file first
        best_output_path = get_output_path(BASE_DIR, config, best_y, best_x, f"z{best_z}".replace(".", "p")) # Wait, cache uses output_dir?
        # Actually backtest_pair uses cache_dir as output_dir in the loop above: run_config["output_dir"] = cache_dir
        # So the file is in cache_dir.
        
        # Let's reconstruct the path where it was saved
        # run_config above used cache_dir.
        cached_file_path = os.path.join(cache_dir, f"jansen_backtest_{best_y}_{best_x}_z{best_z}.feather")
        
        best_results = None
        if os.path.exists(cached_file_path):
            df = pd.read_feather(cached_file_path)
            df["open_time_dt"] = pd.to_datetime(df["open_time_dt"])
            best_results = df.sort_values("open_time_dt").set_index("open_time_dt")
        
        if best_results is None:
             # Re-run if missing
            best_config = dict(config)
            best_config["symbol_y"] = best_y
            best_config["symbol_x"] = best_x
            best_config["entry_z"] = float(best_z)
            best_config["output_dir"] = cache_dir
            best_results = backtest_pair(
                best_config, save_output=False
            )
            
        # Save final plot to the actual output dir
        plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()

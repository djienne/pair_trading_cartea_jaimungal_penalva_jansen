import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import backtest_pair, plot_equity
from utils import load_config, resolve_thresholds, summarize_results, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep z-score entry thresholds for Jansen backtest."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated z-score thresholds to test.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    thresholds = resolve_thresholds(config, args.thresholds)

    rows = []
    for z in thresholds:
        run_config = dict(config)
        run_config["entry_z"] = float(z)
        results = backtest_pair(run_config, save_output=False)
        summary = summarize_results(results)
        rows.append(
            {
                "entry_z": z,
                "avg_log_return": summary["avg_log_return"],
                "sharpe": summary["sharpe"],
                "trades": summary["trades"],
            }
        )

    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    
    min_trades = int(config.get("min_trades", 20))
    # Filter for minimum trade requirement
    valid_results = table[table["trades"] >= min_trades]

    if valid_results.empty:
        print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        print(f"\nNo result met the minimum trade requirement ({min_trades} trades).")
        return

    best = valid_results.iloc[0]

    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(
        f"\nBest entry_z by sharpe ratio (min {min_trades} trades): {best['entry_z']} "
        f"(sharpe={best['sharpe']:.4f}, trades={best['trades']})"
    )

    # Re-run best case to generate plot
    best_config = dict(config)
    best_config["entry_z"] = float(best["entry_z"])
    best_results = backtest_pair(best_config, save_output=True)
    
    output_dir = resolve_path(BASE_DIR, config.get("output_dir", "output"))
    name = f"{best_config.get('symbol_y', 'BNB')}_{best_config.get('symbol_x', 'SOL')}_z{best['entry_z']}"
    show_plot = bool(config.get("show_plot", True))
    
    plot_equity(best_results, output_dir, name, show_plot)


if __name__ == "__main__":
    main()

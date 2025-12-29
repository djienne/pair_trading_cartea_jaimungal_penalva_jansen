import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from jansen_backtest import backtest_pair
from utils import load_config, resolve_thresholds, summarize_results


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

    table = pd.DataFrame(rows).sort_values("avg_log_return", ascending=False)
    best = table.iloc[0]

    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(
        f"\nBest entry_z by avg log return: {best['entry_z']} "
        f"(avg_log_return={best['avg_log_return']:.6f})"
    )


if __name__ == "__main__":
    main()

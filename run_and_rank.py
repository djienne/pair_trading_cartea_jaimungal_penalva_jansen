import os
import sys
import pandas as pd
import pyarrow.feather as paf
import itertools
from typing import Dict, List, Optional, Tuple

# Add current directory to path
sys.path.append(os.getcwd())

# Import necessary functions
try:
    from bayesian_coint import calibrate_pair as bayesian_calibrate_pair
    from coint_calibrate import calibrate_pair as ols_calibrate_pair
    from ou_calibrate import calibrate_pair as ou_calibrate_pair
    from band_calc import calculate_bands
    from backtest import backtest_pair, plot_equity
    from utils import load_config, pair_id, get_dirs, periods_per_year, annualized_sharpe
except ImportError:
    print("Error: Could not import required modules. Ensure you are in the project root.")
    sys.exit(1)


def get_available_symbols(config: Dict, min_rows: int) -> List[str]:
    feather_dir, _, _ = get_dirs(config)
    interval = config.get("candle_interval", "1d")
    suffix = f"_{interval}.feather"
    
    available = []
    if not os.path.exists(feather_dir):
        return []
        
    for f in os.listdir(feather_dir):
        if f.endswith(suffix):
            path = os.path.join(feather_dir, f)
            try:
                # Count rows cheaply by reading a single column rather than the whole frame.
                try:
                    n_rows = paf.read_table(path, columns=["open_time"]).num_rows
                except Exception:
                    n_rows = len(pd.read_feather(path))
                if n_rows >= min_rows:
                    symbol = f[: -len(suffix)]
                    available.append(symbol)
            except Exception:
                pass
    return sorted(list(set(available)))

def process_single_pair(args: Tuple[str, str, Dict]) -> Optional[Dict]:
    sym_y, sym_x, config = args
    pair = {
        "y_symbol": sym_y,
        "x_symbol": sym_x,
        "name": f"{sym_y}-{sym_x}"
    }
    name = pair["name"]
    start_equity = float(config.get("start_equity", 1000))

    try:
        # Use a copy of config to be absolutely safe in parallel
        local_config = dict(config)
        verbose = bool(local_config.get("verbose", False))

        if verbose:
            print(f"DEBUG: Starting {name}")

        # Choose cointegration method based on config
        coint_method = local_config.get("cointegration_method", "bayesian").lower()
        if coint_method == "ols":
            ols_calibrate_pair(pair, local_config)
        else:
            bayes_n_jobs = int(local_config.get("bayesian_config", {}).get("n_jobs", 1))
            bayesian_calibrate_pair(pair, local_config, n_jobs_override=bayes_n_jobs, show_progress=False)
        if verbose:
            print(f"DEBUG: Cointegration done for {name}")

        ou_calibrate_pair(pair, local_config)
        if verbose:
            print(f"DEBUG: OU done for {name}")

        calculate_bands(pair, local_config)
        if verbose:
            print(f"DEBUG: Bands done for {name}")

        df = backtest_pair(pair, local_config)
        if verbose:
            print(f"DEBUG: Backtest done for {name}")

        if df is None or df.empty:
            if verbose:
                print(f"DEBUG: Dataframe empty for {name}")
            return None

        final_equity = float(df["equity"].iloc[-1])
        print(f"Final equity for {name}: {final_equity:.2f} USD")
        ret_pct = (final_equity / start_equity - 1) * 100
        trades = int(df["turnover"].gt(0).sum()) if "turnover" in df.columns else 0

        # Annualized Sharpe from per-bar strategy returns (365/yr for daily crypto, shared helper).
        interval = local_config.get("candle_interval", "1d")
        ppy = periods_per_year(interval, local_config.get("periods_per_year"))
        returns = df["strategy_return"] if "strategy_return" in df.columns else df["equity"].pct_change()
        sharpe = annualized_sharpe(returns, ppy)

        return {
            "Pair": name,
            "Final Equity": final_equity,
            "Return %": ret_pct,
            "Sharpe": sharpe,
            "Trades": trades,
            "y_symbol": sym_y,
            "x_symbol": sym_x
        }
    except Exception as e:
        print(f"Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("config.json not found.")
        return

    config = load_config(config_path)
    
    # Global overrides for ranking phase
    config["log_adf_each_window"] = False
    config["show_plots"] = False
    
    # Use 1000 days of history as requested
    min_rows = 1000
    
    print("--- Scanning Data ---")
    symbols = get_available_symbols(config, min_rows)
    print(f"Found {len(symbols)} symbols with >= {min_rows} rows.")
    
    if len(symbols) < 2:
        print("Need at least 2 symbols to form a pair.")
        return

    # Generate unique combinations
    pairs_list = list(itertools.combinations(symbols, 2))
    print(f"Generated {len(pairs_list)} unique pairs to test.")
    
    results = []

    coint_method = config.get("cointegration_method", "bayesian").lower()
    ranking_metric = config.get("ranking_metric", "sharpe").lower()

    # Validate ranking_metric
    if ranking_metric not in ("sharpe", "returns"):
        print(f"Warning: Invalid ranking_metric '{ranking_metric}', defaulting to 'sharpe'")
        ranking_metric = "sharpe"

    sort_col = "Sharpe" if ranking_metric == "sharpe" else "Final Equity"

    print(f"\n--- Starting Bulk Backtest (Sequential) ---")
    print(f"Cointegration method: {coint_method.upper()}")
    print(f"Ranking metric: {ranking_metric.upper()}")
    print(f"Processing pairs one by one.")
    if coint_method == "bayesian":
        bayes_n_jobs = int(config.get("bayesian_config", {}).get("n_jobs", 1))
        print(f"Each pair's Bayesian calibration will use {bayes_n_jobs} CPU thread(s) (parallel windows).")

    best_metric_so_far = -float('inf')
    best_pair_so_far = None

    tasks = [(sym_y, sym_x, config) for sym_y, sym_x in pairs_list]

    count = 0
    total = len(tasks)
    for task in tasks:
        res = process_single_pair(task)
        count += 1
        if res:
            results.append(res)
            current_metric = res[sort_col]
            if current_metric > best_metric_so_far:
                best_metric_so_far = current_metric
                best_pair_so_far = res["Pair"]

        if count % 10 == 0 or count == total:
            if best_pair_so_far:
                metric_fmt = f"{best_metric_so_far:.2f}" if ranking_metric == "returns" else f"{best_metric_so_far:.3f}"
                metric_label = "USD" if ranking_metric == "returns" else "Sharpe"
                best_info = f" | Best so far: {best_pair_so_far} ({metric_fmt} {metric_label})"
            else:
                best_info = ""
            print(f"Progress: {count}/{total} (Valid: {len(results)}){best_info}")

    # Final Ranking
    metric_name = "Sharpe Ratio" if ranking_metric == "sharpe" else "Equity"
    print(f"\n--- Final Ranking ({metric_name}) ---")
    if not results:
        print("No valid results generated.")
        return

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(sort_col, ascending=False).reset_index(drop=True)
    
    print(df_res.head(50).to_string(index=False))
    
    _, _, output_dir = get_dirs(config)
    rank_suffix = "sharpe" if ranking_metric == "sharpe" else "equity"
    rank_path = os.path.join(output_dir, f"ranking_{rank_suffix}_all.csv")
    df_res.to_csv(rank_path, index=False)
    print(f"\nSaved full ranking to {rank_path}")
    
    # Verification & Plotting for Best Pair
    best_row = df_res.iloc[0]
    best_pair_name = best_row["Pair"]
    
    print(f"\n--- Verifying Best Pair: {best_pair_name} ---")
    best_pair = {
        "y_symbol": best_row["y_symbol"],
        "x_symbol": best_row["x_symbol"],
        "name": best_pair_name
    }
    
    # Re-run in main process to confirm numbers and generate plot
    df_best = backtest_pair(best_pair, config)
    final_val = df_best["equity"].iloc[-1]
    print(f"Main Process Verification Equity: {final_val:.2f} USD")
    
    data_dir = config.get("data_dir", "data")
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    interval = config.get("candle_interval", "1d")
    window = int(config.get("rolling_window_days", 300))
    start_date = df_best.index[0].strftime('%Y%m%d')
    end_date = df_best.index[-1].strftime('%Y%m%d')
    
    rank_label = "SHARPE" if ranking_metric == "sharpe" else "EQUITY"
    filename = f"BEST_{rank_label}_{best_pair_name}_{interval}_w{window}_{start_date}-{end_date}.png"
    filename = filename.replace(os.path.sep, "_").replace(":", "")
    save_path = os.path.join(plots_dir, filename)
    
    try:
        plot_equity(df_best, best_pair_name, save_path, show_plot=False)
        print(f"Best equity plot saved to {save_path}")
    except Exception as e:
        print(f"Error plotting best pair: {e}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
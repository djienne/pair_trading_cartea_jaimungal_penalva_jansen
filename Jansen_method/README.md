# Jansen Method Pair Trading

This project implements a pairs trading strategy inspired by the "Pairs trading in practice" section of `pair_trading_in_practice.pdf`. It utilizes Kalman filtering to smooth price series and estimate a time-varying hedge ratio. A z-score is built from the resulting spread, and trades are executed when the z-score crosses defined thresholds.

## Features

- **Kalman Filter Smoothing**: Reduces noise in individual price series.
- **Dynamic Hedge Ratio**: Uses a Kalman Filter to estimate the cointegration relationship between two assets in real-time.
- **Mean Reversion Strategy**: Trades the z-score of the spread, entering at `entry_z` and exiting when the z-score changes sign (mean reversion).
- **Optimization Tools**: Scripts to sweep z-score thresholds and find the best-performing pairs.
- **Caching**: `pair_sweep.py` uses a signature-based caching mechanism to speed up repeated runs.

## Scripts

### 1. `jansen_backtest.py`
The core backtest engine for a single pair.
```powershell
python jansen_backtest.py
```
It reads `symbol_x` and `symbol_y` from `config.json`, runs the backtest, and saves the results/plots to the `output/` directory.

### 2. `zscore_sweep.py`
Sweeps a grid of z-score thresholds for the pair defined in `config.json`.
```powershell
python zscore_sweep.py --thresholds "1.0,1.5,2.0,2.5,3.0"
```
Useful for finding the optimal entry threshold for a specific pair. It ranks results by `avg_log_return`.

### 3. `pair_sweep.py`
Ranks all possible pairs from the available data based on their performance.
```powershell
python pair_sweep.py
```
- It filters symbols based on `min_history_days`.
- It tests each pair against the `threshold_grid` defined in `config.json`.
- Results are cached in the `cache/` folder using a SHA-256 signature of the data and code.
- It outputs a ranked table based on `avg_log_return` and generates an equity plot for the best-performing pair in `output/`.

## Configuration (`config.json`)

- `data_dir`: Path to the directory containing `.feather` price files.
- `output_dir`: Path where backtest results and plots are saved.
- `interval`: Candle interval (e.g., `1d`).
- `quote`: Quote currency (e.g., `USDT`).
- `symbol_y`: Primary symbol (Lead).
- `symbol_x`: Secondary symbol (Lag/Hedge).
- `start_equity`: Initial capital.
- `entry_z`: Z-score threshold for entering a trade.
- `threshold_grid`: List of z-score thresholds to test in sweep scripts.
- `min_history_days`: Minimum data points required for a symbol to be included in `pair_sweep.py`.
- `fee_rate`: Transaction fee rate (e.g., `0.001` for 0.1%).
- `lookback_days`: Lookback window used for estimating half-life and rolling spread statistics.
- `show_plot`: Boolean to toggle the display of the equity curve window.

## Installation

Ensure you have the following dependencies installed:

```powershell
pip install pandas pykalman matplotlib pyarrow
```

The data files should be in Feather format with a `close` price column and an `open_time_dt` datetime column.
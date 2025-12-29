import json
import os
import hashlib

import numpy as np
import pandas as pd


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def parse_thresholds(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(p) for p in parts]


def resolve_thresholds(config: dict, raw_thresholds: str | None) -> list[float]:
    if raw_thresholds:
        return parse_thresholds(raw_thresholds)
    grid = config.get("threshold_grid", [1.0, 1.5, 2.0, 2.5, 3.0])
    return [float(value) for value in grid]


def list_symbols(
    data_dir: str,
    interval: str,
    quote: str,
    min_history_days: int = 0,
) -> list[str]:
    symbols = []
    suffix = f"_{interval}.feather"
    for name in os.listdir(data_dir):
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)]
        if not base.endswith(quote):
            continue
        symbol = base[: -len(quote)]
        if min_history_days:
            path = os.path.join(data_dir, name)
            df = pd.read_feather(path, columns=["open_time_dt"])
            if df["open_time_dt"].nunique() < min_history_days:
                continue
        symbols.append(symbol)
    return sorted(set(symbols))


def summarize_results(results: pd.DataFrame) -> dict:
    returns = results["strategy_return"].fillna(0.0)
    log_returns = np.log1p(returns)
    total_return = results["equity"].iloc[-1] / results["equity"].iloc[0] - 1
    avg_log_return = log_returns.mean()
    std = returns.std()
    sharpe = np.nan
    if std and np.isfinite(std):
        sharpe = (returns.mean() / std) * np.sqrt(252)
    trades = results["position"].diff().abs().fillna(0.0).sum() / 2
    return {
        "final_equity": results["equity"].iloc[-1],
        "total_return": total_return,
        "avg_log_return": avg_log_return,
        "sharpe": sharpe,
        "trades": trades,
    }


def compute_signature(config: dict, paths: list[str]) -> str:
    payload = {
        "config": config,
        "files": [],
    }
    for path in paths:
        stat = os.stat(path)
        payload["files"].append(
            {
                "path": os.path.normpath(path),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        )
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()

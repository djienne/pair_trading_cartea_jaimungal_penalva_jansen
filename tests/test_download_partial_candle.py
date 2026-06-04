"""C9: drop_unclosed removes the currently-forming candle."""
import pandas as pd

from download_data import drop_unclosed


def test_drops_unclosed_last_bar():
    df = pd.DataFrame({"open_time": [0, 1, 2], "close_time": [100, 200, 300]})
    out = drop_unclosed(df, now_ms=250)  # last bar (close_time 300) has not closed yet
    assert list(out["close_time"]) == [100, 200]
    # Index is reset to a clean RangeIndex (feather requirement).
    assert list(out.index) == [0, 1]


def test_keeps_all_closed_bars():
    df = pd.DataFrame({"open_time": [0, 1, 2], "close_time": [100, 200, 300]})
    out = drop_unclosed(df, now_ms=1000)
    assert list(out["close_time"]) == [100, 200, 300]


def test_no_close_time_column_is_noop():
    df = pd.DataFrame({"open_time": [0, 1, 2]})
    out = drop_unclosed(df, now_ms=1000)
    assert len(out) == 3

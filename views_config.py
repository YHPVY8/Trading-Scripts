#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
from supabase import create_client, Client

SRC_TABLE = os.getenv("GC_INTRADAY_TABLE", "gc_30m")   # 30m bars input (EST wall time)
DST_TABLE = "gc_levels"

# ---------- Supabase connection ----------
def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or
        os.getenv("SUPABASE_KEY")
    )
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
    return create_client(url, key)

# ---------- Pull all rows via pagination ----------
def fetch_all_rows(supabase: Client, table: str, order_col: str = "time", page_size: int = 1000) -> pd.DataFrame:
    all_chunks = []
    start = 0
    while True:
        end = start + page_size - 1
        resp = (
            supabase.table(table)
            .select("*")
            .order(order_col, desc=False)   # oldest -> newest
            .range(start, end)
            .execute()
        )
        chunk = resp.data or []
        if not chunk:
            break
        all_chunks.extend(chunk)
        if len(chunk) < page_size:
            break
        start += page_size

    df = pd.DataFrame(all_chunks)
    if not df.empty and order_col in df.columns:
        # numeric coercion (robust)
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time", "high", "low", "close"]).reset_index(drop=True)
        # Keep source order (EST wall time)
        df = df.sort_values(order_col).reset_index(drop=True)
    return df

# ---------- Helpers (EST wall time, no tz conversion) ----------
def to_wall_time_and_globex_date(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Treat source 'time' as EST wall clock (matches your ES enrichment script)
    ts = pd.to_datetime(d["time"].astype(str), errors="coerce", utc=False)
    # If somehow tz-aware, strip tz to keep wall time semantics
    try:
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
    except Exception:
        pass

    d["time_et"] = ts
    d["tod"] = d["time_et"].dt.strftime("%H:%M")

    base_date = d["time_et"].dt.normalize()
    is_after_18 = d["time_et"].dt.hour >= 18
    d["globex_date"] = (base_date + pd.to_timedelta(is_after_18.astype(int), unit="D")).dt.date
    return d

def _window(df: pd.DataFrame, start_hm: str, end_hm: str) -> pd.DataFrame:
    """Inclusive time window using wall-clock strings HH:MM."""
    return df[(df["tod"] >= start_hm) & (df["tod"] <= end_hm)]

def _nan_to_none(x):
    return None if pd.isna(x) else float(x)

def _hi(window_df: pd.DataFrame):
    """Highest high in window, returns None if window empty or all NaN."""
    if window_df.empty:
        return None
    s = pd.to_numeric(window_df["high"], errors="coerce")
    mx = s.max(skipna=True)
    return _nan_to_none(mx)

def _lo(window_df: pd.DataFrame):
    """Lowest low in window, returns None if window empty or all NaN."""
    if window_df.empty:
        return None
    s = pd.to_numeric(window_df["low"], errors="coerce")
    mn = s.min(skipna=True)
    return _nan_to_none(mn)

def _safe_bool(expr):
    # Treat None as unknown -> None; otherwise cast to bool
    if expr is None:
        return None
    if isinstance(expr, float) and pd.isna(expr):
        return None
    return bool(expr)

def _touched(window_df: pd.DataFrame, level):
    """
    True if any bar in 'window_df' touches 'level' via low <= level <= high.
    Returns None if window or level is missing.
    """
    if window_df is None or window_df.empty or level is None:
        return None
    lows  = pd.to_numeric(window_df["low"],  errors="coerce")
    highs = pd.to_numeric(window_df["high"], errors="coerce")
    mask = (lows <= level) & (highs >= level)
    if mask.isna().all():
        return None
    return bool(mask.fillna(False).any())

# ---------- Compute per-session rows ----------
def compute_gc_levels(df30: pd.DataFrame) -> pd.DataFrame:
    d = to_wall_time_and_globex_date(df30)

    out_rows = []
    for gdate, gdf in d.groupby("globex_date", sort=True):
        # Full-session
        globex_hi = _hi(gdf)
        globex_lo = _lo(gdf)

        # Windows (inclusive, per spec)
        aib_df    = _window(gdf, "20:00", "21:30")
        eib_df    = _window(gdf, "03:00", "03:30")
        pre_df    = _window(gdf, "04:00", "08:00")
        adjib_df  = _window(gdf, "08:30", "09:00")
        am_df     = _window(gdf, "09:30", "11:30")
        pm_df     = _window(gdf, "12:00", "15:30")
        adjrth_df = _window(gdf, "08:30", "15:30")

        aibh = _hi(aib_df); aibl = _lo(aib_df)
        eibh = _hi(eib_df); eibl = _lo(eib_df)
        pre_hi = _hi(pre_df); pre_lo = _lo(pre_df)
        adj_ibh = _hi(adjib_df); adj_ibl = _lo(adjib_df)
        am_hi = _hi(am_df); am_lo = _lo(am_df)
        pm_hi = _hi(pm_df); pm_lo = _lo(pm_df)
        adj_rth_hi = _hi(adjrth_df); adj_rth_lo = _lo(adjrth_df)

        # aIB mid
        aib_mid = None
        if (aibh is not None) and (aibl is not None):
            rng = aibh - aibl
            if rng is not None and not pd.isna(rng) and rng >= 0:
                aib_mid = (aibh + aibl) / 2.0

        # Range + extensions (guard empty/degenerate range)
        aib_range = None
        aibh12x = aibh15x = aibh2x = None
        aibl12x = aibl15x = aibl2x = None
        if (aibh is not None) and (aibl is not None):
            aib_range = aibh - aibl
            if aib_range is not None and not pd.isna(aib_range) and aib_range >= 0:
                aibh12x = aibh + 0.20 * aib_range
                aibh15x = aibh + 0.50 * aib_range
                aibh2x  = aibh + 1.00 * aib_range
                aibl12x = aibl - 0.20 * aib_range
                aibl15x = aibl - 0.50 * aib_range
                aibl2x  = aibl - 1.00 * aib_range
            else:
                aib_range = None  # normalize NaN/negative to None

        # --- TOUCH LOGIC everywhere ---
        # Premarket hits (true touch during pre window)
        aibh_broke_premarket = _touched(pre_df, aibh)
        aibl_broke_premarket = _touched(pre_df, aibl)
        aib_mid_hit_premarket = _touched(pre_df, aib_mid)

        # Adjusted RTH hits (true touch during adj RTH window)
        aibh_broke_adj_rth = _touched(adjrth_df, aibh)
        aibl_broke_adj_rth = _touched(adjrth_df, aibl)
        aib_mid_hit_rth    = _touched(adjrth_df, aib_mid)

        # RTH extension hits (touch of those levels during adj RTH window)
        aibh12x_hit = _touched(adjrth_df, aibh12x)
        aibh15x_hit = _touched(adjrth_df, aibh15x)
        aibh2x_hit  = _touched(adjrth_df, aibh2x)

        aibl12x_hit = _touched(adjrth_df, aibl12x)
        aibl15x_hit = _touched(adjrth_df, aibl15x)
        aibl2x_hit  = _touched(adjrth_df, aibl2x)

        out_rows.append({
            "trade_date": gdate,
            "day": pd.to_datetime(gdate).strftime("%a"),

            "aibh": aibh, "aibl": aibl,
            "eibh": eibh, "eibl": eibl,
            "premarket_hi": pre_hi, "premarket_lo": pre_lo,
            "adj_ibh": adj_ibh, "adj_ibl": adj_ibl,
            "am_hi": am_hi, "am_lo": am_lo,
            "pm_hi": pm_hi, "pm_lo": pm_lo,
            "globex_hi": globex_hi, "globex_lo": globex_lo,
            "adj_rth_hi": adj_rth_hi, "adj_rth_lo": adj_rth_lo,

            # aIB mid value
            "aib_mid": aib_mid,

            # Premarket touches
            "aibh_broke_premarket": _safe_bool(aibh_broke_premarket),
            "aibl_broke_premarket": _safe_bool(aibl_broke_premarket),
            "aib_mid_hit_premarket": _safe_bool(aib_mid_hit_premarket),

            # Adj RTH touches
            "aibh_broke_adj_rth":   _safe_bool(aibh_broke_adj_rth),
            "aibl_broke_adj_rth":   _safe_bool(aibl_broke_adj_rth),
            "aib_mid_hit_rth":      _safe_bool(aib_mid_hit_rth),

            # Range + levels
            "aib_range": aib_range,
            "aibh12x": aibh12x, "aibh15x": aibh15x, "aibh2x": aibh2x,
            "aibl12x": aibl12x, "aibl15x": aibl15x, "aibl2x": aibl2x,

            # Extension hits (touch)
            "aibh12x_hit_rth": _safe_bool(aibh12x_hit),
            "aibh15x_hit_rth": _safe_bool(aibh15x_hit),
            "aibh2x_hit_rth":  _safe_bool(aibh2x_hit),
            "aibl12x_hit_rth": _safe_bool(aibl12x_hit),
            "aibl15x_hit_rth": _safe_bool(aibl15x_hit),
            "aibl2x_hit_rth":  _safe_bool(aibl2x_hit),
        })

    return pd.DataFrame(out_rows)

# ---------- Upsert ----------
def upsert_levels(supabase: Client, df_results: pd.DataFrame, batch_size: int = 1000) -> None:
    if df_results.empty:
        print("No results to upsert.")
        return

    d = df_results.copy()

    # Format dates as ISO strings for the conflict key
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # JSON-safe sanitization: remove infs, convert NaN -> None
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.astype(object)
    d = d.where(pd.notna(d), None)

    total = len(d)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = d.iloc[start:end].to_dict(orient="records")
        (
            supabase.table(DST_TABLE)
            .upsert(batch, on_conflict=["trade_date"])
            .execute()
        )
        print(f"Upserted {start+1:,}–{end:,} / {total:,}")

# ---------- Main ----------
def main() -> int:
    sb = get_client()
    df30 = fetch_all_rows(sb, SRC_TABLE, order_col="time", page_size=1000)
    if df30.empty:
        raise RuntimeError(f"{SRC_TABLE} returned no rows.")
    print(f"Loaded {len(df30):,} rows from {SRC_TABLE}; range {df30['time'].min()} → {df30['time'].max()}")

    results = compute_gc_levels(df30)
    print(f"Computed {len(results):,} gc_levels rows.")
    upsert_levels(sb, results)
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

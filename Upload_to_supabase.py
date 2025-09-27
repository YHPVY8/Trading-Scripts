#!/usr/bin/env python3
import pandas as pd
import numpy as np
from supabase import create_client
from pathlib import Path
import os

# ==== 1) Per-table config: csv path, time type, and columns that EXIST in that table ====
TABLES = {
    # DAILY / WEEKLY (DATE): format 'YYYY-MM-DD'
    "daily_es": {
        "csv": "/Users/calesmith/Documents/Trading View Exports/Globex Daily/ES_Daily.csv",
        "time_kind": "date",
        "columns": [
            "time","open","high","low","close",
            "200MA","50MA","20MA","10MA","5MA",
            "Volume","Volume MA","ATR"
        ],
    },
    "es_weekly": {
        "csv": "/Users/calesmith/Documents/Trading View Exports/Globex Weekly/ES_Weekly.csv",
        "time_kind": "date",
        "columns": [
            "time","open","high","low","close",
            "200MA","50MA","20MA","10MA","5MA",
            "Volume","Volume MA","ATR"
        ],
    },

    # INTRADAY (TIMESTAMPTZ)
    "es_30m": {
        "csv": "/Users/calesmith/Documents/Trading View Exports/Globex 30m/ES_30m.csv",
        "time_kind": "timestamptz",
        "columns": [
            "time","open","high","low","close",
            "200MA","50MA","20MA","10MA","5MA",
            "Volume","Volume MA","ATR"
        ],
    },
    "es_2hr": {
        "csv": "/Users/calesmith/Documents/Trading View Exports/Globex 2 HR/ES_2 HR.csv",
        "time_kind": "timestamptz",
        "columns": [
            "time","open","high","low","close",
            "200MA","50MA","20MA","10MA","5MA",
            "Volume","Volume MA","ATR"
        ],
    },
    "es_4hr": {
        "csv": "/Users/calesmith/Documents/Trading View Exports/Globex 4 HR/ES_4 HR.csv",
        "time_kind": "timestamptz",
        "columns": [
            "time","open","high","low","close",
            "200MA","50MA","20MA","10MA","5MA",
            "Volume","Volume MA","ATR"
        ],
    },
}

# ==== 2) Supabase connection ====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def _normalize_intraday_time(series: pd.Series) -> pd.Series:
    # Keep your original timestamp, just strip trailing timezone offset like -04:00
    s = series.astype(str).str.replace(r"([+-]\d{2}):(\d{2})$", "", regex=True)
    # Parse & format back to ISO without offset to ensure Postgres accepts it
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%dT%H:%M:%S")

def _clean_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    # Replace odd placeholders and unicode minus BEFORE numeric coercion
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(str)

        # Normalize unicode minus (−) to ASCII '-'
        s = s.str.replace("\u2212", "-", regex=False)

        # Treat common non-numeric placeholders as NaN (exact matches)
        s = s.replace({
            "": np.nan,
            " ": np.nan,
            "NaN": np.nan,
            "nan": np.nan,
            "NULL": np.nan,
            "null": np.nan,
            "--": np.nan,
            "—": np.nan,   # em dash
            "–": np.nan,   # en dash
            ".": np.nan,
            "∞": np.nan,
        })

        # Now coerce
        df[col] = pd.to_numeric(s, errors="coerce")

    # Replace infinities (just in case) with NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # IMPORTANT: cast numeric columns to object, then replace NaN with None
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)
            # set None (which becomes SQL NULL) where value is NaN
            df.loc[pd.isna(df[col]), col] = None

    return df

def load_and_clean(csv_path: str, time_kind: str, keep_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        raise ValueError(f"'time' column not found in {csv_path}")

    # Time normalization
    if time_kind == "timestamptz":
        df["time"] = _normalize_intraday_time(df["time"])
    else:  # date
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.date.astype(str)

    # Drop invalid times
    before = len(df)
    df = df[df["time"].notna()]
    after = len(df)
    if after < before:
        print(f"  ⚠️ dropped {before - after} invalid/missing rows in {Path(csv_path).name}")

    # Align to the table schema columns
    present = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in keep_cols]

    if missing:
        print(f"  ℹ️ {Path(csv_path).name}: missing columns (will be omitted): {missing}")
    if extra:
        print(f"  ℹ️ {Path(csv_path).name}: extra columns ignored (not in table): {extra}")

    if not present:
        raise ValueError(f"No overlapping columns between CSV and table schema for {csv_path}")

    df = df[present].copy()

    # Numeric sanitization (everything except 'time')
    numeric_cols = [c for c in df.columns if c != "time"]
    df = _clean_numeric_columns(df, numeric_cols)

    # Final safety: ensure absolutely no NaN/inf remain anywhere
    # (They shouldn't, because we cast to object & set None)
    # But we’ll do a paranoid check:
    if any(isinstance(v, float) and (np.isnan(v) or v in (np.inf, -np.inf))
           for v in df[numeric_cols].to_numpy().ravel() if v is not None):
        # As a last resort, replace any remaining with None
        df[numeric_cols] = df[numeric_cols].applymap(
            lambda x: None if (isinstance(x, float) and (pd.isna(x) or x in (np.inf, -np.inf))) else x
        )

    return df

def _safe_insert(table_name: str, records: list[dict], skipped: list):
    """Insert records; on 'JSON could not be generated' errors, bisect to isolate bad rows."""
    if not records:
        return
    try:
        sb.table(table_name).insert(records).execute()
    except Exception as e:
        msg = str(e)
        # Only split if it's the JSON/float issue; otherwise re-raise
        if "Out of range float values are not JSON compliant" in msg or "JSON could not be generated" in msg:
            if len(records) == 1:
                # Skip the single bad row but report it
                skipped.append(records[0])
                print(f"  ⚠️ Skipping 1 bad row for {table_name}: {records[0]}")
                return
            mid = len(records) // 2
            _safe_insert(table_name, records[:mid], skipped)
            _safe_insert(table_name, records[mid:], skipped)
        else:
            raise  # different error—surface it

def replace_table(table_name: str, df: pd.DataFrame):
    # Clear table safely
    sb.table(table_name).delete().gte("time", "0001-01-01").execute()

    records = df.to_dict(orient="records")
    if not records:
        print(f"  (empty dataset) nothing to insert for {table_name}")
        return

    # Use safe inserter to avoid one bad row killing the whole upload
    skipped = []
    BATCH = 5000
    for i in range(0, len(records), BATCH):
        chunk = records[i:i+BATCH]
        _safe_insert(table_name, chunk, skipped)

    if skipped:
        print(f"  ⚠️ {len(skipped)} row(s) were skipped in {table_name} due to invalid numeric values.")

def main():
    for table_name, meta in TABLES.items():
        csv_path = meta["csv"]
        time_kind = meta["time_kind"]
        keep_cols = meta["columns"]

        path = Path(csv_path)
        if not path.exists():
            print(f"⚠️ File not found: {csv_path}")
            continue

        print(f"\n📤 Uploading {path.name} → {table_name} ({time_kind})")
        try:
            df = load_and_clean(csv_path, time_kind, keep_cols)
            replace_table(table_name, df)
            print(f"✅ Uploaded ~{len(df)} rows to {table_name} (some rows may be skipped if invalid)")
            print("  📊 tail(3):")
            print(df.tail(3)[[c for c in df.columns if c in ('time','open','high','low','close')]])
        except Exception as e:
            print(f"❌ Error for {table_name}: {e}")

if __name__ == "__main__":
    main()

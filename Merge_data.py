#!/usr/bin/env python3
"""
tv_update_and_append.py

- Scans ~/Downloads for TradingView CSV exports with names like:
  "CME_MINI_ES1!, 30 - 2025-09-12T144701.339.csv"
  "CME_MINI_ES1!, 3 (28).csv", "CME_MINI_ES1!, 1D (96).csv", etc.

- For each timeframe found, updates the matching master CSV:
  * Drops the last row of the master (incomplete-bar).
  * Finds the same timestamp in the new export and replaces the last row
    with the updated bar, then appends any rows after that.
  * Keeps master headers exactly as-is.
  * Makes a timestamped backup of the master before writing (max 2 kept).
  * Deletes the processed export from Downloads after success.
"""

import os
import glob
import re
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

# --- CONFIGURE YOUR MASTER PATHS HERE ---
downloads = Path.home() / "Downloads"

MASTERS = {
    "30": Path("/Users/calesmith/Documents/Trading View Exports/Globex 30m/ES_30m.csv"),
    "3":  Path("/Users/calesmith/Documents/Trading View Exports/Globex 3 min/ES - 3 min.csv"),
    "60": Path("/Users/calesmith/Documents/Trading View Exports/Globex 30m/ES_30m.csv"),  # adjust if needed
    "120":Path("/Users/calesmith/Documents/Trading View Exports/Globex 2 HR/ES_2 HR.csv"),
    "240":Path("/Users/calesmith/Documents/Trading View Exports/Globex 4 HR/ES_4 HR.csv"),
    "1D": Path("/Users/calesmith/Documents/Trading View Exports/Globex Daily/ES_Daily.csv"),
    "1W": Path("/Users/calesmith/Documents/Trading View Exports/Globex Weekly/ES_Weekly.csv"),
}

# --- helper functions ---
def find_latest_exports():
    """Return dict: token -> Path(latest matching file in Downloads)."""
    pattern = str(downloads / "CME_MINI_ES1!*.csv")
    files = glob.glob(pattern)
    if not files:
        return {}

    files.sort(key=os.path.getmtime, reverse=True)  # newest first
    found = {}
    for f in files:
        stem = Path(f).stem
        m = re.search(r',\s*([0-9]+(?:D|W)?)', stem, re.IGNORECASE)
        if not m:
            continue
        token = m.group(1).upper()
        token = token.replace(' ', '')
        if token in MASTERS and token not in found:
            found[token] = Path(f)
        if len(found) == len(MASTERS):
            break
    return found

def backup_file(path: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("/Users/calesmith/Documents/Trading View Exports/Back Up Files")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (path.stem + f".backup_{ts}" + path.suffix)
    shutil.copy(path, backup)

    # keep only 2 most recent
    backups = sorted(
        backup_dir.glob(path.stem + ".backup_*" + path.suffix),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    for old_backup in backups[2:]:
        old_backup.unlink()

    return backup

def safe_read_csv(path: Path):
    return pd.read_csv(path, dtype=str)

def update_master(master_path: Path, new_export_path: Path):
    """Update master with new export."""
    if not master_path.exists():
        raise FileNotFoundError(f"Master file not found: {master_path}")

    df_master = safe_read_csv(master_path)
    df_new = safe_read_csv(new_export_path)

    if df_new.shape[1] != df_master.shape[1]:
        raise ValueError(
            f"Column mismatch: master has {df_master.shape[1]} cols, new export has {df_new.shape[1]} cols"
        )

    df_new.columns = df_master.columns.tolist()

    try:
        df_master["__time_dt"] = pd.to_datetime(df_master["time"])
        df_new["__time_dt"] = pd.to_datetime(df_new["time"])
    except Exception as e:
        raise ValueError(f"Failed to parse 'time' column as datetimes. Error: {e}")

    if df_master.shape[0] == 0:
        df_new.drop(columns="__time_dt", inplace=True)
        df_new.to_csv(master_path, index=False)
        return {"added": len(df_new), "replaced": 0}

    last_master_time = df_master["__time_dt"].iloc[-1]
    match_idxs = df_new.index[df_new["__time_dt"] == last_master_time].tolist()

    replaced = 0
    appended = 0

    if match_idxs:
        idx = match_idxs[0]
        df_master_drop_last = df_master.iloc[:-1].drop(columns="__time_dt")
        df_to_append = df_new.iloc[idx:].drop(columns="__time_dt")
        df_updated = pd.concat([df_master_drop_last, df_to_append], ignore_index=True)
        replaced = 1
        appended = len(df_to_append) - 1
    else:
        le_idxs = df_new.index[df_new["__time_dt"] <= last_master_time].tolist()
        if le_idxs:
            idx = le_idxs[-1]
            df_master_drop_last = df_master.iloc[:-1].drop(columns="__time_dt")
            df_to_append = df_new.iloc[idx+1:].drop(columns="__time_dt")
            df_updated = pd.concat([df_master_drop_last, df_to_append], ignore_index=True)
            replaced = 1
            appended = len(df_to_append)
        else:
            df_combined = pd.concat(
                [df_master.drop(columns="__time_dt"), df_new.drop(columns="__time_dt")],
                ignore_index=True
            )
            df_combined = (
                df_combined.drop_duplicates(subset=["time"], keep="last")
                .sort_values("time")
                .reset_index(drop=True)
            )
            appended = max(0, len(df_combined) - len(df_master))
            replaced = 0
            df_updated = df_combined

    # 🔹 Normalize Daily and Weekly file dates to mm/dd/yy
    if master_path.name in ["ES_Daily.csv", "ES_Weekly.csv"]:
        try:
            df_updated["time"] = (
                pd.to_datetime(df_updated["time"], errors="coerce")
                .dt.strftime("%m/%d/%y")
            )
        except Exception as e:
            print(f"⚠️ Could not normalize {master_path.name} dates: {e}")

    backup = backup_file(master_path)
    df_updated.to_csv(master_path, index=False)

    return {
        "backup": str(backup),
        "replaced": replaced,
        "appended": appended,
        "final_rows": len(df_updated),
    }

def main():
    print("Scanning Downloads for TradingView exports...")
    found = find_latest_exports()
    if not found:
        print("⚠️ No TradingView exports detected in Downloads matching pattern.")
        return

    summary = []
    for token, new_path in found.items():
        master_path = MASTERS.get(token)
        if not master_path:
            print(f"⚠️ No master file configured for token '{token}', skipping {new_path.name}")
            continue
        try:
            result = update_master(master_path, new_path)
            try:
                new_path.unlink()
            except Exception:
                pass
            print(f"✅ {master_path.name} updated — replaced:{result.get('replaced',0)} appended:{result.get('appended',0)} (backup: {result.get('backup','-')})")
            summary.append((master_path.name, result))
        except Exception as e:
            print(f"❌ Error for {master_path.name}: {e}")

    if summary:
        print("\nSummary:")
        for name, res in summary:
            print(f" - {name}: replaced={res.get('replaced',0)} appended={res.get('appended',0)} final_rows={res.get('final_rows','?')}")
    else:
        print("No files updated.")

if __name__ == "__main__":
    main()

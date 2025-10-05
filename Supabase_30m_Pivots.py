#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from supabase import create_client, Client

# --- Supabase connection ---
SUPABASE_URL = "https://kjgaieellljetntsdytt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqZ2FpZWVsbGxqZXRudHNkeXR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODM2MjM5NCwiZXhwIjoyMDczOTM4Mzk0fQ.Miwj8itGx3bfUBSHOIjdZLtSYIETuoYakzYJrCt83kQ"

print("URL:", SUPABASE_URL)
print("KEY exists?", bool(SUPABASE_KEY))
print("KEY length:", len(SUPABASE_KEY) if SUPABASE_KEY else "None")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- helpers ----------
def fetch_all_rows(table: str, order_col: str = "time", page_size: int = 1000) -> pd.DataFrame:
    """Pull ALL rows from a table by paginating with .range()."""
    all_chunks = []
    start = 0
    while True:
        end = start + page_size - 1
        resp = (
            supabase.table(table)
            .select("*")
            .order(order_col, desc=False)
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
        df[order_col] = pd.to_datetime(df[order_col])
        df = df.sort_values(order_col).reset_index(drop=True)
    return df

# ---------- Step 1: Load ALL source data (30m prices) ----------
df = fetch_all_rows("es_30m", order_col="time", page_size=1000)
if df.empty:
    raise RuntimeError("es_30m returned no rows; cannot compute pivots.")

print(f"Loaded {len(df):,} rows from es_30m.")
print(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")

# ---------- Step 2: Add Globex Date & Day ----------
# Globex date: times >=18:00 belong to next calendar day; else same day
df["time"] = pd.to_datetime(df["time"])
df["globex_date"] = df["time"].apply(
    lambda t: (t + pd.Timedelta(days=1)).date() if t.hour >= 18 else t.date()
)
df["day"] = pd.to_datetime(df["globex_date"]).dt.strftime("%A")

# ---------- Step 3: Compute pivots ----------
results = []
for i in range(1, len(df)):
    pHi = df.loc[i - 1, "high"]
    pLo = df.loc[i - 1, "low"]
    pCL = df.loc[i - 1, "close"]

    pivot = (pHi + pLo + pCL + pCL) / 4
    r1 = (pivot * 2) - pLo
    s1 = (pivot * 2) - pHi
    r2 = pivot + (pHi - pLo)
    s2 = pivot - (pHi - pLo)
    r05 = (pivot + r1) / 2
    s05 = (pivot + s1) / 2
    r025 = (pivot + r05) / 2
    s025 = (pivot + s05) / 2
    r15 = (r1 + r2) / 2
    s15 = (s1 + s2) / 2
    r3 = pHi + (2 * (pivot - pLo))
    s3 = pLo - (2 * (pHi - pivot))

    current_high = df.loc[i, "high"]
    current_low = df.loc[i, "low"]

    hit_conditions = {
        "hit_pivot": current_high >= pivot >= current_low,
        "hit_r025": current_high >= r025 >= current_low,
        "hit_s025": current_high >= s025 >= current_low,
        "hit_r05":  current_high >= r05  >= current_low,
        "hit_s05":  current_high >= s05  >= current_low,
        "hit_r1":   current_high >= r1   >= current_low,
        "hit_s1":   current_high >= s1   >= current_low,
        "hit_r15":  current_high >= r15  >= current_low,
        "hit_s15":  current_high >= s15  >= current_low,
        "hit_r2":   current_high >= r2   >= current_low,
        "hit_s2":   current_high >= s2   >= current_low,
        "hit_r3":   current_high >= r3   >= current_low,
        "hit_s3":   current_high >= s3   >= current_low,
    }

    results.append({
        "time": df.loc[i, "time"],  # will convert below
        "globex_date": df.loc[i, "globex_date"],
        "day": df.loc[i, "day"],
        "phi":   round(pHi,    2),
        "plo":   round(pLo,    2),
        "pcl":   round(pCL,    2),
        "pivot": round(pivot,  2),
        "r025":  round(r025,   2),
        "s025":  round(s025,   2),
        "r05":   round(r05,    2),
        "s05":   round(s05,    2),
        "r1":    round(r1,     2),
        "s1":    round(s1,     2),
        "r15":   round(r15,    2),
        "s15":   round(s15,    2),
        "r2":    round(r2,     2),
        "s2":    round(s2,     2),
        "r3":    round(r3,     2),
        "s3":    round(s3,     2),
        **hit_conditions
    })

df_results = pd.DataFrame(results)
print(f"Computed {len(df_results):,} pivot rows.")

# ---------- Step 4: Convert datetimes to strings & UPSERT ----------
if not df_results.empty:
    if "time" in df_results.columns:
        df_results["time"] = pd.to_datetime(df_results["time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    if "globex_date" in df_results.columns:
        df_results["globex_date"] = pd.to_datetime(df_results["globex_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    BATCH = 1000
    total = len(df_results)
    for start in range(0, total, BATCH):
        end = start + BATCH
        batch = df_results.iloc[start:end]
        supabase.table("es_30m_pivot_levels") \
            .upsert(batch.to_dict(orient="records"), on_conflict=["time"]) \
            .execute()
        print(f"  - Upserted rows {start+1:,}–{min(end, total):,}")

    print(f"Upserted {total:,} rows into es_30m_pivot_levels.")
else:
    print("No data to insert.")

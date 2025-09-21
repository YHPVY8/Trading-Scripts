#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from supabase import create_client, Client
import os

# --- Supabase connection ---
SUPABASE_URL = "https://kjgaieellljetntsdytt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqZ2FpZWVsbGxqZXRudHNkeXR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODM2MjM5NCwiZXhwIjoyMDczOTM4Mzk0fQ.Miwj8itGx3bfUBSHOIjdZLtSYIETuoYakzYJrCt83kQ"

print("URL:", SUPABASE_URL)
print("KEY exists?", bool(SUPABASE_KEY))
print("KEY length:", len(SUPABASE_KEY) if SUPABASE_KEY else "None")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Step 1: Load source data (daily prices) from Supabase ---
response = supabase.table("daily_es").select("*").execute()
df = pd.DataFrame(response.data)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# --- Step 2: Add extra columns for calculations ---
df["Day"] = df["time"].dt.strftime('%A')
df["date"] = df["time"].dt.strftime('%Y-%m-%d')

results = []  # collect rows to insert into pivot_levels table
num_days = min(1000, len(df))

for i in range(len(df) - num_days, len(df)):
    prior_day_index = i - 1
    if prior_day_index < 0:
        continue

    pHi = df.loc[prior_day_index, "high"]
    pLo = df.loc[prior_day_index, "low"]
    pCL = df.loc[prior_day_index, "close"]

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

    # Get the current day high/low to check for hits
    current_high = df.loc[i, "high"]
    current_low = df.loc[i, "low"]

    hit_conditions = {
        "hit_pivot": current_high >= pivot >= current_low,
        "hit_r025": current_high >= r025 >= current_low,
        "hit_s025": current_high >= s025 >= current_low,
        "hit_r05": current_high >= r05 >= current_low,
        "hit_s05": current_high >= s05 >= current_low,
        "hit_r1": current_high >= r1 >= current_low,
        "hit_s1": current_high >= s1 >= current_low,
        "hit_r15": current_high >= r15 >= current_low,
        "hit_s15": current_high >= s15 >= current_low,
        "hit_r2": current_high >= r2 >= current_low,
        "hit_s2": current_high >= s2 >= current_low,
        "hit_r3": current_high >= r3 >= current_low,
        "hit_s3": current_high >= s3 >= current_low,
    }

    results.append({
        "date": df.loc[i, "date"],
        "day": df.loc[i, "Day"],
        "phi": pHi,    # <-- lowercase key
        "plo": pLo,    # <-- lowercase key
        "pcl": pCL,    # <-- lowercase key
        "pivot": pivot,
        "r025": r025,
        "s025": s025,
        "r05": r05,
        "s05": s05,
        "r1": r1,
        "s1": s1,
        "r15": r15,
        "s15": s15,
        "r2": r2,
        "s2": s2,
        "r3": r3,
        "s3": s3,
        **hit_conditions
    })


# --- Step 3: Convert to DataFrame and push to Supabase ---
df_results = pd.DataFrame(results)


# Insert new rows
if not df_results.empty:
    supabase.table("es_daily_pivot_levels").insert(df_results.to_dict(orient="records")).execute()
    print(f"Inserted {len(df_results)} rows into es_daily_pivot_levels.")
else:
    print("No data to insert.")

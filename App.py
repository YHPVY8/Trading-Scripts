#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client

# ---- CONFIG ----
st.set_page_config(page_title="📊 Trading Dashboard", layout="wide")

SUPABASE_URL = "https://kjgaieellljetntsdytt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtqZ2FpZWVsbGxqZXRudHNkeXR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODM2MjM5NCwiZXhwIjoyMDczOTM4Mzk0fQ.Miwj8itGx3bfUBSHOIjdZLtSYIETuoYakzYJrCt83kQ"
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLES = {
    "Daily ES": ("daily_es", "time"),
    "Weekly ES": ("es_weekly", "time"),
    "30m ES": ("es_30m", "time"),
    "2h ES": ("es_2hr", "time"),
    "4h ES": ("es_4hr", "time"),
    "Daily Pivots": ("es_daily_pivot_levels", "date"),
    "Weekly Pivots": ("es_weekly_pivot_levels", "date"),
    "2h Pivots": ("es_2hr_pivot_levels", "time"),
    "4h Pivots": ("es_4hr_pivot_levels", "time"),
    "30m Pivots": ("es_30m_pivot_levels", "time"),
    "Range Extensions": ("es_range_extensions", "date"),
}

st.title("📊 Trading Dashboard")

# ---- Sidebar ----
choice = st.sidebar.selectbox("Select data set", list(TABLES.keys()))
limit = st.sidebar.number_input("Number of rows to load", value=1000, min_value=100, step=100)

table_name, date_col = TABLES[choice]

# ---- Load ----
response = (
    sb.table(table_name)
    .select("*")
    .order(date_col, desc=True)
    .limit(limit)
    .execute()
)
df = pd.DataFrame(response.data)

if df.empty:
    st.error("No data returned.")
    st.stop()

# --- Sort so latest is at bottom ---
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

# --- CLEANUP PER DATASET ---
# 1️⃣ Remove 'id' from Daily ES
if choice == "Daily ES" and "id" in df.columns:
    df = df.drop(columns=["id"])

# 2️⃣ Standardize date/time formatting
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

if "time" in df.columns:
    if choice in ["Daily ES", "Weekly ES"]:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M")

# 3️⃣ Restrict columns for pivots
if choice == "Daily Pivots":
    keep_cols = ["date", "day", "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                 "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols if c in df.columns]]

elif choice == "Weekly Pivots":
    keep_cols = ["date", "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                 "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols if c in df.columns]]

elif choice in ["2h Pivots", "4h Pivots", "30m Pivots"]:
    keep_cols = ["time","globex_date","day",
                 "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                 "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols if c in df.columns]]

# 4️⃣ Restrict columns for ES datasets
elif choice in ["Daily ES", "Weekly ES", "30m ES", "2h ES", "4h ES"]:
    keep = ["time","open","high","low","close","200MA","50MA","20MA","10MA","5MA","Volume","ATR"]
    df = df[[c for c in keep if c in df.columns]]

# 5️⃣ Restrict columns for Range Extensions
elif choice == "Range Extensions":
    keep = [
        "date","day","range","14_Day_Avg_Range","range_gt_avg","range_gt_80_avg",
        "op_lo","op_lo_14_avg","op_lo_gt_avg","range_gt_80_op_lo",
        "hi_op","hi_op_14_avg","hi_op_gt_avg","range_gt_80_hi_op",
        "hit_both_80","hit_both_14_avg",
        "range_gt_120_avg","range_gt_120_op_lo","range_gt_120_hi_op"
    ]
    df = df[[c for c in keep if c in df.columns]]

# ---- Formatting numeric columns ----
numeric_cols = [
    "range","14_Day_Avg_Range","op_lo","op_lo_14_avg","hi_op","hi_op_14_avg"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)

# ---- Highlight + Bold Headers ----
def color_hits(val):
    if val is True or str(val).lower() == "true":
        return "background-color: #98FB98"
    return ""

def bold_headers(styler):
    return styler.set_table_styles(
        [{"selector": "thead th", "props": [("font-weight", "bold")]}]
    )

# ---- Render table ----
if choice in [
    "Daily Pivots","Weekly Pivots","2h Pivots","4h Pivots",
    "30m Pivots","Range Extensions"
]:
    hit_cols = [c for c in df.columns if any(s in c.lower() for s in ["hit", "gt", ">"])]
    styled = df.style.map(color_hits, subset=hit_cols)
    styled = bold_headers(styled)
    st.dataframe(styled, use_container_width=True, height=600)
else:
    styled = bold_headers(df.style)
    st.dataframe(styled, use_container_width=True, height=600)

# ---- Download button ----
st.download_button(
    "💾 Download filtered CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"{choice.lower().replace(' ','_')}_filtered.csv",
    mime="text/csv"
)

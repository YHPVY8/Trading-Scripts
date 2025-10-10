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

# Remove id if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Date/time cleanup
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

if "time" in df.columns:
    if choice in ["Daily ES", "Weekly ES"]:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M")

# ---- Range Extensions View ----
if choice == "Range Extensions":
    keep_cols = [
        "date", "day", "range", "14_Day_Avg_Range", "range_gt_avg", "range_gt_80_avg",
        "op_lo", "op_lo_14_avg", "op_lo_gt_avg", "range_gt_80_op_lo",
        "hi_op", "hi_op_14_avg", "hi_op_gt_avg", "range_gt_80_hi_op",
        "hit_both_80", "hit_both_14_avg",
        "range_gt_120_avg", "range_gt_120_op_lo", "range_gt_120_hi_op"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

# ---- Pivots (Daily/Weekly/2h/4h/30m) ----
elif choice in ["Daily Pivots", "Weekly Pivots", "2h Pivots", "4h Pivots", "30m Pivots"]:
    keep_cols = [
        "date" if "date" in df.columns else "time",
        "day" if "day" in df.columns else None,
        "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
        "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

# ---- ES price datasets ----
elif choice in ["Daily ES", "Weekly ES", "30m ES", "2h ES", "4h ES"]:
    keep = ["time","open","high","low","close","200MA","50MA","20MA","10MA","5MA","Volume","ATR"]
    df = df[[c for c in keep if c in df.columns]]

# ---- Numeric Formatting ----
numeric_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64"]]
for col in numeric_cols:
    df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)

if "Volume" in df.columns:
    df["Volume"] = df["Volume"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else x)

# ---- Multi-filter Sidebar ----
filters = []
num_filters = st.sidebar.number_input("Number of filters", 0, 5, 0)
if st.sidebar.button("Clear Filters"):
    st.rerun()

for n in range(num_filters):
    col = st.sidebar.selectbox(f"Column #{n+1}", df.columns, key=f"fcol{n}")
    op = st.sidebar.selectbox(f"Op #{n+1}", ["equals","contains","greater than","less than"], key=f"fop{n}")
    val = st.sidebar.text_input(f"Value #{n+1}", key=f"fval{n}")
    if col and op and val:
        filters.append((col, op, val))

for col, op, val in filters:
    if op == "equals":
        df = df[df[col].astype(str) == val]
    elif op == "contains":
        df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
    elif op == "greater than":
        df = df[pd.to_numeric(df[col], errors='coerce') > float(val

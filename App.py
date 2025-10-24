#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client

# ---- CONFIG ----
st.set_page_config(page_title="📊 Trading Dashboard", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
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

st.title("Trading Dashboard")

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
if choice == "Daily ES" and "id" in df.columns:
    df = df.drop(columns=["id"])

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

if "time" in df.columns:
    if choice in ["Daily ES", "Weekly ES"]:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M")

# --- Restrict columns ---
if choice == "Daily Pivots":
    keep_cols = ["date","day","hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                 "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols if c in df.columns]]

elif choice == "Weekly Pivots":
    keep_cols = ["date","hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                 "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols if c in df.columns]]

elif choice in ["2h Pivots","4h Pivots","30m Pivots"]:
    keep_cols = ["time","globex_date","day","hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                 "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols if c in df.columns]]

elif choice in ["Daily ES","Weekly ES","30m ES","2h ES","4h ES"]:
    keep = ["time","open","high","low","close","200MA","50MA","20MA","10MA","5MA","Volume","ATR"]
    df = df[[c for c in keep if c in df.columns]]

elif choice == "Range Extensions":
    keep = [
        "date","day","range","14_Day_Avg_Range","range_gt_avg","range_gt_80_avg",
        "op_lo","op_lo_14_avg","op_lo_gt_avg","range_gt_80_op_lo",
        "hi_op","hi_op_14_avg","hi_op_gt_avg","range_gt_80_hi_op",
        "hit_both_80","hit_both_14_avg",
        "range_gt_120_avg","range_gt_120_op_lo","range_gt_120_hi_op"
    ]
    df = df[[c for c in keep if c in df.columns]]

# ---- Format all numeric columns ----
for col in df.columns:
    if df[col].dtype in ["float64", "float32", "int64", "int32"]:
        if "volume" in col.lower():
            df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else x)
        else:
            df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)

# ---- Multi-condition filter ----
filters = []
num_filters = st.sidebar.number_input("Number of filters", 0, 5, 0)

if st.sidebar.button("Clear Filters"):
    st.rerun()

for n in range(num_filters):
    col = st.sidebar.selectbox(f"Column #{n+1}", df.columns, key=f"fcol{n}")
    op = st.sidebar.selectbox(f"Op #{n+1}", ["equals","contains","greater than","less than"], key=f"fop{n}")
    val = st.sidebar.text_input(f"Value #{n+1}", key=f"fval{n}")
    if col and op and val:
        filters.append((col,op,val))

for col, op, val in filters:
    if op == "equals":
        df = df[df[col].astype(str) == val]
    elif op == "contains":
        df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
    elif op == "greater than":
        df = df[pd.to_numeric(df[col], errors='coerce') > float(val)]
    elif op == "less than":
        df = df[pd.to_numeric(df[col], errors='coerce') < float(val)]

# ---- Highlight + Bold Headers ----
def color_hits(val):
    if val is True or str(val).lower() == "true":
        return "background-color: #98FB98"
    return ""

def bold_headers(styler):
    return styler.set_table_styles([{"selector": "thead th", "props": [("font-weight", "bold")]}])

# ---- Display ----
if choice in [
    "Daily Pivots","Weekly Pivots","2h Pivots","4h Pivots",
    "30m Pivots","Range Extensions"
]:
    hit_cols = [c for c in df.columns if any(s in c.lower() for s in ["hit","gt",">"])]
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

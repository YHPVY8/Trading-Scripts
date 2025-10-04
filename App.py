#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client
import os

# ---- CONFIG ----
st.set_page_config(page_title="Trading Dashboard", layout="wide")

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
}

# Which columns to actually show for pivot tables
PIVOT_COLS = [
    "date","day",
    "phi","plo","pcl","pivot",
    "r025","s025","r05","s05",
    "r1","s1","r15","s15","r2","s2","r3","s3",
    "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
    "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3",
]

# Numeric columns for pivots (so we coerce/round the right ones)
PIVOT_NUM_COLS = [
    "phi","plo","pcl","pivot",
    "r025","s025","r05","s05",
    "r1","s1","r15","s15","r2","s2","r3","s3",
]

# Numeric columns for price tables
PRICE_NUM_COLS = [
    "open","high","low","close",
    "200MA","50MA","20MA","10MA","5MA",
    "Volume","Volume MA","ATR",
]

def coerce_and_round(df: pd.DataFrame, cols: list[str], decimals: int = 2) -> pd.DataFrame:
    """
    Force numeric parsing for the specific columns:
     - Strip thousands separators (',') if present
     - Convert to numeric
     - Round to `decimals`
    Leaves missing or non-numeric as NaN.
    """
    for c in cols:
        if c in df.columns:
            # Work via string → remove commas → to_numeric
            s = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce").round(decimals)
    return df

def style_hits(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    hit_cols = [c for c in df.columns if c.startswith("hit")]
    def color_hits(val):
        if val is True or str(val).lower() == "true":
            return "background-color: #98FB98"
        return ""
    sty = df.style
    if hit_cols:
        sty = sty.map(color_hits, subset=hit_cols)
    # Format numeric columns with 2 decimals for display (keeps booleans clean)
    num_cols = df.select_dtypes(include=["float", "float64", "int", "int64"]).columns
    fmt = {c: "{:.2f}" for c in num_cols}
    sty = sty.format(fmt)
    return sty

st.title("📊 Trading Dashboard")

# ---- Sidebar ----
choice = st.sidebar.selectbox("Select data set", list(TABLES.keys()))
limit = st.sidebar.number_input("Number of rows to load", value=1000, min_value=100, step=100)

table_name, date_col = TABLES[choice]

# ---- Load ----
resp = (
    sb.table(table_name)
    .select("*")
    .order(date_col, desc=True)  # newest first from server
    .limit(limit)
    .execute()
)
df = pd.DataFrame(resp.data)

if df.empty:
    st.error("No data returned.")
    st.stop()

# Sort ascending so newest is at the bottom in the view
df = df.sort_values(date_col)

# ---- Column selection & numeric coercion
if choice in ("Daily Pivots", "Weekly Pivots"):
    # restrict to pivot columns that exist
    keep = [c for c in PIVOT_COLS if c in df.columns]
    df = df[keep]
    df = coerce_and_round(df, PIVOT_NUM_COLS, decimals=2)
else:
    # price tables: coerce the known numeric columns that exist
    price_cols_present = [c for c in PRICE_NUM_COLS if c in df.columns]
    df = coerce_and_round(df, price_cols_present, decimals=2)

# ---- Multi-condition filter
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
        df = df[pd.to_numeric(df[col], errors='coerce') > float(val)]
    elif op == "less than":
        df = df[pd.to_numeric(df[col], errors='coerce') < float(val)]

# ---- Display
if any(c.startswith("hit") for c in df.columns):
    st.dataframe(style_hits(df), width='stretch', height=600)
else:
    # Format numeric display to 2dp on non-pivot tables too
    num_cols = df.select_dtypes(include=["float", "float64", "int", "int64"]).columns
    sty = df.style.format({c: "{:.2f}" for c in num_cols})
    st.dataframe(sty, width='stretch', height=600)

# ---- Download
st.download_button(
    "💾 Download filtered CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"{choice.lower().replace(' ','_')}_filtered.csv",
    mime="text/csv"
)

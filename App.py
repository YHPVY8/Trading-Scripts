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

# Columns we want to show for pivot tables
PIVOT_COLS = [
    "date","day",
    "phi","plo","pcl","pivot",
    "r025","s025","r05","s05",
    "r1","s1","r15","s15","r2","s2","r3","s3",
    "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
    "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3",
]

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

# Sort ascending so newest ends at bottom in the view
df = df.sort_values(date_col)

# If we're on a pivots table, restrict to known pivot cols (that exist)
if choice in ("Daily Pivots", "Weekly Pivots"):
    keep = [c for c in PIVOT_COLS if c in df.columns]
    df = df[keep]

# --- Coerce numeric + round 2dp (won’t touch bool hit_* columns)
numeric_cols = df.select_dtypes(include=["float", "float64", "int", "int64"]).columns.tolist()
# If some numeric fields arrived as strings, coerce them before rounding
for c in df.columns:
    if c not in ("date", "day") and not c.startswith("hit"):
        df[c] = pd.to_numeric(df[c], errors="ignore")

df[numeric_cols] = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
df[numeric_cols] = df[numeric_cols].round(2)

# --- Multi-condition filter ---
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

# ---- Highlight hits (any dataset with hit_ columns) ----
def color_hits(val):
    if val is True or str(val).lower() == "true":
        return "background-color: #98FB98"
    return ""

if any(c.startswith("hit") for c in df.columns):
    styled = df.style.map(color_hits, subset=[c for c in df.columns if c.startswith("hit")])
    st.dataframe(styled, width='stretch', height=600)
else:
    st.dataframe(df, width='stretch', height=600)

# ---- Download button ----
st.download_button(
    "💾 Download filtered CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"{choice.lower().replace(' ','_')}_filtered.csv",
    mime="text/csv"
)

# ---- Debug / QA: compare a single row with a fresh fetch from Supabase ----
with st.expander("🔍 Debug: Compare with Supabase row-by-row"):
    if "date" in df.columns:
        dates = df["date"].astype(str).unique().tolist()
        pick = st.selectbox("Pick a date to compare", dates[::-1])  # show recent first
        if pick:
            # Row displayed in the grid (after rounding)
            shown = df[df["date"].astype(str) == pick]
            st.write("Row in the app (post-rounding/filters):")
            st.dataframe(shown, width='stretch')

            # Fresh fetch from Supabase for the exact same date
            fresh = sb.table(table_name).select("*").eq("date", pick).execute()
            fresh_df = pd.DataFrame(fresh.data)
            st.write("Raw row fetched directly from Supabase (no rounding):")
            st.dataframe(fresh_df, width='stretch')
    else:
        st.info("No 'date' column available to compare.")

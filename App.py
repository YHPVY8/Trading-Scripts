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
    .order(date_col, desc=True)  # get most recent rows first
    .limit(limit)
    .execute()
)
df = pd.DataFrame(response.data)

if df.empty:
    st.error("No data returned.")
    st.stop()

# Sort ascending so newest is at bottom
df = df.sort_values(date_col)

if choice == "Daily Pivots":
    keep_cols = [
        "date","day",
        "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
        "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

# Round numeric columns to 2 decimals
for col in df.select_dtypes(include=["float", "float64", "int"]).columns:
    df[col] = df[col].round(2)

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


# ---- Highlight for pivots ----
def color_hits(val):
    if val is True or str(val).lower() == "true":
        return "background-color: #98FB98"
    return ""

if choice == "Daily Pivots":
    styled = df.style.applymap(color_hits, subset=[c for c in df.columns if c.startswith("hit")])
    st.dataframe(styled, use_container_width=True, height=600)
else:
    st.dataframe(df, use_container_width=True, height=600)

st.download_button(
    "💾 Download filtered CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"{choice.lower().replace(' ','_')}_filtered.csv",
    mime="text/csv"
)

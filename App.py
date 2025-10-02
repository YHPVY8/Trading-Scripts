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

st.title("📊 Trading Dashboard")

# ---- Sidebar ----
choice = st.sidebar.selectbox("Select data set", list(TABLES.keys()))
limit = st.sidebar.number_input("Number of rows to load", value=1000, min_value=100, step=100)

table_name, date_col = TABLES[choice]

# ---- Load ----
if choice == "Daily Pivots":
    # Fetch most recent `limit` rows but show oldest first
    response = (
        sb.table(table_name)
        .select("*")
        .order(date_col, desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = df.sort_values(date_col)  # so user sees oldest at top
        # Keep only hit columns and date/day
        keep_cols = [
            "date","day",
            "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
            "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        # Format numbers
        for col in df.select_dtypes(include=["float", "float64", "int"]).columns:
            df[col] = df[col].round(2)
else:
    # Generic tables (Daily ES etc.)
    response = (
        sb.table(table_name)
        .select("*")
        .order(date_col, desc=False)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(response.data)
    if not df.empty:
        df = df.sort_values(date_col)

if df.empty:
    st.error("No data returned.")
    st.stop()

# ---- Filter UI ----
col_to_filter = st.sidebar.selectbox("Filter column", ["None"] + list(df.columns))
if col_to_filter != "None":
    unique_vals = df[col_to_filter].dropna().unique().tolist()
    if len(unique_vals) < 100:
        selected = st.sidebar.multiselect(f"Filter {col_to_filter}", sorted(unique_vals))
        if selected:
            df = df[df[col_to_filter].isin(selected)]
    else:
        text_val = st.sidebar.text_input(f"Search in {col_to_filter}")
        if text_val:
            df = df[df[col_to_filter].astype(str).str.contains(text_val, case=False, na=False)]

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

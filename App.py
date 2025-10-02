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

# ---- Load pivots table ----
st.title("📊 Trading Dashboard — Daily Pivots")

limit = st.sidebar.number_input("How many rows to load?", value=2000, min_value=100, step=100)

response = (
    sb.table("es_daily_pivot_levels")
    .select("*")
    .order("date", desc=True)
    .limit(limit)
    .execute()
)

if not response.data:
    st.error("No data returned.")
    st.stop()

df = pd.DataFrame(response.data)

# Ensure correct column order & drop extras
keep_cols = [
    "date","day",
    "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
    "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"
]
df = df[[c for c in keep_cols if c in df.columns]]

# Format dates & numbers cleanly
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"]).dt.date

# ---- Filtering UI ----
col_to_filter = st.sidebar.selectbox("Filter column", ["None"] + [c for c in df.columns if c not in ("date",)])
if col_to_filter != "None":
    unique_vals = df[col_to_filter].dropna().unique().tolist()
    # Show dropdown for discrete or text input fallback
    if len(unique_vals) < 100:
        selected = st.sidebar.multiselect(f"Filter {col_to_filter}", sorted(unique_vals))
        if selected:
            df = df[df[col_to_filter].isin(selected)]
    else:
        text_val = st.sidebar.text_input(f"Search in {col_to_filter}")
        if text_val:
            df = df[df[col_to_filter].astype(str).str.contains(text_val, case=False, na=False)]

# ---- Color hit columns ----
def color_hits(val):
    if val is True or str(val).lower() == "true":
        return "background-color: #98FB98"
    return ""

st.dataframe(
    df.style.applymap(color_hits, subset=[c for c in df.columns if c.startswith("hit")]),
    height=600,
    use_container_width=True
)

# ---- Scroll to bottom option ----
if st.button("⬇️ Scroll to latest"):
    st.write("Use the table scrollbar — newest rows are already loaded at top by default.")

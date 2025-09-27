#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client
import altair as alt
import os

# ---- CONFIG ----
st.set_page_config(page_title="Trading Dashboard", layout="wide")

# Your Supabase credentials (move to Streamlit secrets later)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLES = {
    "Daily ES": "daily_es",
    "Weekly ES": "es_weekly",
    "30m ES": "es_30m",
    "2h ES": "es_2hr",
    "4h ES": "es_4hr",
}

st.title("📊 Trading Dashboard")

# ---- Sidebar controls ----
table_choice = st.sidebar.selectbox("Select timeframe", list(TABLES.keys()))
limit = st.sidebar.number_input("How many rows to load?", value=500, min_value=50, step=50)

# ---- Load data ----
table_name = TABLES[table_choice]
st.write(f"Loading data from **{table_name}** …")
response = sb.table(table_name).select("*").order("time", desc=True).limit(limit).execute()

if not response.data:
    st.error("No data returned. Check table or credentials.")
    st.stop()

df = pd.DataFrame(response.data)
df = df.sort_values("time")  # chronological

# ---- Show data ----
st.subheader("Raw data preview")
st.dataframe(df.tail(20))

# ---- Chart ----
if all(c in df.columns for c in ["time", "close"]):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="time:T", y="close:Q")
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("Could not plot — missing time/close columns.")

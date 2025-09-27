import streamlit as st
import pandas as pd
from supabase import create_client

# ---------- Page setup ----------
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Trading Dashboard")

# ---------- Connect to Supabase ----------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Helper Functions ----------
def load_table(table_name, limit=1000, date_col="time"):
    """Fetch last `limit` rows ordered by date_col descending, then re-sort ascending."""
    rows = (
        sb.table(table_name)
        .select("*")
        .order(date_col, desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(rows.data)
    if not df.empty:
        # Sort ascending so most recent ends at bottom
        df = df.sort_values(date_col)
        # Drop id column if present
        if "id" in df.columns:
            df = df.drop(columns=["id"])
    return df

def format_numbers(df):
    """Round floats to 2 decimals."""
    num_cols = df.select_dtypes(include=["float", "float64", "int"]).columns
    df[num_cols] = df[num_cols].round(2)
    return df

def highlight_hits(val, col):
    """Highlight True/✓ in hit columns."""
    if col.lower().startswith("hit") and (val is True or str(val).lower() == "true" or val == "✓"):
        return "background-color: #98FB98"
    return ""

def styled_dataframe(df):
    return df.style.apply(
        lambda s: [highlight_hits(v, s.name) for v in s],
        axis=0
    )

# ---------- Sidebar Navigation ----------
section = st.sidebar.radio("Go to", ["Source Data", "Pivots"])

# ========== SOURCE DATA ==========
if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"]
    )
    data = load_table(table, limit=1000, date_col="time")
    if not data.empty:
        data = format_numbers(data)
        st.dataframe(
            data,
            width="stretch",
            hide_index=True,
            height=600
        )
    else:
        st.warning("No data found.")

# ========== PIVOTS ==========
elif section == "Pivots":
    st.header("Daily Pivots")
    pivots = load_table("es_daily_pivot_levels", limit=1000, date_col="date")
    if not pivots.empty:
        pivots = format_numbers(pivots)
        st.dataframe(
            styled_dataframe(pivots),
            width="stretch",
            hide_index=True,
            height=600
        )
    else:
        st.warning("No pivots found.")

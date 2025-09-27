import streamlit as st
import pandas as pd
from supabase import create_client

# ---------- Page setup ----------
st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("Trading Dashboard")

# ---- Connect to Supabase ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Sidebar Navigation ----
section = st.sidebar.radio("Go to", ["Source Data", "Pivots"])

def load_table(table_name, limit=1000):
    """Fetch last N rows sorted by time desc, then return ascending for normal reading."""
    rows = (
        sb.table(table_name)
        .select("*")
        .order("time", desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(rows.data)
    if not df.empty:
        df = df.sort_values("time")
    return df

def format_numbers(df):
    for col in df.select_dtypes(include=["float","int"]).columns:
        df[col] = df[col].round(2)
    return df

# ========== SOURCE DATA ==========
if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"]
    )
    data = load_table(table)
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
    pivots = load_table("es_daily_pivot_levels")
    if not pivots.empty:
        pivots = format_numbers(pivots)
        # Light green highlight for hit columns
        hit_cols = [c for c in pivots.columns if c.lower().startswith("hit")]
        def highlight_hits(val, col):
            if col in hit_cols and (val is True or str(val).lower() == "true" or val == "✓"):
                return "background-color: #98FB98"
            return ""
        styled = pivots.style.apply(
            lambda s: [highlight_hits(v, s.name) for v in s],
            axis=0
        )
        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            height=600
        )
    else:
        st.warning("No pivots found.")

import streamlit as st
from supabase import create_client
import pandas as pd

st.title("Trading Dashboard")

# ---- Connect to Supabase ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Sidebar Navigation ----
section = st.sidebar.radio("Go to", ["Source Data", "Pivots"])

# ========== SOURCE DATA ==========
if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"]
    )

    # fetch a bigger chunk — adjust limit if needed
    rows = sb.table(table).select("*").limit(50000).execute()
    data = pd.DataFrame(rows.data)

    if not data.empty:
        st.dataframe(
            data,
            use_container_width=True,   # full screen width
            hide_index=True,
            height=600                  # scrollable like a spreadsheet
        )
        st.caption(f"Showing {len(data)} rows (adjust limit in code if needed).")
    else:
        st.warning("No data found.")

# ========== PIVOTS ==========
elif section == "Pivots":
    st.header("Daily Pivots")

    # Directly show the pivots table from Supabase
    rows = sb.table("es_daily_pivot_levels").select("*").execute()
    pivots = pd.DataFrame(rows.data)

    if not pivots.empty:
        st.dataframe(
            pivots,
            use_container_width=True,
            hide_index=True,
            height=600
        )
        st.caption(f"Showing {len(pivots)} rows from es_daily_pivot_levels.")
    else:
        st.warning("No pivot data found in es_daily_pivot_levels.")

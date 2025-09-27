import streamlit as st
from supabase import create_client
import pandas as pd
import runpy  # to run your pivots script

st.title("Trading Dashboard")

# ---- Connect to Supabase ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Sidebar Navigation ----
section = st.sidebar.radio("Go to", ["Source Data", "Pivots"])

if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox("Select a table", ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"])
    rows = sb.table(table).select("*").limit(100).execute()
    data = pd.DataFrame(rows.data)
    if not data.empty:
        st.dataframe(data)
    else:
        st.warning("No data found.")

elif section == "Pivots":
    st.header("Daily Pivots")
    st.info("Running Supabase_Daily_Pivots.py...")
    # run your pivot script (it can return a dataframe or write output to file/db)
    try:
        runpy.run_path("Supabase_Daily_Pivots.py")
        st.success("Pivot script executed.")
    except Exception as e:
        st.error(f"Error running pivots: {e}")

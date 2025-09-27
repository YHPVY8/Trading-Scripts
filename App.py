import streamlit as st
from supabase import create_client
import pandas as pd
import runpy

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

    # show latest 10,000 rows (adjust limit as needed)
    rows = sb.table(table).select("*").limit(10000).execute()
    data = pd.DataFrame(rows.data)

    if not data.empty:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
            height=600  # scrollable like a spreadsheet
        )
    else:
        st.warning("No data found.")

# ========== PIVOTS ==========
elif section == "Pivots":
    st.header("Daily Pivots")

    if st.button("Run Daily Pivots"):
        st.info("Running Supabase_Daily_Pivots.py ...")
        try:
            # Run the pivot script
            runpy.run_path("Supabase_Daily_Pivots.py")
            st.success("Pivot script executed ✅")

            # --- Show pivots if your script outputs a CSV ---
            try:
                pivots = pd.read_csv("daily_pivots.csv")  # adjust filename if needed
                st.dataframe(
                    pivots,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
            except FileNotFoundError:
                st.info("Pivot script ran, but no CSV found. Make sure the script saves one.")
        except Exception as e:
            st.error(f"Error running pivots: {e}")

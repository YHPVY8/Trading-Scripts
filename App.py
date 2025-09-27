#!/usr/bin/env python3
import streamlit as st
from supabase import create_client
import pandas as pd
import runpy

# ----------------------------
# CONFIGURE PAGE
# ----------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        /* Remove default horizontal padding */
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        /* Make tables stretch full width */
        .stDataFrame {
            width: 100% !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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

    # Fetch more rows (adjust limit if needed)
    rows = sb.table(table).select("*").limit(20000).execute()
    data = pd.DataFrame(rows.data)

    if not data.empty:
        # --- AgGrid for spreadsheet feel ---
        from st_aggrid import AgGrid, GridOptionsBuilder

        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100)
        gb.configure_side_bar()  # optional: column visibility, filter, etc.
        grid_options = gb.build()

        AgGrid(
            data,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True
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

            # Show pivots from Supabase
            rows = sb.table("es_daily_pivot_levels").select("*").limit(20000).execute()
            pivots = pd.DataFrame(rows.data)

            if not pivots.empty:
                from st_aggrid import AgGrid, GridOptionsBuilder

                gb = GridOptionsBuilder.from_dataframe(pivots)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100)
                gb.configure_side_bar()
                grid_options = gb.build()

                AgGrid(
                    pivots,
                    gridOptions=grid_options,
                    height=600,
                    fit_columns_on_grid_load=True
                )
            else:
                st.info("No pivot data found in table `es_daily_pivot_levels`.")
        except Exception as e:
            st.error(f"Error running pivots: {e}")

#!/usr/bin/env python3
import streamlit as st
from supabase import create_client
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# ----------------------------
# CONFIGURE PAGE
# ----------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
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

    rows = sb.table(table).select("*").limit(20000).execute()
    data = pd.DataFrame(rows.data)

    if not data.empty:
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100)
        gb.configure_side_bar()

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

    rows = sb.table("es_daily_pivot_levels").select("*").limit(20000).execute()
    pivots = pd.DataFrame(rows.data)

    if not pivots.empty:
        gb = GridOptionsBuilder.from_dataframe(pivots)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100)
        gb.configure_side_bar()

        # Conditional styling: highlight True in green for Hit... columns
        green_style = JsCode("""
            function(params) {
                if (params.value === true) {
                    return {
                        'backgroundColor': '#98FB98',
                        'color': 'black'
                    }
                }
            }
        """)
        for col in pivots.columns:
            if col.lower().startswith("hit") or "pivot" in col.lower():
                gb.configure_column(col, cellStyle=green_style)

        grid_options = gb.build()

        AgGrid(
            pivots,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,   # <-- this fixes the JsCode error
            width='stretch'             # replaces use_container_width
        )
    else:
        st.info("No pivot data found in `es_daily_pivot_levels`.")

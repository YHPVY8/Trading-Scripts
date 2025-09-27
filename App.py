import streamlit as st
from supabase import create_client
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

st.set_page_config(layout="wide")
st.title("Trading Dashboard")

# ---- Connect to Supabase ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Sidebar Navigation ----
section = st.sidebar.radio("Go to", ["Source Data", "Pivots"])

# ---------- Utility: AG Grid builder ----------
def show_aggrid(df, highlight_hits=False):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, autoSize=True, value=True, precision=2)

    if highlight_hits:
        # Highlight True cells with light green
        cellstyle_jscode = JsCode(
            """
            function(params) {
              if (params.value === true) {
                return { 'backgroundColor': '#98FB98' }
              }
            }
            """
        )
        for col in df.columns:
            if col.lower().startswith("hit"):  # any column starting with Hit
                gb.configure_column(col, cellStyle=cellstyle_jscode)

    grid = gb.build()
    AgGrid(
        df,
        gridOptions=grid,
        theme="streamlit",
        fit_columns_on_grid_load=True,
        height=600,
    )

# ========== SOURCE DATA ==========
if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"]
    )

    # get last 1000 rows, newest first, then sort ascending
    rows = (
        sb.table(table)
        .select("*")
        .order("time", desc=True)
        .limit(1000)
        .execute()
    )
    data = pd.DataFrame(rows.data).sort_values("time")

    if not data.empty:
        show_aggrid(data)
    else:
        st.warning("No data found.")

# ========== PIVOTS ==========
elif section == "Pivots":
    st.header("Daily Pivots")

    # get last 1000 pivots
    rows = (
        sb.table("es_daily_pivot_levels")
        .select("*")
        .order("time", desc=True)
        .limit(1000)
        .execute()
    )
    pivots = pd.DataFrame(rows.data).sort_values("time")

    if not pivots.empty:
        show_aggrid(pivots, highlight_hits=True)
    else:
        st.warning("No pivots data found.")

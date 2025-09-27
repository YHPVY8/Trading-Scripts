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

# ---------- SOURCE DATA ----------
if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"]
    )

    # Get total row count
    count_resp = sb.table(table).select("count", count="exact").execute()
    total = count_resp.count if hasattr(count_resp, "count") else 0

    # Pull last 1,000 rows (or all if fewer)
    start = max(total - 1000, 0)
    rows = sb.table(table).select("*").range(start, total - 1).execute()
    data = pd.DataFrame(rows.data)

    if not data.empty:
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()

        st.subheader(f"Showing last {len(data)} rows (newest last)")
        AgGrid(
            data,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True,
            width='stretch'
        )
    else:
        st.warning("No data found.")

# ---------- PIVOTS ----------
elif section == "Pivots":
    st.header("Daily Pivots")
    pivots_resp = sb.table("es_daily_pivot_levels").select("*").limit(1000).execute()
    pivots = pd.DataFrame(pivots_resp.data)

    if not pivots.empty:
        gb = GridOptionsBuilder.from_dataframe(pivots)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_grid_options(domLayout='normal')

        # Green highlight for any column that starts with "hit"
        green_style = JsCode("""
        function(params) {
          if (params.value === true) {
            return {'backgroundColor': '#98FB98','color': 'black'}
          }
        }
        """)
        for col in pivots.columns:
            if col.lower().startswith("hit"):
                gb.configure_column(col, cellStyle=green_style)

        grid_options = gb.build()
        st.subheader("Pivot Levels (last 1000 rows)")
        AgGrid(
            pivots,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            width='stretch'
        )
    else:
        st.info("No pivot data found. Run your pivot script to populate the table.")

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

# ---------- Shared: auto-size JS for AgGrid ----------
auto_size_js = JsCode("""
function(params) {
  const allIds = [];
  params.columnApi.getAllColumns().forEach(col => allIds.push(col.getColId()));
  // Auto-size to fit contents (second arg=false => don't skip headers)
  params.columnApi.autoSizeColumns(allIds, false);
}
""")

# ---------- SOURCE DATA ----------
if section == "Source Data":
    st.header("Source Data")
    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"]
    )

    # Get total count, then last 1000 rows by range
    count_resp = sb.table(table).select("count", count="exact").execute()
    total = getattr(count_resp, "count", 0) or 0

    start = max(total - 1000, 0)
    rows = sb.table(table).select("*").range(start, max(total - 1, 0)).execute()
    data = pd.DataFrame(rows.data)

    if not data.empty:
        # If a time/date column exists, sort ascending (oldest -> newest)
        for c in ["time", "date"]:
            if c in data.columns:
                data = data.sort_values(c, ascending=True).reset_index(drop=True)
                break

        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_grid_options(domLayout='normal', onGridReady=auto_size_js)  # <-- auto-fit

        grid_options = gb.build()
        st.subheader(f"Showing last {len(data)} rows of ~{total} (newest last)")
        AgGrid(
            data,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            width='stretch'
        )
    else:
        st.warning("No data found.")

# ---------- PIVOTS ----------
elif section == "Pivots":
    st.header("Daily Pivots")

    # Fetch the latest 1000 rows by date DESC, then display ascending
    # (fallback to 'time' if your pivots table uses that)
    order_col = "date"
    # detect fallback
    cols_probe = sb.table("es_daily_pivot_levels").select("*").limit(1).execute()
    if cols_probe.data and isinstance(cols_probe.data, list):
        if "date" not in cols_probe.data[0] and "time" in cols_probe.data[0]:
            order_col = "time"

    pivots_resp = (
        sb.table("es_daily_pivot_levels")
        .select("*")
        .order(order_col, desc=True)
        .limit(1000)
        .execute()
    )
    pivots = pd.DataFrame(pivots_resp.data)

    if not pivots.empty:
        # Sort ascending for natural reading order (oldest -> newest)
        pivots = pivots.sort_values(order_col, ascending=True).reset_index(drop=True)

        gb = GridOptionsBuilder.from_dataframe(pivots)
        gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_side_bar()
        gb.configure_grid_options(domLayout='normal', onGridReady=auto_size_js)  # <-- auto-fit

        # Light green for boolean hit columns
        green_style = JsCode("""
        function(params) {
          if (params.value === true) {
            return {'backgroundColor': '#98FB98', 'color': 'black'};
          }
        }
        """)
        for col in pivots.columns:
            if col.lower().startswith("hit"):
                gb.configure_column(col, cellStyle=green_style)

        grid_options = gb.build()
        st.subheader("Pivot Levels (latest 1000 rows)")
        AgGrid(
            pivots,
            gridOptions=grid_options,
            height=600,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            width='stretch'
        )
    else:
        st.info("No pivot data found in `es_daily_pivot_levels`.")

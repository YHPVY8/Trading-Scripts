# App.py
import streamlit as st
import pandas as pd
from supabase import create_client
from streamlit_aggrid import AgGrid, GridOptionsBuilder, JsCode

# ---------- Page setup ----------
st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.markdown("""
<style>
/* reduce outer padding to truly use the full width */
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.title("Trading Dashboard")

# ---------- Supabase ----------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Helpers ----------
DATE_LIKE_COLS = {"time", "date", "day"}

def fetch_last_n(table: str, order_col: str, n: int) -> pd.DataFrame:
    """Get last n rows by order_col desc from Supabase, then sort ascending to view naturally."""
    res = (
        sb.table(table)
        .select("*")
        .order(order_col, desc=True)
        .limit(n)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df
    # If the order column looks like a date/time string, sort ascending for natural reading
    if order_col in df.columns:
        df = df.sort_values(order_col, ascending=True)
    return df.reset_index(drop=True)

def coerce_numeric_inplace(df: pd.DataFrame) -> None:
    """Try to cast non date-like columns to numeric so AgGrid knows they're numbers."""
    for col in df.columns:
        if col not in DATE_LIKE_COLS:
            df[col] = pd.to_numeric(df[col], errors="ignore")

def make_grid(df: pd.DataFrame, height: int = 720, highlight_hit_cols: bool = False):
    """
    Build an AgGrid with:
      - two-decimal formatting for numeric values,
      - auto-size columns to contents on first render,
      - optional green highlight for hit_* boolean columns.
    """
    gb = GridOptionsBuilder.from_dataframe(df)

    # Default column settings
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=False,
        autoHeight=False,
    )

    # Two-decimal numeric formatter (keeps true numeric types for sorting)
    value_formatter = JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined) return '';
            if (typeof params.value === 'number') { return params.value.toFixed(2); }
            let n = Number(params.value);
            return isNaN(n) ? params.value : n.toFixed(2);
        }
    """)

    # Detect numeric columns (float/int) and apply formatter
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        gb.configure_columns(numeric_cols, type=["numericColumn"], valueFormatter=value_formatter)

    # Highlight all columns that start with "hit_"
    if highlight_hit_cols:
        hit_cols = [c for c in df.columns if c.lower().startswith("hit_")]
        if hit_cols:
            cell_style = JsCode("""
                function(params) {
                    if (params.value === true || params.value === 'True') {
                        return {'backgroundColor': '#98FB98'};  // palegreen
                    }
                    return null;
                }
            """)
            gb.configure_columns(hit_cols, cellStyle=cell_style)

    # Auto-size columns to fit CONTENT on first render
    on_first_data_rendered = JsCode("""
        function(params) {
            const allCols = [];
            params.columnApi.getAllColumns().forEach(c => allCols.push(c.getColId()));
            // autosize to contents
            params.columnApi.autoSizeColumns(allCols, false);
            // and if grid is still wider, then stretch to fit
            params.api.sizeColumnsToFit();
        }
    """)
    gb.configure_grid_options(
        onFirstDataRendered=on_first_data_rendered,
        animateRows=True,
        rowSelection="single",
        suppressDragLeaveHidesColumns=True,
    )

    grid_options = gb.build()

    return AgGrid(
        df,
        gridOptions=grid_options,
        height=height,
        fit_columns_on_grid_load=False,   # we'll use autoSize + sizeColumnsToFit instead
        allow_unsafe_jscode=True,         # required to use JsCode in Streamlit Cloud
        enable_enterprise_modules=False,
        theme="balham",
    )

# ---------- UI ----------
section = st.sidebar.radio("Go to", ["Source Data", "Pivots"])

# ====== SOURCE DATA ======
if section == "Source Data":
    st.header("Source Data")

    table = st.selectbox(
        "Select a table",
        ["daily_es", "es_weekly", "es_30m", "es_2hr", "es_4hr"],
        index=0
    )

    # Determine the order column per table
    order_col = "time"   # all your source tables use a 'time' column
    df = fetch_last_n(table, order_col, n=1000)

    if df.empty:
        st.warning("No data found.")
    else:
        # Make sure numbers are numeric for proper sorting & formatting
        coerce_numeric_inplace(df)
        make_grid(df, height=740, highlight_hit_cols=False)

# ====== PIVOTS ======
elif section == "Pivots":
    st.header("Daily Pivots (latest 1,000)")

    # Your pivot table uses a 'date' column (string 'YYYY-MM-DD')
    order_col = "date"
    table = "es_daily_pivot_levels"

    pivots = fetch_last_n(table, order_col, n=1000)

    if pivots.empty:
        st.warning("No pivot data found.")
    else:
        # Ensure numeric columns are numeric for formatting/sorting
        coerce_numeric_inplace(pivots)

        # Show with highlight for all hit_* columns and 2-decimal formatting
        make_grid(pivots, height=740, highlight_hit_cols=True)

#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client

# import the tables/view builder
from views_config import build_tables

# import the table with price levels below the pivot "hits"
from views_extras import render_current_levels

# NEW: SPX helper (filters DF + shows metrics; keeps generic styling)
from views_extras import spx_opening_range_filter_and_metrics


# ---- CONFIG ----
st.set_page_config(page_title="Trading Dashboard", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# CHANGED: pull base tables + YAML/DB-defined views
TABLES = build_tables(sb)

st.title("Trading Dashboard")

# ---- Sidebar ----
choice = st.sidebar.selectbox("Select data set", list(TABLES.keys()))
limit = st.sidebar.number_input("Number of rows to load", value=1000, min_value=100, step=100)

# NEW: normalize the selected config (tuple OR dict)
cfg = TABLES[choice]
if isinstance(cfg, tuple):
    table_name, date_col = cfg
    keep_cols = None
    header_labels = {}
else:
    table_name = cfg["table"]
    date_col = cfg.get("date_col", "date")
    keep_cols = cfg.get("keep")
    header_labels = cfg.get("labels", {})

# ---- Load ----
response = (
    sb.table(table_name)
      .select("*")
      .order(date_col, desc=True)
      .limit(limit)
      .execute()
)
df = pd.DataFrame(response.data)

if df.empty:
    st.error("No data returned.")
    st.stop()

# --- Sort so latest is at bottom (tolerant to schema drift) ---
if date_col not in df.columns:
    for fallback in ["trade_date", "date", "time"]:
        if fallback in df.columns:
            date_col = fallback
            break

if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

# --- CLEANUP PER DATASET ---
if choice == "Daily ES" and "id" in df.columns:
    df = df.drop(columns=["id"])

# Format common date-like columns
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

if "trade_date" in df.columns:
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

if "time" in df.columns:
    if choice in ["Daily ES", "Weekly ES"]:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M")

# --- Restrict columns for fixed datasets ---
if choice == "Daily Pivots":
    keep_cols_fixed = ["date","day","hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                       "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols_fixed if c in df.columns]]

elif choice == "Weekly Pivots":
    keep_cols_fixed = ["date","hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                       "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols_fixed if c in df.columns]]

elif choice in ["2h Pivots","4h Pivots","30m Pivots"]:
    keep_cols_fixed = ["time","globex_date","day","hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
                       "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"]
    df = df[[c for c in keep_cols_fixed if c in df.columns]]

elif choice in ["Daily ES","Weekly ES","30m ES","2h ES","4h ES"]:
    keep_fixed = ["time","open","high","low","close","200MA","50MA","20MA","10MA","5MA","Volume","ATR"]
    df = df[[c for c in keep_fixed if c in df.columns]]

elif choice == "Range Extensions":
    keep_fixed = [
        "date","day","range","14_Day_Avg_Range","range_gt_avg","range_gt_80_avg",
        "op_lo","op_lo_14_avg","op_lo_gt_avg","range_gt_80_op_lo",
        "hi_op","hi_op_14_avg","hi_op_gt_avg","range_gt_80_hi_op",
        "hit_both_80","hit_both_14_avg",
        "range_gt_120_avg","range_gt_120_op_lo","range_gt_120_hi_op"
    ]
    df = df[[c for c in keep_fixed if c in df.columns]]

elif choice in ["RTH Pivots", "ON Pivots"]:
    # Mirror Daily Pivots but use trade_date instead of date
    keep_cols_fixed = [
        "trade_date", "day",
        "hit_pivot","hit_r025","hit_s025","hit_r05","hit_s05",
        "hit_r1","hit_s1","hit_r15","hit_s15","hit_r2","hit_s2","hit_r3","hit_s3"
    ]
    df = df[[c for c in keep_cols_fixed if c in df.columns]]

# --- Generic view-level subset (from YAML/DB views) ---
if keep_cols:
    # Ensure SPX extension columns are kept even if the view omitted them
    if choice == "SPX Opening Range":
        required_spx = [
            "hit_20_up","hit_50_up","hit_100_up",
            "hit_20_down","hit_50_down","hit_100_down",
            "max_ext_up","max_ext_down",
        ]
        # preserve original order while appending any missing required fields
        keep_cols = list(dict.fromkeys(list(keep_cols) + required_spx))
    df = df[[c for c in keep_cols if c in df.columns]]

# --- SPX Opening Range: filter to a single OR window + show metrics (keep generic styling)
if choice == "SPX Opening Range":
    df = spx_opening_range_filter_and_metrics(df)
    if df.empty:
        st.info("No rows after applying the SPX window filter.")
        st.stop()

# ---- Format all numeric columns ----
for col in df.columns:
    if df[col].dtype in ["float64", "float32", "int64", "int32"]:
        if "volume" in col.lower():
            df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else x)
        else:
            df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)

# ---- Multi-condition filter ----
filters = []
num_filters = st.sidebar.number_input("Number of filters", 0, 5, 0)

if st.sidebar.button("Clear Filters"):
    st.rerun()

for n in range(num_filters):
    col = st.sidebar.selectbox(f"Column #{n+1}", df.columns, key=f"fcol{n}")
    op = st.sidebar.selectbox(f"Op #{n+1}", ["equals","contains","greater than","less than"], key=f"fop{n}")
    val = st.sidebar.text_input(f"Value #{n+1}", key=f"fval{n}")
    if col and op and val:
        filters.append((col,op,val))

for col, op, val in filters:
    if op == "equals":
        df = df[df[col].astype(str) == val]
    elif op == "contains":
        df = df[df[col].astype(str).str.contains(val, case=False, na=False)]
    elif op == "greater than":
        df = df[pd.to_numeric(df[col], errors='coerce') > float(val)]
    elif op == "less than":
        df = df[pd.to_numeric(df[col], errors='coerce') < float(val)]

# -------------------- Styling (Headers + Borders + Highlights) --------------------
def color_hits(val):
    if val is True or str(val).lower() == "true":
        return "background-color: #98FB98"
    return ""

THICK_BORDER_AFTER = {
    "Daily Pivots": ["day","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
    "2h Pivots":    ["day","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
    "4h Pivots":    ["day","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
    "30m Pivots":   ["day","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
    "Weekly Pivots": ["date","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
    "RTH Pivots": ["day","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
    "ON Pivots":  ["day","hit_pivot","hit_s025","hit_s05","hit_s1","hit_s15","hit_s2","hit_s3"],
}

HEADER_LABELS = {
    "Daily Pivots": {
        "date": "Date", "day": "Day",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "Weekly Pivots": {
        "date": "Date",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "2h Pivots": {
        "time": "Time", "globex_date": "Globex Date", "day": "Day",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "4h Pivots": {
        "time": "Time", "globex_date": "Globex Date", "day": "Day",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "30m Pivots": {
        "time": "Time", "globex_date": "Globex Date", "day": "Day",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "RTH Pivots": {
        "trade_date": "Date", "day": "Day",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "ON Pivots": {
        "trade_date": "Date", "day": "Day",
        "hit_pivot": "Pivot", "hit_r025": "R025", "hit_s025": "S025",
        "hit_r05": "R0.5", "hit_s05": "S0.5",
        "hit_r1": "R1", "hit_s1": "S1",
        "hit_r15": "R1.5", "hit_s15": "S1.5",
        "hit_r2": "R2", "hit_s2": "S2",
        "hit_r3": "R3", "hit_s3": "S3",
    },
    "SPX Opening Range": {
        "trade_date": "Date",
        "day_of_week": "Day",
        "symbol": "Symbol",
        "open_location": "Open Location",
        "or_window": "OR Time",
        "orh": "ORH",
        "orl": "ORL",
        "or_range": "OR Range",
        "first_break": "First Break",
        "broke_up": "Broke Up",
        "broke_down": "Broke Down",
        "broke_both": "Broke Both",
    },
}

# Detect boolean-like columns anywhere in the dataframe
def detect_bool_like_columns(df: pd.DataFrame):
    bool_cols = []
    for c in df.columns:
        s = df[c]
        if s.dtype == bool:
            bool_cols.append(c)
        else:
            nonnull = s.dropna().astype(str).str.lower().unique()
            if len(nonnull) > 0 and set(nonnull).issubset({"true", "false"}):
                bool_cols.append(c)
    return bool_cols

# Build the styled table
def make_excelish_styler(df: pd.DataFrame, choice: str):
    styler = df.style.hide(axis="index")
    table_styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {"selector": "th, td", "props": [("border", "1px solid #E5E7EB"), ("padding", "6px 8px")]},
        {"selector": "thead th", "props": [("font-weight", "700"), ("color", "#000")]}
    ]

    # Add thick right borders for key columns (only if they exist in df)
    border_after = THICK_BORDER_AFTER.get(choice, [])
    border_after = [c for c in border_after if c in df.columns]
    nths = [df.columns.get_loc(c) + 1 for c in border_after]
    for n in nths:
        table_styles.append({"selector": f"td:nth-child({n})", "props": [("border-right", "3px solid #000")]})
        table_styles.append({"selector": f"th:nth-child({n})", "props": [("border-right", "3px solid #000")]})

    styler = styler.set_table_styles(table_styles)

    # Highlight ALL boolean-like cols, plus legacy hit/gt cols
    bool_cols = detect_bool_like_columns(df)
    hit_cols = [c for c in df.columns if any(s in c.lower() for s in ["hit", "gt", ">"])]
    highlight_cols = sorted(set(bool_cols + hit_cols))
    if highlight_cols:
        styler = styler.map(color_hits, subset=highlight_cols)

    return styler

# ---- Display (sticky header + scrollable container) ----
styled = make_excelish_styler(df, choice)
html_table = styled.to_html()

# Merge global header labels with per-view labels
labels = HEADER_LABELS.get(choice, {}).copy()
labels.update(header_labels)
for orig, new in labels.items():
    html_table = html_table.replace(f">{orig}<", f">{new}<")

st.markdown("""
    <style>
    .scroll-table-container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #E5E7EB;
    }
    .scroll-table-container table {
        width: 100%;
        border-collapse: collapse;
    }
    /* Sticky header with medium-grey bg + persistent black "bottom border" */
    .scroll-table-container thead th {
        position: sticky;
        top: 0;
        z-index: 3;
        background-color: #d0d0d0 !important;
        color: #000;
        font-weight: 700;
        background-image: linear-gradient(to bottom,
            rgba(0,0,0,0) calc(100% - 3px),
            #000 calc(100% - 3px),
            #000 100%
        ) !important;
        background-clip: padding-box;
        border-bottom: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="scroll-table-container">{html_table}</div>', unsafe_allow_html=True)

# optional spacer
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# Pivot levels block (only renders for pivot tables)
render_current_levels(sb, choice, table_name, date_col)

# Download
st.download_button(
    "💾 Download filtered CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name=f"{choice.lower().replace(' ','_')}_filtered.csv",
    mime="text/csv",
    key=f"download_{choice.lower().replace(' ','_')}"
)

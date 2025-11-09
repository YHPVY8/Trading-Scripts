# pages/01_Views_Manager.py
import json
import streamlit as st
import pandas as pd
from supabase import create_client

st.set_page_config(page_title="Views Manager", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("Views Manager")

# --- Rerun helper (works across Streamlit versions) ---
def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# --- Helpers ---
def list_base_tables():
    # Keep in sync with BASE_TABLES in views_config.py
    return [
        ("daily_es", "Daily ES"),
        ("es_weekly", "Weekly ES"),
        ("es_30m", "30m ES"),
        ("es_2hr", "2h ES"),
        ("es_4hr", "4h ES"),
        ("es_daily_pivot_levels", "Daily Pivots"),
        ("es_weekly_pivot_levels", "Weekly Pivots"),
        ("es_2hr_pivot_levels", "2h Pivots"),
        ("es_4hr_pivot_levels", "4h Pivots"),
        ("es_30m_pivot_levels", "30m Pivots"),
        ("es_range_extensions", "Range Extensions"),
        ("es_trade_day_summary", "ES Trade Day Summary"),
        ("es_rth_pivot_levels", "RTH Pivots"),
        ("es_on_pivot_levels",  "ON Pivots"),
        ("spx_opening_range_stats", "SPX Opening Range"),

    ]

def get_columns_for_table(table_name: str):
    # Try to fetch one row to inspect columns
    try:
        data = sb.table(table_name).select("*").limit(1).execute().data
        if not data:
            return []
        return list(pd.DataFrame(data).columns)
    except Exception:
        return []

def load_views():
    try:
        rows = (sb.table("dashboard_views")
                  .select("*")
                  .order("sort_order")
                  .execute()
                  .data)
        return rows
    except Exception as e:
        st.error("Could not load dashboard_views from Supabase.")
        return []

def save_view(payload, existing_id=None):
    try:
        if existing_id:
            resp = (sb.table("dashboard_views")
                      .update(payload)
                      .eq("id", existing_id)
                      .execute())
        else:
            resp = (sb.table("dashboard_views")
                      .upsert(payload, on_conflict="view_name")
                      .execute())
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def delete_view(view_id):
    try:
        sb.table("dashboard_views").delete().eq("id", view_id).execute()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


# --- UI: Existing views list ---
st.subheader("Existing Views")

views = load_views()
if views:
    for r in views:
        with st.expander(f"{r['sort_order']:02d} · {r['view_name']} ({'enabled' if r.get('is_enabled') else 'disabled'})", expanded=False):
            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                new_name = st.text_input("View name", r["view_name"], key=f"nm{r['id']}")
                is_enabled = st.checkbox("Enabled", value=r.get("is_enabled", True), key=f"en{r['id']}")
                sort_order = st.number_input("Sort order", value=int(r.get("sort_order", 0)), step=1, key=f"so{r['id']}")
            with c2:
                tbl = st.text_input("Source table", r["table_name"], key=f"tb{r['id']}")
                date_col = st.text_input("Date col", r.get("date_col","date"), key=f"dc{r['id']}")
                cols_available = get_columns_for_table(tbl)
                keep_default = r.get("keep_columns") or []
                keep = st.multiselect("Columns to keep (order matters)", cols_available, default=keep_default, key=f"kp{r['id']}")
            with c3:
                # Simple labels editor: show mapping for selected keep columns
                labels = r.get("labels") or {}
                st.caption("Header labels (optional)")
                new_labels = {}
                for col in keep or []:
                    new_labels[col] = st.text_input(f"Label: {col}", value=labels.get(col, col), key=f"lb{r['id']}{col}")

            col_a, col_b, col_c = st.columns([1,1,1])
            with col_a:
                if st.button("Save changes", key=f"sv{r['id']}"):
                    payload = {
                        "view_name": new_name,
                        "table_name": tbl,
                        "date_col": date_col or "date",
                        "keep_columns": keep,
                        "labels": new_labels,
                        "is_enabled": is_enabled,
                        "sort_order": int(sort_order),
                    }
                    if save_view(payload, existing_id=r["id"]):
                        st.success("Saved")
                        _safe_rerun()
            with col_b:
                if st.button("Duplicate", key=f"dup{r['id']}"):
                    payload = {
                        "view_name": f"{r['view_name']} (copy)",
                        "table_name": r["table_name"],
                        "date_col": r.get("date_col","date"),
                        "keep_columns": r.get("keep_columns") or [],
                        "labels": r.get("labels") or {},
                        "is_enabled": r.get("is_enabled", True),
                        "sort_order": int(r.get("sort_order", 0)) + 1,
                    }
                    if save_view(payload, existing_id=None):
                        st.success("Duplicated")
                        _safe_rerun()
            with col_c:
                if st.button("Delete", key=f"del{r['id']}"):
                    if delete_view(r["id"]):
                        st.success("Deleted")
                        _safe_rerun()
else:
    st.info("No views yet. Create one below.")

st.markdown("---")

# --- UI: Create new view ---
st.subheader("Create New View")

base_tables = list_base_tables()
tbl_map = {label: name for name, label in base_tables}
label_choice = st.selectbox("Source table", [lbl for _, lbl in base_tables], index=base_tables.index(("es_trade_day_summary","ES Trade Day Summary")) if ("es_trade_day_summary","ES Trade Day Summary") in base_tables else 0)
table_name = tbl_map[label_choice]

cols = get_columns_for_table(table_name)
if not cols:
    st.warning("Could not infer columns from the source table (empty or unreachable). You can still type column names manually.")
view_name = st.text_input("View name", value="New View")
date_col = st.selectbox("Date column", options=(["trade_date","date","time"] + cols), index=0 if "trade_date" in cols else (1 if "date" in cols else 2))

keep = st.multiselect("Columns to keep (order matters)", options=cols, default=[c for c in ["trade_date","day"] if c in cols])

st.caption("Optional header labels:")
labels = {}
for col in keep:
    labels[col] = st.text_input(f"Label for {col}", value=col, key=f"newlbl_{col}")

cL, cR = st.columns([1,2])
with cL:
    sort_order = st.number_input("Sort order", value=0, step=1)
    is_enabled = st.checkbox("Enabled", value=True)

if st.button("Save new view"):
    payload = {
        "view_name": view_name,
        "table_name": table_name,
        "date_col": date_col,
        "keep_columns": keep,
        "labels": labels,
        "is_enabled": is_enabled,
        "sort_order": int(sort_order),
    }
    if save_view(payload, existing_id=None):
        st.success("Created")
        _safe_rerun()

import streamlit as st
import pandas as pd
from supabase import create_client
from datetime import date, timedelta

# -------------------- Existing content (kept) --------------------
LEVEL_ALIASES = [
    ("Pivot", ["Pivot", "pivot"]),
    ("R0.5",  ["R05", "r05"]),
    ("S0.5",  ["S05", "s05"]),
    ("R1",    ["R1",  "r1"]),
    ("S1",    ["S1",  "s1"]),
    ("R1.5",  ["R15", "r15"]),
    ("S1.5",  ["S15", "s15"]),
    ("R2",    ["R2",  "r2"]),
    ("S2",    ["S2",  "s2"]),
    ("R3",    ["R3","r3"]),
    ("S3",    ["S3","s3"]),
]

PIVOT_TABLES = {
    "es_daily_pivot_levels",
    "es_weekly_pivot_levels",
    "es_2hr_pivot_levels",
    "es_4hr_pivot_levels",
    "es_30m_pivot_levels",
    "es_rth_pivot_levels",
    "es_on_pivot_levels",
    
    # --- Added GC pivot tables ---
    "gc_daily_pivot_levels",
    "gc_weekly_pivot_levels",
}

# --- Render current pivot levels (generalized for GC and ES) ---
def render_current_levels(sb, choice: str, table_name: str, date_col: str):
    norm = _normalize_table_name(table_name)
    is_pivot = (norm in PIVOT_TABLES) or norm.endswith("_pivot_levels")
    if not is_pivot:
        # Small diagnostic so you know why nothing prints
        st.caption(f"ℹ️ Skipping levels (table '{table_name}' not recognized as a pivot-level table).")
        return

    levels = fetch_current_levels(sb, table_name, date_col)

    if not any(levels.values()):
        # Show what keys were present in the latest record to help diagnose column-name mismatches
        try:
            probe = (
                sb.table(table_name)
                  .select("*")
                  .order(date_col, desc=True)
                  .limit(1)
                  .execute()
            )
            keys = ", ".join(sorted((probe.data or [{}])[0].keys()))
        except Exception:
            keys = "unknown"
        st.caption(f"ℹ️ No current levels found (table='{table_name}', date_col='{date_col}'). Latest row keys: {keys}")
        return

    st.markdown("#### Current period levels")
    lines = []
    for label, _ in LEVEL_ALIASES:
        v = levels.get(label)
        if v is not None:
            try:
                lines.append(f"**{label}**  {v:,.2f}")
            except Exception:
                lines.append(f"**{label}**  {v}")
    if lines:
        st.markdown("\n".join(f"- {ln}" for ln in lines))


# --- GC Daily Pivots metrics ---
def render_gc_daily_pivots_metrics(df: pd.DataFrame) -> None:
    """
    Dynamic tiles for GC Daily Pivots. Runs on the *already-filtered* DataFrame.
    """
    if df is None or df.empty:
        return

    dff = df.copy()

    # Helper functions for bool/numeric conversions
    def _to_bool(col: str) -> pd.Series:
        if col not in dff: return pd.Series(dtype="boolean")
        s = dff[col]
        if s.dtype == bool:
            return s.astype("boolean")
        return (
            s.astype(str).str.strip().str.lower()
             .map({"true": True, "1": True, "yes": True, "y": True,
                   "false": False, "0": False, "no": False, "n": False})
             .astype("boolean")
        )

    def _num(col: str) -> pd.Series:
        if col not in dff: return pd.Series(dtype="float64")
        return pd.to_numeric(dff[col], errors="coerce")

    # Metrics for GC Daily
    gc_pivot = _num("pivot")
    gc_r1 = _num("r1")
    gc_s1 = _num("s1")
    gc_r2 = _num("r2")
    gc_s2 = _num("s2")
    gc_r3 = _num("r3")
    gc_s3 = _num("s3")

    # Layout
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### GC Daily Pivots")
        st.markdown(f"**Pivot:** {gc_pivot.mean():.2f}")
        st.markdown(f"**R1:** {gc_r1.mean():.2f}")
        st.markdown(f"**S1:** {gc_s1.mean():.2f}")
        st.markdown(f"**R2:** {gc_r2.mean():.2f}")
        st.markdown(f"**S2:** {gc_s2.mean():.2f}")

    with c2:
        st.markdown("### GC Extended Pivots")
        st.markdown(f"**R3:** {gc_r3.mean():.2f}")
        st.markdown(f"**S3:** {gc_s3.mean():.2f}")

    return dff


# --- GC Weekly Pivots metrics ---
def render_gc_weekly_pivots_metrics(df: pd.DataFrame) -> None:
    """
    Dynamic tiles for GC Weekly Pivots. Runs on the *already-filtered* DataFrame.
    """
    if df is None or df.empty:
        return

    dff = df.copy()

    # Metrics for GC Weekly
    gc_weekly_pivot = _num("pivot")
    gc_weekly_r1 = _num("r1")
    gc_weekly_s1 = _num("s1")
    gc_weekly_r2 = _num("r2")
    gc_weekly_s2 = _num("s2")
    gc_weekly_r3 = _num("r3")
    gc_weekly_s3 = _num("s3")

    # Layout
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### GC Weekly Pivots")
        st.markdown(f"**Pivot:** {gc_weekly_pivot.mean():.2f}")
        st.markdown(f"**R1:** {gc_weekly_r1.mean():.2f}")
        st.markdown(f"**S1:** {gc_weekly_s1.mean():.2f}")
        st.markdown(f"**R2:** {gc_weekly_r2.mean():.2f}")
        st.markdown(f"**S2:** {gc_weekly_s2.mean():.2f}")

    with c2:
        st.markdown("### GC Extended Pivots")
        st.markdown(f"**R3:** {gc_weekly_r3.mean():.2f}")
        st.markdown(f"**S3:** {gc_weekly_s3.mean():.2f}")

    return dff


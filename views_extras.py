# views_extras.py
import streamlit as st

# Map display label → possible underlying column names in DB
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
    # Uncomment if you want:
    # ("R3",    ["R3","r3"]),
    # ("S3",    ["S3","s3"]),
]

# We’ll detect “pivot tables” by the underlying table name, not by the UI label
PIVOT_TABLES = {
    "es_daily_pivot_levels",
    "es_weekly_pivot_levels",
    "es_2hr_pivot_levels",
    "es_4hr_pivot_levels",
    "es_30m_pivot_levels",
    "es_rth_pivot_levels",
    "es_on_pivot_levels",
}

def _first_present(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None

def fetch_current_levels(sb, table_name: str, date_col: str) -> dict:
    try:
        needed = {date_col}
        for _, cands in LEVEL_ALIASES:
            needed.update(cands)
        select_str = ",".join(sorted(needed))

        resp = (
            sb.table(table_name)
              .select(select_str)
              .order(date_col, desc=True)
              .limit(1)
              .execute()
        )
        rows = resp.data or []
        if not rows:
            return {}
        rec = rows[0]

        out = {}
        for label, cands in LEVEL_ALIASES:
            v = _first_present(rec, cands)
            # try coercion to float for formatting
            if isinstance(v, str):
                try:
                    v = float(v.replace(",", ""))
                except Exception:
                    pass
            out[label] = v
        return out
    except Exception:
        return {}

def render_current_levels(sb, choice: str, table_name: str, date_col: str):
    # Show levels for any of the known pivot-level tables (works even if UI label is a custom View name)
    if (table_name not in PIVOT_TABLES) and (not table_name.endswith("_pivot_levels")):
        return

    levels = fetch_current_levels(sb, table_name, date_col)

    # Debug caption if nothing found (you can remove this once confirmed)
    if not any(levels.values()):
        st.caption(f"ℹ️ No current levels found (table='{table_name}', date_col='{date_col}').")
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
        st.markdown("\n".join(lines))

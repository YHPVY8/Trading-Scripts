# pages/Trading Performance.py
#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client
import hashlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ---- Supabase client (your pattern) ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ---------- Utilities ----------
def _clean_header(name: str) -> str:
    """strip BOM/spaces, collapse inner spaces, lowercase"""
    if name is None:
        return ""
    name = str(name).replace("\ufeff", "").strip()
    name = " ".join(name.split())
    return name.lower()

def _to_iso_est(ts_str: str):
    """
    Convert 'MM/DD/YYYY HH:MM:SS -03:00' -> 'YYYY-MM-DD HH:MM:SS' (no tz)
    Returns None if unparsable.
    """
    if not ts_str:
        return None
    s = str(ts_str).strip()
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and len(parts[1]) == 6 and parts[1][3] == ":" and (parts[1][0] in "+-"):
        s = parts[0]
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _synth_id(row: dict) -> str:
    """Deterministic synthesized Id if CSV lacks an Id."""
    key = "|".join([
        str(row.get("contractname", "")).strip().upper(),
        str(row.get("enteredat", "")).strip(),
        str(row.get("exitedat", "")).strip(),
        str(row.get("entryprice", "")).strip(),
        str(row.get("exitprice", "")).strip(),
        str(row.get("size", "")).strip(),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"SYN-{digest}"

def _as_float(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        v = pd.to_numeric(x, errors="coerce")
        return None if pd.isna(v) else float(v)

# ---------- CSV -> canonical rows (via pandas) ----------
def _read_csv_to_rows(uploaded_bytes: bytes):
    """
    Robust CSV reader using pandas:
    - auto-detect delimiter (sep=None, engine='python')
    - preserves all columns
    - normalizes headers
    Returns (rows, debug_info)
    """
    # Let pandas guess delimiter; handle BOM automatically
    df = pd.read_csv(
        pd.io.common.BytesIO(uploaded_bytes),
        sep=None, engine="python", dtype=str, keep_default_na=False
    )

    # Normalize headers to canonical keys
    original_headers = list(df.columns)
    norm_headers = [_clean_header(h) for h in original_headers]
    df.columns = norm_headers

    # Alias map → canonical
    alias_map = {
        "id": "id",
        "trade id": "id",
        "tradeid": "id",
        "external id": "id",

        "contractname": "contractname",
        "contract": "contractname",
        "market": "contractname",
        "symbol": "contractname",

        "enteredat": "enteredat",
        "entry time": "enteredat",
        "entry": "enteredat",

        "exitedat": "exitedat",
        "exit time": "exitedat",
        "exit": "exitedat",

        "entryprice": "entryprice",
        "entry price": "entryprice",
        "exitprice": "exitprice",
        "exit price": "exitprice",

        "size": "size",
        "quantity": "size",
        "qty": "size",

        "type": "type",
        "side": "type",

        "pnl": "pnl",
        "p&l": "pnl",
        "profit": "pnl",

        "fees": "fees",
        "fee": "fees",

        "commissions": "commissions",
        "commission": "commissions",

        # Extras we ignore downstream but keep in debug
        "tradeday": "tradeday",
        "tradeduration": "tradeduration",
    }

    # Collapse aliases: take the first present alias for each canonical
    canonical_cols = {}
    for src, canon in alias_map.items():
        if src in df.columns and canon not in canonical_cols:
            canonical_cols[canon] = src

    # Build a canonical dict row-by-row
    rows = []
    for _, r in df.iterrows():
        row = {}
        for canon, src in canonical_cols.items():
            row[canon] = r.get(src, "")
        rows.append(row)

    debug = {
        "original_headers": original_headers,
        "normalized_headers": norm_headers,
        "canonical_mapped": canonical_cols,
        "row_count": len(rows),
    }
    return rows, debug

# ---------- Upsert ----------
def _upsert_trades_from_rows(rows):
    """
    Upsert canonical rows into tj_trades.
    Returns (inserted, skipped, errors, sample_payloads).
    """
    inserted = skipped = 0
    errs = []
    samples = []

    for r in rows:
        try:
            ext_id = (r.get("id") or "").strip()
            if not ext_id:
                ext_id = _synth_id(r)  # safe fallback

            t = (r.get("type", "") or "").strip().lower()
            if t in ("long", "buy", "b"):
                side = "long"
            elif t in ("short", "sell", "s"):
                side = "short"
            else:
                side = None

            qty_val = r.get("size")
            contract = (r.get("contractname") or "").strip().upper()
            symbol = contract or None

            f1 = _as_float(r.get("fees")) or 0.0
            f2 = _as_float(r.get("commissions")) or 0.0
            fees_val = float(f1) + float(f2)

            payload = {
                "user_id": USER_ID,
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": _to_iso_est(r.get("enteredat")),
                "exit_ts_est":  _to_iso_est(r.get("exitedat")),
                "entry_px":  _as_float(r.get("entryprice")),
                "exit_px":   _as_float(r.get("exitprice")),
                "qty":       _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees":      fees_val,
                "source": "csv",
            }

            if len(samples) < 5:
                samples.append(payload.copy())

            if not payload["entry_ts_est"] or not payload["exit_ts_est"]:
                raise ValueError(f"Bad timestamps: {r.get('enteredat')} / {r.get('exitedat')}")

            sb.table("tj_trades").upsert(payload, on_conflict="user_id,external_trade_id").execute()
            inserted += 1

        except Exception as e:
            skipped += 1
            if len(errs) < 15:
                errs.append({"external_trade_id": r.get("id"), "error": repr(e)})

    return inserted, skipped, errs, samples

@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    res = sb.table("tj_trades").select(
        "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
        "pnl_gross,fees,pnl_net,planned_risk,r_multiple,review_status"
    ).eq("user_id", USER_ID).order("entry_ts_est", desc=True).limit(limit).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"]  = pd.to_datetime(df["exit_ts_est"])
    return df

def _save_comments(trade_ids, body):
    if not body or not trade_ids:
        return
    rows = [{"trade_id": tid, "user_id": USER_ID, "body": body} for tid in trade_ids]
    sb.table("tj_trade_comments").insert(rows).execute()

def _save_tags(trade_ids, tags):
    if not tags or not trade_ids:
        return
    rows = [{"trade_id": tid, "user_id": USER_ID, "tag": t} for tid in trade_ids for t in tags]
    sb.table("tj_trade_tags").upsert(rows, on_conflict="trade_id,tag").execute()

def _add_to_group(trade_ids, group_id=None, new_group_name=None, notes=None):
    if new_group_name:
        g = sb.table("tj_trade_groups").insert(
            {"user_id": USER_ID, "name": new_group_name, "notes": notes}
        ).execute().data[0]
        group_id = g["id"]
    if group_id and trade_ids:
        rows = [{"group_id": group_id, "trade_id": tid} for tid in trade_ids]
        sb.table("tj_trade_group_members").upsert(rows, on_conflict="group_id,trade_id").execute()
    return group_id

def _get_groups():
    res = sb.table("tj_trade_groups").select("id,name,created_at").eq("user_id", USER_ID).order("created_at", desc=True).execute()
    return res.data or []

# ---------- UI ----------
st.title("Trading Performance (EST)")

tab_upload, tab_trades, tab_groups, tab_guards = st.tabs(["Upload", "Trades", "Groups", "Guardrails"])

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    up = st.file_uploader("Drop your trade export (tabs/commas OK; headers auto-detected)", type=["csv"])
    if up:
        rows, dbg = _read_csv_to_rows(up.read())

        # Diagnostics: what we actually saw
        st.caption("Original headers: " + ", ".join(dbg["original_headers"]))
        st.caption("Normalized headers: " + ", ".join(dbg["normalized_headers"]))
        st.caption("Canonical mapping: " + ", ".join([f"{v}→{k}" for k, v in dbg["canonical_mapped"].items()]))
        st.caption(f"Row count detected: {dbg['row_count']}")

        if not rows:
            st.error("No data rows found after parsing.")
        else:
            ins, skip, errs, samples = _upsert_trades_from_rows(rows)
            if ins > 0:
                st.success(f"Imported {ins} rows (skipped {skip}).")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Imported {ins} rows (skipped {skip}).")
                if samples:
                    with st.expander("Sample payloads attempted (first 5)"):
                        st.json(samples)
                if errs:
                    with st.expander("Sample errors (first 15)"):
                        st.json(errs)

# ---- Trades ----
with tab_trades:
    st.subheader("Trades")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet — upload a CSV in the Upload tab.")
    else:
        orig = df.copy()
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        rename_map = {"entry_ts_est": "Entry (EST)", "exit_ts_est": "Exit (EST)"}
        df_display = df.rename(columns=rename_map)

        editable_cols = ["selected", "planned_risk", "r_multiple", "review_status"]
        edited = st.data_editor(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("✓", help="Select for bulk actions"),
                "planned_risk": st.column_config.NumberColumn("Planned Risk", step=0.01),
                "r_multiple": st.column_config.NumberColumn("R Multiple", step=0.01),
                "review_status": st.column_config.SelectboxColumn(
                    "Review Status", options=["unreviewed", "flagged", "reviewed"]
                ),
            },
            disabled=[c for c in df_display.columns if c not in editable_cols],
            num_rows="fixed",
        )
        edited = edited.rename(columns={v: k for k, v in rename_map.items()})

        diff_cols = ["planned_risk", "r_multiple", "review_status"]
        to_update = []
        merged = edited.merge(orig[["id"] + diff_cols], on="id", suffixes=("", "_old"))
        for _, r in merged.iterrows():
            changed = any(
                (pd.isna(r[c]) and not pd.isna(r[f"{c}_old"])) or
                (not pd.isna(r[c]) and pd.isna(r[f"{c}_old"])) or
                (r[c] != r[f"{c}_old"])
                for c in diff_cols
            )
            if changed:
                to_update.append({
                    "id": r["id"],
                    "planned_risk": r["planned_risk"],
                    "r_multiple": r["r_multiple"],
                    "review_status": r["review_status"],
                })
        if to_update:
            sb.table("tj_trades").upsert(to_update, on_conflict="id").execute()
            st.toast(f"Saved {len(to_update)} inline edit(s)")
            st.cache_data.clear()

        selected_ids = edited.loc[edited["selected"] == True, "id"].tolist()
        with st.form("bulk_actions_native", clear_on_submit=True):
            st.write(f"Selected: {len(selected_ids)}")
            comment = st.text_area("Add comment (markdown)")
            tag_str = st.text_input("Add tags (comma-separated)")
            gmode = st.radio("Group", ["None", "Add to existing", "Create new"], horizontal=True)

            existing = None
            new_group = None
            notes = None
            if gmode == "Add to existing":
                groups = _get_groups()
                if groups:
                    labels = [f"{g['name']} ({g['id'][:6]})" for g in groups]
                    idx = st.selectbox(
                        "Pick group",
                        list(range(len(labels))) if labels else [],
                        format_func=lambda i: labels[i] if labels else None
                    )
                    if groups and len(groups) > 0:
                        existing = groups[idx]["id"]
                else:
                    st.info("No groups yet — choose 'Create new'.")
            elif gmode == "Create new":
                new_group = st.text_input("New group name")
                notes = st.text_input("Group notes (optional)")

            do = st.form_submit_button("Apply")

        if do and selected_ids:
            if comment:
                _save_comments(selected_ids, comment)
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            if tags:
                _save_tags(selected_ids, tags)
            if gmode == "Add to existing" and existing:
                _add_to_group(selected_ids, group_id=existing)
            elif gmode == "Create new" and new_group:
                _add_to_group(selected_ids, new_group_name=new_group, notes=notes)
            st.success("Saved")
            st.cache_data.clear()
            st.rerun()

# ---- Groups ----
with tab_groups:
    st.subheader("Groups")
    groups = _get_groups()
    if not groups:
        st.info("No groups yet. Create one from the Trades tab.")
    else:
        gmap = {f"{g['name']} ({g['id'][:6]})": g["id"] for g in groups}
        choice = st.selectbox("Choose a group", list(gmap.keys()))
        gid = gmap[choice]
        mem = sb.table("tj_trade_group_members").select("trade_id, tj_trades(*)").eq("group_id", gid).execute().data
        rows = [m["tj_trades"] for m in mem if m.get("tj_trades")]
        gdf = pd.DataFrame(rows or [])
        if gdf.empty:
            st.info("No trades in this group yet.")
        else:
            show_cols = ["external_trade_id","symbol","side","entry_ts_est","exit_ts_est","qty","pnl_net","r_multiple","review_status"]
            st.dataframe(gdf[show_cols], use_container_width=True)
            st.write(f"Group PnL (net): {gdf['pnl_net'].sum():,.2f}")

# ---- Guardrails ----
with tab_guards:
    st.subheader("Guardrails (quick checks)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        df = df.sort_values("entry_ts_est")
        st.metric("Win rate", f"{(df['pnl_net']>0).mean():.0%}")
        st.metric("Trades (7d)", str((df["entry_ts_est"] >= (pd.Timestamp.now() - pd.Timedelta(day

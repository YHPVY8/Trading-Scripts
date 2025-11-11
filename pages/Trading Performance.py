# pages/Trading Performance.py
#!/usr/bin/env python3
import io
import hashlib
from datetime import timedelta

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ===== CONFIG =====
# Your DB requires user_id NOT NULL -> enable scoping
USE_USER_SCOPING = True
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ---------- Utilities ----------
def _clean_header(name: str) -> str:
    if name is None:
        return ""
    s = str(name).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s.lower()

def _to_iso_est(ts_str: str):
    """Keep EST wall-clock: strip trailing timezone token like '-03:00' if present; return 'YYYY-MM-DD HH:MM:SS' or None."""
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
    x_str = str(x).replace(",", "").strip()
    try:
        return float(x_str)
    except Exception:
        v = pd.to_numeric(x_str, errors="coerce")
        return None if pd.isna(v) else float(v)

# ---------- CSV -> canonical rows ----------
def _read_csv_to_rows(uploaded_bytes: bytes):
    df = pd.read_csv(
        io.BytesIO(uploaded_bytes),
        sep=None,                 # auto-detect comma or tab
        engine="python",
        dtype=str,
        keep_default_na=False,    # keep empty cells as "", not NaN
    )
    original_headers = list(df.columns)
    norm_headers = [_clean_header(h) for h in original_headers]
    df.columns = norm_headers

    alias_map = {
        "id": "id", "trade id": "id", "tradeid": "id", "external id": "id",
        "contractname": "contractname", "contract": "contractname", "market": "contractname", "symbol": "contractname",
        "enteredat": "enteredat", "entry time": "enteredat", "entry": "enteredat",
        "exitedat": "exitedat", "exit time": "exitedat", "exit": "exitedat",
        "entryprice": "entryprice", "entry price": "entryprice",
        "exitprice": "exitprice", "exit price": "exitprice",
        "size": "size", "quantity": "size", "qty": "size",
        "type": "type", "side": "type",
        "pnl": "pnl", "p&l": "pnl", "profit": "pnl",
        "fees": "fees", "fee": "fees",
        "commissions": "commissions", "commission": "commissions",
        "tradeday": "tradeday", "tradeduration": "tradeduration",
    }

    canonical_cols = {}
    for src, canon in alias_map.items():
        if src in df.columns and canon not in canonical_cols:
            canonical_cols[canon] = src

    rows = []
    for _, r in df.iterrows():
        row = {}
        for canon, src in canonical_cols.items():
            row[canon] = r.get(src, "")
        rows.append(row)

    debug = {
        "original_headers": original_headers,
        "normalized_headers": norm_headers,
        "canonical_mapped_pairs": sorted([(k, v) for k, v in canonical_cols.items()]),
        "row_count": len(rows),
    }
    return rows, debug

# ---------- insert-or-update (manual upsert keyed by user_id + external_trade_id) ----------
def _insert_or_update_trade(payload: dict):
    q = sb.table("tj_trades").select("id", count="exact") \
        .eq("external_trade_id", payload["external_trade_id"]) \
        .eq("user_id", payload["user_id"]) \
        .limit(1)
    res = q.execute()
    existing = (res.data or [])
    if existing:
        tid = existing[0]["id"]
        sb.table("tj_trades").update(payload).eq("id", tid).execute()
        return "updated"
    else:
        sb.table("tj_trades").insert(payload).execute()
        return "inserted"

# ---------- Upsert loop ----------
def _upsert_trades_from_rows(rows):
    inserted = skipped = 0
    errs, samples = [], []

    for r in rows:
        try:
            ext_id = (r.get("id") or "").strip() or _synth_id(r)
            t = (r.get("type", "") or "").strip().lower()
            side = "long" if t in ("long", "buy", "b") else ("short" if t in ("short", "sell", "s") else None)

            qty_val = r.get("size")
            symbol = (r.get("contractname") or "").strip().upper() or None

            f1 = _as_float(r.get("fees")) or 0.0
            f2 = _as_float(r.get("commissions")) or 0.0
            fees_val = float(f1) + float(f2)

            entry_iso = _to_iso_est(r.get("enteredat"))
            exit_iso  = _to_iso_est(r.get("exitedat"))

            if not entry_iso:
                raise ValueError(f"Missing/Bad EnteredAt: {r.get('enteredat')}")

            payload = {
                "user_id": USER_ID,                 # <- REQUIRED by your DB
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": entry_iso,
                "exit_ts_est":  exit_iso,           # may be None for open trades
                "entry_px":  _as_float(r.get("entryprice")),
                "exit_px":   _as_float(r.get("exitprice")),
                "qty":       _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees":      fees_val,
                "source": "csv",
            }

            if len(samples) < 5:
                samples.append(dict(payload))

            status = _insert_or_update_trade(payload)
            if status in ("inserted", "updated"):
                inserted += 1
            else:
                skipped += 1

        except Exception as e:
            skipped += 1
            if len(errs) < 25:
                errs.append({
                    "id": r.get("id"),
                    "enteredat": r.get("enteredat"),
                    "exitedat": r.get("exitedat"),
                    "error": str(e),
                })

    return inserted, skipped, errs, samples

# ---------- Load ----------
@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    sel = sb.table("tj_trades").select(
        "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
        "pnl_gross,fees,pnl_net,planned_risk,r_multiple,review_status"
    ).eq("user_id", USER_ID)
    res = sel.order("entry_ts_est", desc=True).limit(limit).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"]  = pd.to_datetime(df["exit_ts_est"])
    return df

def _fetch_tags_for(trade_ids):
    if not trade_ids:
        return {}
    data = sb.table("tj_trade_tags").select("trade_id,tag").in_("trade_id", trade_ids).execute().data
    tagmap = {}
    for r in data or []:
        tagmap.setdefault(r["trade_id"], []).append(r["tag"])
    return tagmap

def _save_comments(trade_ids, body):
    if not body or not trade_ids:
        return
    rows = [{"trade_id": tid, "user_id": USER_ID, "body": body} for tid in trade_ids]
    sb.table("tj_trade_comments").insert(rows).execute()

def _save_tags(trade_ids, tags):
    if not tags or not trade_ids:
        return
    rows = [{"trade_id": tid, "user_id": USER_ID, "tag": t} for tid in trade_ids for t in tags]
    sb.table("tj_trade_tags").upsert(rows, on_conflict="trade_id,tag,user_id").execute()

def _add_to_group(trade_ids, group_id=None, new_group_name=None, notes=None):
    if new_group_name:
        g = sb.table("tj_trade_groups").insert({"user_id": USER_ID, "name": new_group_name, "notes": notes}).execute().data[0]
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

# Guard to avoid endless rerun loop after import
if "just_imported" not in st.session_state:
    st.session_state.just_imported = False

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    up = st.file_uploader("Drop your trade export", type=["csv"])
    if up and not st.session_state.just_imported:
        rows, dbg = _read_csv_to_rows(up.read())
        st.caption("Original headers: " + ", ".join(dbg["original_headers"]))
        st.caption("Normalized headers: " + ", ".join(dbg["normalized_headers"]))
        st.caption("Canonical mapped pairs: " + ", ".join([f"{pair[1]} -> {pair[0]}" for pair in dbg["canonical_mapped_pairs"]]))
        st.caption(f"Row count detected: {dbg['row_count']}")
        if rows:
            ins, skip, errs, samples = _upsert_trades_from_rows(rows)
            if ins > 0:
                st.success(f"Imported {ins} trades (skipped {skip}).")
            else:
                st.error(f"Imported {ins} trades (skipped {skip}).")
            if samples:
                with st.expander("Sample payloads attempted (first 5)"):
                    st.json(samples)
            if errs:
                with st.expander("Errors (first 25)"):
                    st.json(errs)
            st.cache_data.clear()
            # avoid infinite refresh churn — mark once, then rerun
            st.session_state.just_imported = True
            st.rerun()
        else:
            st.error("No rows found.")
    elif st.session_state.just_imported:
        st.info("Upload complete. Switch tabs or upload again.")
        # reset the flag if user clears/re-uploads
        if up is None:
            st.session_state.just_imported = False

# ---- Trades ----
with tab_trades:
    st.subheader("Trades")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        tagmap = _fetch_tags_for(df["id"].tolist())
        df["tags"] = df["id"].map(lambda i: ", ".join(sorted(tagmap.get(i, []))))

        rename_map = {
            "external_trade_id": "Trade ID",
            "entry_ts_est": "Entry (EST)",
            "exit_ts_est": "Exit (EST)",
            "pnl_net": "PnL (Net)",
        }

        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        df_display = df.rename(columns=rename_map)

        view_cols = [
            "selected", "Trade ID", "symbol", "side",
            "Entry (EST)", "Exit (EST)", "qty", "PnL (Net)",
            "r_multiple", "review_status", "tags"
        ]
        view_cols = [c for c in view_cols if c in df_display.columns]

        edited = st.data_editor(
            df_display[view_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("✓"),
                "PnL (Net)": st.column_config.NumberColumn("PnL (Net)", step=0.01, format="%.2f"),
                "r_multiple": st.column_config.NumberColumn("R Multiple", step=0.01),
                "review_status": st.column_config.SelectboxColumn("Review Status", options=["unreviewed", "flagged", "reviewed"]),
            },
            disabled=[c for c in view_cols if c not in ("selected", "r_multiple", "review_status")],
            num_rows="fixed",
        )

        edited_back = edited.rename(columns={v: k for k, v in rename_map.items()})
        edited_back = edited_back.merge(
            df[["external_trade_id", "id", "r_multiple", "review_status"]],
            on="external_trade_id",
            how="left",
            suffixes=("", "_old"),
        )

        # Save inline edits (r_multiple / review_status) safely
        diff_cols = ["r_multiple", "review_status"]
        to_update = []
        for _, r in edited_back.iterrows():
            rec = {}
            changed = False
            for c in diff_cols:
                new = r.get(c)
                old = r.get(f"{c}_old")
                if pd.isna(new) and pd.isna(old):
                    continue
                if (pd.isna(new) and not pd.isna(old)) or (not pd.isna(new) and pd.isna(old)) or (new != old):
                    rec[c] = new
                    changed = True
            if changed and r.get("id"):
                rec["id"] = r["id"]
                to_update.append(rec)

        if to_update:
            sb.table("tj_trades").upsert(to_update, on_conflict="id").execute()
            st.toast(f"Saved {len(to_update)} edits")
            st.cache_data.clear()

        st.markdown("---")
        selected_ids = edited_back.loc[edited_back.get("selected", False) == True, "id"].tolist()
        st.write(f"Selected: {len(selected_ids)}")

        with st.form("bulk_actions", clear_on_submit=True):
            comment = st.text_area("Add comment")
            tag_str = st.text_input("Add tags (comma-separated)")
            do = st.form_submit_button("Apply")
        if do and selected_ids:
            if comment:
                _save_comments(selected_ids, comment)
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            if tags:
                _save_tags(selected_ids, tags)
            st.success("Saved")
            st.cache_data.clear()
            st.rerun()

# ---- Groups ----
with tab_groups:
    st.subheader("Groups")
    groups = _get_groups()
    if not groups:
        st.info("No groups yet.")
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
        st.metric("Win rate", f"{(df['pnl_net'] > 0).mean():.0%}")
        st.metric("Trades (7d)", str((df["entry_ts_est"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))).sum()))
        st.metric("Net P&L (sum)", f"{df['pnl_net'].sum():,.2f}")

        hits = []
        ts = df["entry_ts_est"].tolist()
        i = j = 0
        while i < len(ts):
            while j < len(ts) and (ts[j] - ts[i]) <= timedelta(minutes=60):
                j += 1
            if j - i > 6:
                hits.append({"rule": "trades_per_60m_gt_6", "start": ts[i], "end": ts[j-1], "count": j - i})
            i += 1

        streak = 0
        start = None
        for _, r in df.iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                if streak == 0: start = r["entry_ts_est"]
                streak += 1
                if streak > 3:
                    hits.append({"rule": "consecutive_losses_gt_3", "start": start, "end": r["exit_ts_est"], "count": streak})
            else:
                streak = 0
                start = None

        last_loss_exit = None
        for _, r in df.sort_values("exit_ts_est").iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                last_loss_exit = r["exit_ts_est"]
            else:
                if last_loss_exit is not None and (r["entry_ts_est"] - last_loss_exit).total_seconds() <= 5 * 60:
                    hits.append({"rule": "reentry_under_5m_after_loss", "loss_exit": last_loss_exit, "reentry": r["entry_ts_est"]})
                last_loss_exit = None

        if hits:
            st.dataframe(pd.DataFrame(hits), use_container_width=True)
        else:
            st.success("No guardrail hits.")

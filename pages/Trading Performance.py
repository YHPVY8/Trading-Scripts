# pages/Trading Stats.py
#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import io, csv

st.set_page_config(page_title="Trading Stats (EST)", layout="wide")

# ---- Supabase client ----
sb = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ---------- DB helpers ----------
def _upsert_trades_from_rows(rows):
    inserted = skipped = 0
    for r in rows:
        try:
            ext_id = r.get("Id")
            if not ext_id:
                skipped += 1
                continue

            t = (r.get("Type","") or "").strip().lower()
            side = "long" if t.startswith("b") else ("short" if t.startswith("s") else None)

            payload = {
                "user_id": USER_ID,
                "external_trade_id": ext_id,
                "symbol": (r.get("Symbol") or r.get("Market") or "").upper() or None,
                "side": side,
                # Keep EST clock-time as-is (naive timestamps)
                "entry_ts_est": pd.to_datetime(r.get("EnteredAt")) if r.get("EnteredAt") else None,
                "exit_ts_est":  pd.to_datetime(r.get("ExitedAt"))  if r.get("ExitedAt")  else None,
                "entry_px": float(r["EntryPrice"]) if r.get("EntryPrice") not in (None,"") else None,
                "exit_px":  float(r["ExitPrice"])  if r.get("ExitPrice")  not in (None,"") else None,
                "qty":      float(r["Quantity"])   if r.get("Quantity")   not in (None,"") else None,
                "pnl_gross":float(r["PnL"])        if r.get("PnL")        not in (None,"") else None,
                "fees":     float(r["Fees"])       if r.get("Fees")       not in (None,"") else 0.0,
                "source": "csv",
            }
            sb.table("tj_trades").upsert(payload, on_conflict="user_id,external_trade_id").execute()
            inserted += 1
        except Exception:
            skipped += 1
    return inserted, skipped

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
    if not body or not trade_ids: return
    rows = [{"trade_id": tid, "user_id": USER_ID, "body": body} for tid in trade_ids]
    sb.table("tj_trade_comments").insert(rows).execute()

def _save_tags(trade_ids, tags):
    if not tags or not trade_ids: return
    rows = [{"trade_id": tid, "user_id": USER_ID, "tag": t} for tid in trade_ids for t in tags]
    sb.table("tj_trade_tags").upsert(rows, on_conflict="trade_id,tag").execute()

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
st.title("Trading Stats (EST)")

tab_upload, tab_trades, tab_groups, tab_guards = st.tabs(["Upload", "Trades", "Groups", "Guardrails"])

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    up = st.file_uploader("Drop your trade export (Id, Type, EnteredAt, ExitedAt, EntryPrice, ExitPrice, Quantity, PnL, Fees, ...)", type=["csv"])
    if up:
        content = up.read().decode("utf-8", errors="ignore")
        rows = list(csv.DictReader(io.StringIO(content)))
        ins, skip = _upsert_trades_from_rows(rows)
        st.success(f"Imported {ins} rows (skipped {skip}).")
        st.cache_data.clear()
        st.rerun()

# ---- Trades ----
with tab_trades:
    st.subheader("Trades")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet — upload a CSV in the Upload tab.")
    else:
        # Editable fields
        editable = ["planned_risk", "r_multiple", "review_status"]

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_selection("multiple", use_checkbox=True)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=50)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        for c in editable:
            gb.configure_column(c, editable=True)
        gb.configure_column("entry_ts_est", headerName="Entry (EST)")
        gb.configure_column("exit_ts_est", headerName="Exit (EST)")
        grid = AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.MODEL_CHANGED | GridUpdateMode.SELECTION_CHANGED,
            height=520,
        )

        # Persist inline edits for planned_risk / r_multiple / review_status
        if "data" in grid and grid["data"] is not None:
            edited = pd.DataFrame(grid["data"])
            if not edited[["id"]+editable].equals(df[["id"]+editable]):
                rows = edited[["id"]+editable].to_dict(orient="records")
                sb.table("tj_trades").upsert(rows, on_conflict="id").execute()
                st.toast("Saved inline edits")
                st.cache_data.clear()

        sel = grid.get("selected_rows", [])
        sel_ids = [r["id"] for r in sel]

        with st.form("bulk_actions", clear_on_submit=True):
            st.write(f"Selected: {len(sel_ids)}")
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
                    idx = st.selectbox("Pick group", list(range(len(labels))), format_func=lambda i: labels[i] if labels else None)
                    if groups:
                        existing = groups[idx]["id"]
                else:
                    st.info("No groups yet — choose 'Create new'.")
            elif gmode == "Create new":
                new_group = st.text_input("New group name")
                notes = st.text_input("Group notes (optional)")
            do = st.form_submit_button("Apply")

        if do and sel_ids:
            if comment:
                _save_comments(sel_ids, comment)
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            if tags:
                _save_tags(sel_ids, tags)
            if gmode == "Add to existing" and existing:
                _add_to_group(sel_ids, group_id=existing)
            elif gmode == "Create new" and new_group:
                _add_to_group(sel_ids, new_group_name=new_group, notes=notes)
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
        # join: members -> trades
        mem = sb.table("tj_trade_group_members").select("trade_id, tj_trades(*)").eq("group_id", gid).execute().data
        rows = [m["tj_trades"] for m in mem if m.get("tj_trades")]
        gdf = pd.DataFrame(rows or [])
        if gdf.empty:
            st.info("No trades in this group yet.")
        else:
            show_cols = ["external_trade_id","symbol","side","entry_ts_est","exit_ts_est","qty","pnl_net","r_multiple","review_status"]
            st.dataframe(gdf[show_cols])
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
        st.metric("Trades (7d)", str((df["entry_ts_est"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))).sum()))
        st.metric("Net P&L (sum)", f"{df['pnl_net'].sum():,.2f}")

        # Simple detectors
        from datetime import timedelta
        hits = []

        # 1) Max trades per 60 min > 6
        ts = df["entry_ts_est"].tolist()
        i = j = 0
        while i < len(ts):
            while j < len(ts) and (ts[j] - ts[i]) <= timedelta(minutes=60):
                j += 1
            count = j - i
            if count > 6:
                hits.append({"rule":"trades_per_60m>6","start":ts[i],"end":ts[j-1],"count":count})
            i += 1

        # 2) Max consecutive losses > 3
        streak = 0; start = None
        for _, r in df.iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                if streak == 0: start = r["entry_ts_est"]
                streak += 1
                if streak > 3:
                    hits.append({"rule":"consecutive_losses>3","start":start,"end":r["exit_ts_est"],"count":streak})
            else:
                streak = 0; start = None

        # 3) Re-entry under 5 minutes after a loss
        last_loss_exit = None
        for _, r in df.sort_values("exit_ts_est").iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                last_loss_exit = r["exit_ts_est"]
            else:
                if last_loss_exit is not None and (r["entry_ts_est"] - last_loss_exit).total_seconds() <= 5*60:
                    hits.append({"rule":"reentry_under_5m_after_loss","loss_exit":last_loss_exit,"reentry":r["entry_ts_est"]})
                last_loss_exit = None

        if hits:
            st.dataframe(pd.DataFrame(hits))
        else:
            st.success("No guardrail hits.")

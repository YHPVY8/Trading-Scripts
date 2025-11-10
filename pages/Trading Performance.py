# pages/Trading Performance.py
#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from supabase import create_client
import io, csv

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ---- Supabase client (your pattern) ----
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ---------- DB helpers ----------
def _upsert_trades_from_rows(rows):
    """
    Upsert CSV rows into tj_trades using your actual headers:
    Id, ContractName, EnteredAt, ExitedAt, EntryPrice, ExitPrice, Size, Type, PnL, Fees, Commissions
    Keeps EST/clock time by stripping the trailing timezone and formatting in ISO for PostgREST.
    Shows a few errors if any rows fail.
    """
    def _to_iso(ts_str: str):
        if not ts_str:
            return None
        s = str(ts_str).strip()
        # Strip trailing " +/-HH:MM" (e.g., '11/04/2025 15:00:17 -03:00' -> '11/04/2025 15:00:17')
        parts = s.rsplit(" ", 1)
        if len(parts) == 2 and len(parts[1]) == 6 and parts[1][3] == ":" and (parts[1][0] in "+-"):
            s = parts[0]
        # Parse and reformat to ISO 'YYYY-MM-DD HH:MM:SS'
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    inserted = 0
    skipped = 0
    errs = []  # collect first few errors to display

    for r in rows:
        try:
            ext_id = r.get("Id")
            if not ext_id:
                skipped += 1
                continue

            # Side mapping supports Long/Short and Buy/Sell
            t = (r.get("Type", "") or "").strip().lower()
            side = "long" if t in ("long", "buy", "b") else ("short" if t in ("short", "sell", "s") else None)

            # Quantity = Size
            qty_val = r.get("Size") or r.get("Quantity") or r.get("Qty")

            # Symbol = ContractName
            contract = (r.get("ContractName") or r.get("Symbol") or "").strip().upper()
            symbol = contract or None

            # Fees = Fees + Commissions
            f1 = pd.to_numeric(r.get("Fees"), errors="coerce")
            f2 = pd.to_numeric(r.get("Commissions"), errors="coerce")
            fees_val = (0 if pd.isna(f1) else float(f1)) + (0 if pd.isna(f2) else float(f2))

            payload = {
                "user_id": USER_ID,
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": _to_iso(r.get("EnteredAt")),
                "exit_ts_est":  _to_iso(r.get("ExitedAt")),
                "entry_px":  float(r["EntryPrice"]) if r.get("EntryPrice")  not in (None, "") else None,
                "exit_px":   float(r["ExitPrice"])  if r.get("ExitPrice")   not in (None, "") else None,
                "qty":       float(qty_val)         if qty_val               not in (None, "") else None,
                "pnl_gross": float(r["PnL"])        if r.get("PnL")         not in (None, "") else None,
                "fees":      fees_val,
                "source": "csv",
            }

            # sanity: require minimally entry_ts_est and exit_ts_est
            if not payload["entry_ts_est"] or not payload["exit_ts_est"]:
                raise ValueError(f"Bad timestamps for Id={ext_id}: {r.get('EnteredAt')} / {r.get('ExitedAt')}")

            sb.table("tj_trades").upsert(payload, on_conflict="user_id,external_trade_id").execute()
            inserted += 1

        except Exception as e:
            skipped += 1
            if len(errs) < 5:
                errs.append({"Id": r.get("Id"), "error": str(e)})

    return inserted, skipped, errs



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
st.title("Trading Performance (EST)")

tab_upload, tab_trades, tab_groups, tab_guards = st.tabs(["Upload", "Trades", "Groups", "Guardrails"])

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    up = st.file_uploader(
        "Drop your trade export (Id, ContractName, EnteredAt, ExitedAt, EntryPrice, ExitPrice, Size, Type, PnL, Fees, Commissions, ...)",
        type=["csv"]
    )
    if up:
        content = up.read().decode("utf-8", errors="ignore")
        rows = list(csv.DictReader(io.StringIO(content)))
        ins, skip, errs = _upsert_trades_from_rows(rows)
        st.success(f"Imported {ins} rows (skipped {skip}).")
        if errs:
            st.error("Sample errors:")
            st.json(errs)
        st.cache_data.clear()
        st.rerun()


# ---- Trades ----
with tab_trades:
    st.subheader("Trades")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet — upload a CSV in the Upload tab.")
    else:
        orig = df.copy()

        # Selection column for bulk actions
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        # Pretty labels for timestamps
        rename_map = {"entry_ts_est": "Entry (EST)", "exit_ts_est": "Exit (EST)"}
        df_display = df.rename(columns=rename_map)

        # Editable columns in the grid
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

        # Back to original column names
        edited = edited.rename(columns={v: k for k, v in rename_map.items()})

        # ---------- Persist inline edits (planned_risk / r_multiple / review_status) ----------
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

        # ---------- Bulk actions (comments, tags, groups) ----------
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
        # join: members -> trades
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
            st.dataframe(pd.DataFrame(hits), use_container_width=True)
        else:
            st.success("No guardrail hits.")

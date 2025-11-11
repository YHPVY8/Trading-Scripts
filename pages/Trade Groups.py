# pages/Trade Groups.py
#!/usr/bin/env python3
import pandas as pd
import streamlit as st
from supabase import create_client
from datetime import timedelta

st.set_page_config(page_title="Trade Groups (EST)", layout="wide")

# ===== Supabase client (same pattern you use) =====
USE_USER_SCOPING = True  # your tj_trades.user_id is NOT NULL
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ===== Data helpers =====
@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    res = sb.table("tj_trades").select(
        "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
        "pnl_gross,fees,pnl_net,r_multiple,review_status"
    ).eq("user_id", USER_ID).order("entry_ts_est", desc=True).limit(limit).execute()
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"]  = pd.to_datetime(df["exit_ts_est"])
    return df

def _get_groups():
    res = sb.table("tj_trade_groups").select("id,name,notes,created_at").eq("user_id", USER_ID)\
        .order("created_at", desc=True).execute()
    return res.data or []

def _add_to_group(trade_ids, group_id=None, new_group_name=None, notes=None):
    """Create a group (if name provided), then attach selected trades."""
    if new_group_name:
        g = sb.table("tj_trade_groups").insert({"user_id": USER_ID, "name": new_group_name, "notes": notes}).execute().data[0]
        group_id = g["id"]
    if group_id and trade_ids:
        rows = [{"group_id": group_id, "trade_id": tid} for tid in trade_ids]
        # upsert avoids dup membership
        sb.table("tj_trade_group_members").upsert(rows, on_conflict="group_id,trade_id").execute()
    return group_id

def _set_group_notes(group_id, notes):
    sb.table("tj_trade_groups").update({"notes": notes}).eq("id", group_id).execute()

def _fetch_group_members(gid):
    mem = sb.table("tj_trade_group_members").select("trade_id, tj_trades(*)").eq("group_id", gid).execute().data
    rows = [m["tj_trades"] for m in mem if m.get("tj_trades")]
    return pd.DataFrame(rows or [])

# ===== Rollup metrics =====
def _rollup_group_metrics(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {}
    legs = len(trades_df)
    qty = trades_df["qty"].fillna(0).astype(float)
    entry_px = trades_df.get("entry_px")
    exit_px  = trades_df.get("exit_px")
    entry_px = entry_px.astype(float) if entry_px is not None else pd.Series([None]*legs)
    exit_px  = exit_px.astype(float)  if exit_px  is not None else pd.Series([None]*legs)

    w = qty.abs()
    # safe denominators for VWAP (ignore NaN prices)
    w_entry = w.where(entry_px.notna(), 0)
    w_exit  = w.where(exit_px.notna(), 0)
    vwap_entry = (entry_px.where(entry_px.notna(), 0) * w).sum() / (w_entry.sum() if w_entry.sum() else float("nan"))
    vwap_exit  = (exit_px.where(exit_px.notna(), 0)  * w).sum() / (w_exit.sum()  if w_exit.sum()  else float("nan"))
    vwap_entry = None if pd.isna(vwap_entry) else float(vwap_entry)
    vwap_exit  = None if pd.isna(vwap_exit)  else float(vwap_exit)

    total_qty = float(qty.sum())
    if "pnl_net" in trades_df and trades_df["pnl_net"].notna().any():
        net = trades_df["pnl_net"].fillna(0).sum()
    else:
        net = trades_df["pnl_gross"].fillna(0).sum() - trades_df["fees"].fillna(0).sum()

    start_ts = trades_df["entry_ts_est"].min()
    end_ts   = trades_df["exit_ts_est"].max() if "exit_ts_est" in trades_df else None
    symbol   = trades_df["symbol"].dropna().mode().iloc[0] if trades_df["symbol"].notna().any() else ""
    side     = trades_df["side"].dropna().mode().iloc[0] if trades_df["side"].notna().any() else ""

    return {
        "legs": int(legs),
        "total_qty": total_qty,
        "vwap_entry": vwap_entry,
        "vwap_exit":  vwap_exit,
        "pnl_net_sum": float(net),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "symbol": symbol,
        "side": side,
    }

def _autoname_from_metrics(m):
    if not m:
        return "Group"
    d = pd.to_datetime(m["start_ts"]).strftime("%Y-%m-%d") if pd.notna(m["start_ts"]) else ""
    when = (pd.to_datetime(m["end_ts"]).strftime("%H:%M") if pd.notna(m["end_ts"])
            else (pd.to_datetime(m["start_ts"]).strftime("%H:%M") if pd.notna(m["start_ts"]) else ""))
    sym = m.get("symbol") or ""
    side = m.get("side") or ""
    legs = m.get("legs", 0)
    return f"{sym} {side} {d} {when} ({legs} legs)".strip()

def _notes_from_metrics(m, extra=None):
    parts = [f"Legs: {m['legs']}", f"Total Qty: {m['total_qty']:.0f}"]
    if m["vwap_entry"] is not None: parts.append(f"VWAP Entry: {m['vwap_entry']:.2f}")
    if m["vwap_exit"]  is not None: parts.append(f"VWAP Exit: {m['vwap_exit']:.2f}")
    parts.append(f"Net PnL: {m['pnl_net_sum']:.2f}")
    if pd.notna(m["start_ts"]): parts.append(f"Start: {pd.to_datetime(m['start_ts'])}")
    if pd.notna(m["end_ts"]):   parts.append(f"End: {pd.to_datetime(m['end_ts'])}")
    if extra: parts.append(extra)
    return " | ".join(parts)

# ===== UI =====
st.title("Trade Groups (EST)")

tab_make, tab_browse = st.tabs(["Make / Edit Groups", "Browse Groups"])

# ---- Make / Edit Groups ----
with tab_make:
    st.subheader("Select trades to group")
    df = _load_trades()
    if df.empty:
        st.info("No trades available. Upload in Trading Performance → Upload.")
    else:
        # Add selectable checkbox
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        # Display-friendly labels
        rename_map = {
            "external_trade_id": "Trade ID",
            "entry_ts_est": "Entry (EST)",
            "exit_ts_est": "Exit (EST)",
            "pnl_net": "PnL (Net)",
        }
        df_display = df.rename(columns=rename_map)

        view_cols = [
            "selected", "Trade ID", "symbol", "side",
            "Entry (EST)", "Exit (EST)", "qty", "PnL (Net)",
            "r_multiple", "review_status"
        ]
        view_cols = [c for c in view_cols if c in df_display.columns]

        edited = st.data_editor(
            df_display[view_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("✓"),
                "PnL (Net)": st.column_config.NumberColumn("PnL (Net)", step=0.01, format="%.2f"),
            },
            disabled=[c for c in view_cols if c not in ("selected",)],
            num_rows="fixed",
        )

        # Map back to original names and bring internal id for persistence
        back = edited.rename(columns={v: k for k, v in rename_map.items()})
        back = back.merge(
            df[["external_trade_id", "id", "symbol", "side", "entry_ts_est", "exit_ts_est",
                "entry_px", "exit_px", "qty", "pnl_gross", "fees", "pnl_net"]],
            on="external_trade_id",
            how="left"
        )

        selected_ids = back.loc[back.get("selected", False) == True, "id"].tolist()
        st.write(f"Selected: {len(selected_ids)}")

        st.markdown("#### Group action")
        left, right = st.columns([1,2])
        with left:
            gmode = st.radio("Action", ["Create new", "Add to existing"], horizontal=False)
        with right:
            extra_notes = st.text_input("Notes (optional)")

        if gmode == "Add to existing":
            groups = _get_groups()
            if groups:
                labels = [f"{g['name']} ({g['id'][:6]})" for g in groups]
                idx = st.selectbox("Pick group", list(range(len(labels))), format_func=lambda i: labels[i])
                existing_gid = groups[idx]["id"]
            else:
                st.info("No groups yet — switch to 'Create new'.")
                existing_gid = None
        else:
            auto_name = st.checkbox("Auto-name from selection", value=True)
            new_group_name = st.text_input("New group name", value="")

        go = st.button("Apply")

        if go:
            if not selected_ids:
                st.warning("Select at least one trade.")
            else:
                sel_df = back.loc[back["id"].isin(selected_ids)]
                metrics = _rollup_group_metrics(sel_df)
                summary = _notes_from_metrics(metrics, extra=extra_notes if extra_notes else None)
                if gmode == "Add to existing":
                    if existing_gid:
                        _add_to_group(selected_ids, group_id=existing_gid)
                        # Optionally update/append notes (here we overwrite to keep latest summary)
                        _set_group_notes(existing_gid, summary)
                        st.success("Added to existing group and updated notes.")
                        st.cache_data.clear()
                else:
                    name_to_use = _autoname_from_metrics(metrics) if (auto_name or not new_group_name.strip()) else new_group_name.strip()
                    gid = _add_to_group(selected_ids, new_group_name=name_to_use, notes=summary)
                    st.success(f"Created group “{name_to_use}”.")
                    st.cache_data.clear()

# ---- Browse Groups ----
with tab_browse:
    st.subheader("Groups")
    groups = _get_groups()
    if not groups:
        st.info("No groups yet.")
    else:
        left, right = st.columns([1,2], vertical_alignment="top")
        with left:
            label_to_id = {f"{g['name']} ({g['id'][:6]})": g["id"] for g in groups}
            choice = st.selectbox("Choose a group", list(label_to_id.keys()))
            gid = label_to_id[choice]
            current = next(g for g in groups if g["id"] == gid)
            with st.expander("Edit group meta"):
                new_name = st.text_input("Name", value=current["name"] or "")
                new_notes = st.text_area("Notes", value=current.get("notes") or "", height=120)
                if st.button("Save meta"):
                    sb.table("tj_trade_groups").update({"name": new_name, "notes": new_notes}).eq("id", gid).execute()
                    st.success("Saved")
                    st.cache_data.clear()
        with right:
            gdf = _fetch_group_members(gid)
            if gdf.empty:
                st.info("No trades in this group.")
            else:
                # Rollup
                roll = _rollup_group_metrics(gdf)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Legs", f"{roll['legs']}")
                c2.metric("Total Qty", f"{roll['total_qty']:.0f}")
                c3.metric("VWAP Entry", "-" if roll['vwap_entry'] is None else f"{roll['vwap_entry']:.2f}")
                c4.metric("VWAP Exit", "-" if roll['vwap_exit'] is None else f"{roll['vwap_exit']:.2f}")
                st.metric("Net PnL (group)", f"{roll['pnl_net_sum']:.2f}")

                show_cols = ["external_trade_id","symbol","side","entry_ts_est","exit_ts_est","qty",
                             "entry_px","exit_px","pnl_net","r_multiple","review_status"]
                show_cols = [c for c in show_cols if c in gdf.columns]
                st.dataframe(gdf[show_cols].sort_values("entry_ts_est"), use_container_width=True)

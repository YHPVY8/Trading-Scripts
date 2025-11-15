#!/usr/bin/env python3
# pages/01_Trading_Performance.py

from datetime import timedelta
import re

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ===== CONFIG =====
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = st.secrets.get("USER_ID", "00000000-0000-0000-0000-000000000001")

# ===== Session state defaults (shared with Upload page) =====
if "just_imported" not in st.session_state:
    st.session_state.just_imported = False
if "last_imported_external_ids" not in st.session_state:
    st.session_state.last_imported_external_ids = []
if "auto_select_after_upload" not in st.session_state:
    st.session_state.auto_select_after_upload = False

# ---------- Load ----------
@st.cache_data(ttl=60)
def _load_trades(limit=4000):
    res = (
        sb.table("tj_trades")
        .select(
            "id,external_trade_id,symbol,side,entry_ts_est,exit_ts_est,entry_px,exit_px,qty,"
            "pnl_gross,fees,pnl_net,planned_risk,r_multiple,review_status"
        )
        .eq("user_id", USER_ID)
        .order("entry_ts_est", desc=True)
        .limit(limit)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if not df.empty:
        df["entry_ts_est"] = pd.to_datetime(df["entry_ts_est"])
        df["exit_ts_est"] = pd.to_datetime(df["exit_ts_est"])
    return df


def _fetch_tags_for(trade_ids):
    if not trade_ids:
        return {}
    data = (
        sb.table("tj_trade_tags")
        .select("trade_id,tag")
        .in_("trade_id", trade_ids)
        .eq("user_id", USER_ID)
        .execute()
        .data
    )
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
    """Robust tag save:
       1) Try upsert on (trade_id,tag,user_id)
       2) Fallback to (trade_id,tag)
       3) Fallback to manual dedupe: fetch existing then insert only missing
    """
    if not tags or not trade_ids:
        return

    rows = [
        {"trade_id": tid, "user_id": USER_ID, "tag": t}
        for tid in trade_ids
        for t in tags
    ]
    # Attempt #1
    try:
        sb.table("tj_trade_tags").upsert(
            rows, on_conflict="trade_id,tag,user_id"
        ).execute()
        st.toast("Tags saved (conflict: trade_id,tag,user_id)")
        return
    except Exception:
        pass
    # Attempt #2
    try:
        sb.table("tj_trade_tags").upsert(rows, on_conflict="trade_id,tag").execute()
        st.toast("Tags saved (conflict: trade_id,tag)")
        return
    except Exception:
        pass
    # Attempt #3: manual dedupe
    try:
        existing = (
            sb.table("tj_trade_tags")
            .select("trade_id,tag")
            .in_("trade_id", trade_ids)
            .eq("user_id", USER_ID)
            .execute()
            .data
            or []
        )
        existing_set = {(e["trade_id"], e["tag"]) for e in existing}
        to_insert = [r for r in rows if (r["trade_id"], r["tag"]) not in existing_set]
        if to_insert:
            sb.table("tj_trade_tags").insert(to_insert).execute()
            st.toast(f"Tags saved (manual insert {len(to_insert)})")
        else:
            st.toast("Tags already present")
    except Exception as e:
        st.error(f"Saving tags failed: {e}")


def _add_to_group(trade_ids, group_id=None, new_group_name=None, notes=None):
    """Create a group if name provided; then attach trade_ids."""
    if new_group_name:
        g = (
            sb.table("tj_trade_groups")
            .insert({"user_id": USER_ID, "name": new_group_name, "notes": notes})
            .execute()
            .data[0]
        )
        group_id = g["id"]
    if group_id and trade_ids:
        rows = [{"group_id": group_id, "trade_id": tid} for tid in trade_ids]
        try:
            sb.table("tj_trade_group_members").upsert(
                rows, on_conflict="group_id,trade_id"
            ).execute()
        except Exception:
            (
                sb.table("tj_trade_group_members")
                .delete()
                .in_("trade_id", trade_ids)
                .eq("group_id", group_id)
                .execute()
            )
            sb.table("tj_trade_group_members").insert(rows).execute()
    return group_id


def _remove_from_groups(trade_ids):
    """Ungroup: remove selected trades from any groups they belong to."""
    if not trade_ids:
        return
    (
        sb.table("tj_trade_group_members")
        .delete()
        .in_("trade_id", trade_ids)
        .execute()
    )


def _get_groups():
    res = (
        sb.table("tj_trade_groups")
        .select("id,name,notes,created_at")
        .eq("user_id", USER_ID)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def _next_group_name():
    """Auto-increment group name like 'Group 1', 'Group 2', ..."""
    groups = _get_groups()
    nums = []
    for g in groups:
        name = (g.get("name") or "").strip()
        m = re.search(r"(\d+)$", name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    n = (max(nums) + 1) if nums else 1
    return f"Group {n}"


# ===== Group collapsed helpers =====
def _fetch_all_groups_with_members():
    """
    Returns (groups, members_df).
    groups: list of group rows {id,name,notes,created_at}
    members_df: DataFrame of joined group members with trade columns for this USER_ID
    """
    groups = (
        sb.table("tj_trade_groups")
        .select("id,name,notes,created_at")
        .eq("user_id", USER_ID)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )

    mem = (
        sb.table("tj_trade_group_members")
        .select("group_id, trade_id, tj_trades(*)")
        .execute()
        .data
        or []
    )

    rows = []
    for m in mem:
        t = m.get("tj_trades") or {}
        if t and t.get("user_id") == USER_ID:
            t["group_id"] = m["group_id"]
            rows.append(t)

    df = pd.DataFrame(rows or [])
    if not df.empty:
        for c in ("entry_ts_est", "exit_ts_est"):
            if c in df:
                df[c] = pd.to_datetime(df[c])
        for c in ("qty", "entry_px", "exit_px", "pnl_gross", "fees", "pnl_net"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return groups, df


def _rollup_by_group(members_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse trades to one row per group_id with:
    group_id, first_entry, last_exit, legs, total_qty, vwap_entry, vwap_exit,
    symbol(mode), side(mode), pnl_net_sum
    """
    if members_df.empty:
        return pd.DataFrame()

    def _mode(series):
        s = series.dropna()
        return s.mode().iloc[0] if not s.empty else None

    out = []
    for gid, gdf in members_df.groupby("group_id"):
        legs = len(gdf)
        qty = gdf["qty"].fillna(0).abs()

        # VWAP Entry
        ok_e = gdf["entry_px"].notna()
        w_entry = qty.where(ok_e, 0)
        denom_e = w_entry.sum()
        vwap_entry = None
        if denom_e and denom_e != 0:
            vwap_entry = float(
                (gdf["entry_px"].where(ok_e, 0) * qty).sum() / denom_e
            )

        # VWAP Exit
        ok_x = gdf["exit_px"].notna()
        w_exit = qty.where(ok_x, 0)
        denom_x = w_exit.sum()
        vwap_exit = None
        if denom_x and denom_x != 0:
            vwap_exit = float(
                (gdf["exit_px"].where(ok_x, 0) * qty).sum() / denom_x
            )

        first_entry = gdf["entry_ts_est"].min()
        last_exit = gdf["exit_ts_est"].max()
        total_qty = float(gdf["qty"].fillna(0).sum())

        # Net PnL
        pnl_sum = float(
            (gdf["pnl_net"].fillna(0) if "pnl_net" in gdf else 0).sum()
        )
        if pnl_sum == 0 and ("pnl_gross" in gdf or "fees" in gdf):
            pnl_sum = float(
                gdf["pnl_gross"].fillna(0).sum()
                - gdf["fees"].fillna(0).sum()
            )

        out.append(
            {
                "group_id": gid,
                "first_entry": first_entry,
                "last_exit": last_exit,
                "legs": int(legs),
                "total_qty": total_qty,
                "vwap_entry": vwap_entry,
                "vwap_exit": vwap_exit,
                "symbol": _mode(gdf["symbol"]) if "symbol" in gdf else None,
                "side": _mode(gdf["side"]) if "side" in gdf else None,
                "pnl_net_sum": pnl_sum,
            }
        )
    return pd.DataFrame(out)


# ---------- UI ----------
st.title("Trading Performance (EST)")
tab_trades, tab_groups, tab_guards = st.tabs(["Trades", "Groups", "Guardrails"])

# ---- Trades ----
with tab_trades:
    st.subheader("Trades (legs)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        # Toggle whether to auto-select newly uploaded rows
        st.checkbox("Auto-select newly imported trades", key="auto_select_after_upload")

        # Attach tags
        tagmap = _fetch_tags_for(df["id"].tolist())
        df["tags"] = df["id"].map(lambda i: ", ".join(sorted(tagmap.get(i, []))))

        # Display-friendly names
        rename_map = {
            "external_trade_id": "Trade ID",
            "entry_ts_est": "Entry (EST)",
            "exit_ts_est": "Exit (EST)",
            "pnl_net": "PnL (Net)",
        }

        # Insert selection column
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        # Only preselect after upload if toggle is ON
        if (
            st.session_state.auto_select_after_upload
            and st.session_state.last_imported_external_ids
        ):
            df.loc[
                df["external_trade_id"].isin(
                    st.session_state.last_imported_external_ids
                ),
                "selected",
            ] = True

        df_display = df.rename(columns=rename_map)

        view_cols = [
            "selected",
            "Trade ID",
            "symbol",
            "side",
            "Entry (EST)",
            "Exit (EST)",
            "qty",
            "pnl_gross",
            "fees",
            "PnL (Net)",
            "r_multiple",
            "review_status",
            "tags",
        ]
        view_cols = [c for c in view_cols if c in df_display.columns]

        edited = st.data_editor(
            df_display[view_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("✓"),
                "pnl_gross": st.column_config.NumberColumn(
                    "PnL (Gross)", step=0.01, format="%.2f"
                ),
                "fees": st.column_config.NumberColumn(
                    "Fees", step=0.01, format="%.2f"
                ),
                "PnL (Net)": st.column_config.NumberColumn(
                    "PnL (Net)", step=0.01, format="%.2f"
                ),
                "r_multiple": st.column_config.NumberColumn(
                    "R Multiple", step=0.01
                ),
                "review_status": st.column_config.SelectboxColumn(
                    "Review Status",
                    options=["unreviewed", "flagged", "reviewed"],
                ),
                "tags": st.column_config.TextColumn("Tags (read-only)"),
            },
            disabled=[c for c in view_cols if c not in ("selected", "r_multiple", "review_status")],
            num_rows="fixed",
        )

        # Map back to original names and bring internal id for persistence
        edited_back = edited.rename(columns={v: k for k, v in rename_map.items()})
        edited_back = edited_back.merge(
            df[["external_trade_id", "id", "r_multiple", "review_status"]],
            on="external_trade_id",
            how="left",
            suffixes=("", "_old"),
        )

        # Persist inline edits (r_multiple / review_status)
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
                if (pd.isna(new) and not pd.isna(old)) or (
                    not pd.isna(new) and pd.isna(old)
                ) or (new != old):
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
        # Bulk actions: comments, tags, grouping, ungroup
        selected_ids = edited_back.loc[
            edited_back.get("selected", False) == True, "id"
        ].tolist()
        st.write(f"Selected: {len(selected_ids)}")

        with st.form("bulk_actions", clear_on_submit=True):
            comment = st.text_area("Add comment (markdown)")
            tag_str = st.text_input("Add tags (comma-separated)")

            st.markdown("**Grouping**")
            gmode = st.radio(
                "Action",
                ["None", "Add to existing", "Create new (auto-name)", "Remove from group(s)"],
                horizontal=False,
            )

            existing = None
            new_group_name = None
            notes = st.text_input("Group notes (optional)")

            if gmode == "Add to existing":
                groups = _get_groups()
                if groups:
                    labels = [
                        f"{g['name']} ({g['id'][:6]})" for g in groups
                    ]
                    idx = st.selectbox(
                        "Pick group",
                        list(range(len(labels))) if labels else [],
                        format_func=lambda i: labels[i] if labels else None,
                    )
                    if groups and len(groups) > 0:
                        existing = groups[idx]["id"]
                else:
                    st.info("No groups yet — choose 'Create new (auto-name)'.")
            elif gmode == "Create new (auto-name)":
                suggested = _next_group_name()
                st.caption(f"Suggested name: **{suggested}**")
                new_group_name = suggested  # use suggested automatically

            do = st.form_submit_button("Apply")

        if do and selected_ids:
            if comment:
                _save_comments(selected_ids, comment)
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            if tags:
                _save_tags(selected_ids, tags)

            if gmode == "Add to existing" and existing:
                _add_to_group(
                    selected_ids,
                    group_id=existing,
                    notes=notes if notes else None,
                )
            elif gmode == "Create new (auto-name)":
                _add_to_group(
                    selected_ids,
                    new_group_name=new_group_name,
                    notes=notes if notes else None,
                )
            elif gmode == "Remove from group(s)":
                _remove_from_groups(selected_ids)

            st.success("Saved")
            st.cache_data.clear()
            st.session_state.last_imported_external_ids = []
            st.rerun()

# ---- Groups (collapsed + details, with hashtag filter) ----
with tab_groups:
    st.subheader("Groups (collapsed positions)")

    groups, mem_df = _fetch_all_groups_with_members()
    if not groups:
        st.info("No groups yet. Create them from the Trades tab after importing.")
    else:
        # COLLAPSED TABLE (one line per group)
        roll = _rollup_by_group(mem_df)

        # join names/notes
        name_map = {g["id"]: g["name"] for g in groups}
        notes_map = {g["id"]: g.get("notes") for g in groups}
        if not roll.empty:
            roll["name"] = roll["group_id"].map(name_map)
            roll["notes"] = roll["group_id"].map(notes_map)

            # Filters
            c1, c2, c3 = st.columns([1, 2, 3])
            with c1:
                day = st.date_input("Filter by day", value=None)
            with c2:
                hashtag_str = st.text_input(
                    "Filter by #hashtags in notes (comma-separated)", value=""
                )
            with c3:
                sym = st.text_input("Symbol filter (optional)", value="").strip().upper()

            rshow = roll.copy()
            # day filter
            if day is not None:
                try:
                    rshow = rshow[rshow["first_entry"].dt.date == day]
                except Exception:
                    pass
            # hashtag filter (all hashtags must appear in notes)
            if hashtag_str.strip():
                tags = [
                    t.strip().lstrip("#").lower()
                    for t in hashtag_str.split(",")
                    if t.strip()
                ]

                def _has_all_hashtags(txt):
                    s = (txt or "").lower()
                    return all(("#" + t) in s for t in tags)

                rshow = rshow[rshow["notes"].apply(_has_all_hashtags)]
            # symbol filter
            if sym:
                rshow = rshow[(rshow["symbol"].fillna("") == sym)]

            show_cols = [
                "name",
                "symbol",
                "side",
                "first_entry",
                "last_exit",
                "legs",
                "total_qty",
                "vwap_entry",
                "vwap_exit",
                "pnl_net_sum",
                "notes",
            ]
            show_cols = [c for c in show_cols if c in rshow.columns]
            st.markdown("**Collapsed (1 row per group)**")
            st.dataframe(
                rshow[show_cols].sort_values(["first_entry"], ascending=[False]),
                use_container_width=False,
            )

            st.divider()

            # DETAILS for a selected group
            st.markdown("**Group details**")
            label_to_id = {
                f"{name_map.get(g['id'],'(unnamed)')} ({g['id'][:6]})": g["id"]
                for g in groups
            }
            choice = st.selectbox("Pick a group", list(label_to_id.keys()))
            gid = label_to_id[choice]

            gdf = mem_df[mem_df["group_id"] == gid].copy()
            if gdf.empty:
                st.info("No member trades in this group.")
            else:
                # summary metrics for the chosen group
                groll = _rollup_by_group(gdf.assign(group_id=gid))
                if not groll.empty:
                    row = groll.iloc[0]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Legs", f"{int(row['legs'])}")
                    m2.metric("Total Qty", f"{row['total_qty']:.0f}")
                    m3.metric(
                        "VWAP Entry",
                        "-"
                        if row["vwap_entry"] is None
                        else f"{row['vwap_entry']:.2f}",
                    )
                    m4.metric(
                        "VWAP Exit",
                        "-"
                        if row["vwap_exit"] is None
                        else f"{row['vwap_exit']:.2f}",
                    )
                    st.metric("Net PnL (group)", f"{row['pnl_net_sum']:.2f}")

                trade_cols = [
                    "external_trade_id",
                    "symbol",
                    "side",
                    "entry_ts_est",
                    "exit_ts_est",
                    "qty",
                    "entry_px",
                    "exit_px",
                    "pnl_net",
                    "r_multiple",
                    "review_status",
                ]
                trade_cols = [c for c in trade_cols if c in gdf.columns]
                st.dataframe(
                    gdf[trade_cols].sort_values("entry_ts_est"),
                    use_container_width=False,
                )
        else:
            st.info("You have groups, but no member trades yet.")

# ---- Guardrails ----
with tab_guards:
    st.subheader("Guardrails (quick checks)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        df = df.sort_values("entry_ts_est")
        df["pnl_net"] = pd.to_numeric(df["pnl_net"], errors="coerce").fillna(0.0)
        st.metric("Win rate", f"{(df['pnl_net'] > 0).mean():.0%}")
        st.metric(
            "Trades (7d)",
            str(
                (df["entry_ts_est"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))).sum()
            ),
        )
        st.metric("Net PnL (sum)", f"{df['pnl_net'].sum():,.2f}")

        hits = []
        ts = df["entry_ts_est"].tolist()
        i = j = 0
        while i < len(ts):
            while j < len(ts) and (ts[j] - ts[i]) <= timedelta(minutes=60):
                j += 1
            if j - i > 6:
                hits.append(
                    {
                        "rule": "trades_per_60m_gt_6",
                        "start": ts[i],
                        "end": ts[j - 1],
                        "count": j - i,
                    }
                )
            i += 1

        streak = 0
        start_loss = None
        for _, r in df.iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                if streak == 0:
                    start_loss = r["entry_ts_est"]
                streak += 1
                if streak > 3:
                    hits.append(
                        {
                            "rule": "consecutive_losses_gt_3",
                            "start": start_loss,
                            "end": r["exit_ts_est"],
                            "count": streak,
                        }
                    )
            else:
                streak = 0
                start_loss = None

        last_loss_exit = None
        for _, r in df.sort_values("exit_ts_est").iterrows():
            pnl = r["pnl_net"] if pd.notna(r["pnl_net"]) else 0
            if pnl < 0:
                last_loss_exit = r["exit_ts_est"]
            else:
                if last_loss_exit is not None and (
                    r["entry_ts_est"] - last_loss_exit
                ).total_seconds() <= 5 * 60:
                    hits.append(
                        {
                            "rule": "reentry_under_5m_after_loss",
                            "loss_exit": last_loss_exit,
                            "reentry": r["entry_ts_est"],
                        }
                    )
                last_loss_exit = None

        if hits:
            st.dataframe(pd.DataFrame(hits), use_container_width=False)
        else:
            st.success("No guardrail hits.")

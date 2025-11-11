# pages/Trading Performance.py
#!/usr/bin/env python3
import io
import re
import hashlib
from datetime import timedelta

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Trading Performance (EST)", layout="wide")

# ===== CONFIG =====
USE_USER_SCOPING = True  # your DB uses NOT NULL user_id
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
        return "updated", tid
    else:
        r = sb.table("tj_trades").insert(payload).execute()
        tid = (r.data or [{}])[0].get("id")
        return "inserted", tid

# ---------- Upsert loop ----------
def _upsert_trades_from_rows(rows):
    inserted = skipped = 0
    errs, samples = [], []
    inserted_external_ids = []
    inserted_trade_ids = []

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
                "user_id": USER_ID,                 # <- REQUIRED
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": entry_iso,
                "exit_ts_est":  exit_iso,           # may be None
                "entry_px":  _as_float(r.get("entryprice")),
                "exit_px":   _as_float(r.get("exitprice")),
                "qty":       _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees":      fees_val,
                "source": "csv",
            }

            if len(samples) < 5:
                samples.append(dict(payload))

            status, tid = _insert_or_update_trade(payload)
            if status in ("inserted", "updated"):
                inserted += 1
                inserted_external_ids.append(ext_id)
                if tid:
                    inserted_trade_ids.append(tid)
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

    return inserted, skipped, errs, samples, inserted_external_ids, inserted_trade_ids

# ---------- Load ----------
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

def _fetch_tags_for(trade_ids):
    if not trade_ids:
        return {}
    data = sb.table("tj_trade_tags").select("trade_id,tag").in_("trade_id", trade_ids).eq("user_id", USER_ID).execute().data
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

def _next_group_name():
    """Auto-name groups: 'Group N' with N = 1 + max existing numeric suffix for this user."""
    rows = sb.table("tj_trade_groups").select("name").eq("user_id", USER_ID).execute().data or []
    max_n = 0
    for r in rows:
        name = (r.get("name") or "").strip()
        m = re.match(r"(?i)group\s+(\d+)$", name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except Exception:
                pass
    return f"Group {max_n + 1}"

def _add_to_group(trade_ids, group_id=None, new_group_name=None, notes=None):
    """Create a group if name is blank -> auto name; then attach trade_ids."""
    if new_group_name is not None and new_group_name.strip() == "":
        new_group_name = None
    if new_group_name is None and group_id is None:
        new_group_name = _next_group_name()

    if new_group_name:
        g = sb.table("tj_trade_groups").insert(
            {"user_id": USER_ID, "name": new_group_name, "notes": notes}
        ).execute().data[0]
        group_id = g["id"]

    if group_id and trade_ids:
        rows = [{"group_id": group_id, "trade_id": tid} for tid in trade_ids]
        sb.table("tj_trade_group_members").upsert(rows, on_conflict="group_id,trade_id").execute()
    return group_id

def _remove_from_group(trade_ids, group_id=None, remove_from_all=False):
    """Un-group trades: either from a specific group, or from all groups the trades belong to."""
    if not trade_ids:
        return
    if remove_from_all:
        # delete any membership rows for these trades (any group)
        sb.table("tj_trade_group_members").delete().in_("trade_id", trade_ids).execute()
    elif group_id:
        # delete memberships only for selected group
        # (no 'and_' API in supabase-py for composite keys; do two filters)
        # We need to fetch memberships and delete by primary key if available,
        # but many setups allow delete with both filters chained:
        sb.table("tj_trade_group_members").delete().eq("group_id", group_id).in_("trade_id", trade_ids).execute()

def _get_groups():
    res = sb.table("tj_trade_groups").select("id,name,notes,created_at").eq("user_id", USER_ID).order("created_at", desc=True).execute()
    return res.data or []

# ===== Group collapsed helpers =====
def _fetch_all_groups_with_members():
    """
    Returns (groups, members_df).
    groups: list {id,name,notes,created_at}
    members_df: DataFrame of joined group members with trade columns for this USER_ID
    """
    groups = sb.table("tj_trade_groups")\
        .select("id,name,notes,created_at")\
        .eq("user_id", USER_ID)\
        .order("created_at", desc=True)\
        .execute().data or []

    mem = sb.table("tj_trade_group_members")\
        .select("group_id, trade_id, tj_trades(*)")\
        .execute().data or []

    rows = []
    for m in mem or []:
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
        vwap_entry = (gdf["entry_px"].where(ok_e, 0) * qty).sum() / (w_entry.sum() if w_entry.sum() else float("nan"))
        vwap_entry = None if pd.isna(vwap_entry) else float(vwap_entry)

        # VWAP Exit
        ok_x = gdf["exit_px"].notna()
        w_exit = qty.where(ok_x, 0)
        vwap_exit = (gdf["exit_px"].where(ok_x, 0) * qty).sum() / (w_exit.sum() if w_exit.sum() else float("nan"))
        vwap_exit = None if pd.isna(vwap_exit) else float(vwap_exit)

        first_entry = gdf["entry_ts_est"].min()
        last_exit   = gdf["exit_ts_est"].max()
        total_qty   = gdf["qty"].fillna(0).sum()

        # Net PnL
        if "pnl_net" in gdf and gdf["pnl_net"].notna().any():
            pnl_sum = gdf["pnl_net"].fillna(0).sum()
        else:
            pnl_sum = gdf["pnl_gross"].fillna(0).sum() - gdf["fees"].fillna(0).sum()

        out.append({
            "group_id": gid,
            "first_entry": first_entry,
            "last_exit": last_exit,
            "legs": int(legs),
            "total_qty": float(total_qty),
            "vwap_entry": vwap_entry,
            "vwap_exit": vwap_exit,
            "symbol": _mode(gdf["symbol"]) if "symbol" in gdf else None,
            "side": _mode(gdf["side"]) if "side" in gdf else None,
            "pnl_net_sum": float(pnl_sum),
        })
    return pd.DataFrame(out)

def _extract_hashtags(notes: str):
    if not notes:
        return []
    return re.findall(r"#(\w+)", notes)

# ---------- UI ----------
st.title("Trading Performance (EST)")
tab_upload, tab_trades, tab_groups, tab_guards = st.tabs(["Upload", "Trades", "Groups", "Guardrails"])

# session_state: we still track last import list, but we will NOT auto-select
if "last_imported_external_ids" not in st.session_state:
    st.session_state.last_imported_external_ids = []

# ---- Upload ----
with tab_upload:
    st.subheader("Upload CSV")
    up = st.file_uploader("Drop your trade export", type=["csv"])
    if up:
        rows, dbg = _read_csv_to_rows(up.read())
        st.caption("Original headers: " + ", ".join(dbg["original_headers"]))
        st.caption("Normalized headers: " + ", ".join(dbg["normalized_headers"]))
        st.caption("Canonical mapped pairs: " + ", ".join([f"{pair[1]} -> {pair[0]}" for pair in dbg["canonical_mapped_pairs"]]))
        st.caption(f"Row count detected: {dbg['row_count']}")
        if rows:
            ins, skip, errs, samples, new_ext_ids, new_trade_ids = _upsert_trades_from_rows(rows)
            if ins > 0:
                st.success(f"Imported/updated {ins} trades (skipped {skip}).")
            else:
                st.error(f"Imported/updated {ins} trades (skipped {skip}).")
            if samples:
                with st.expander("Sample payloads attempted (first 5)"):
                    st.json(samples)
            if errs:
                with st.expander("Errors (first 25)"):
                    st.json(errs)
            st.cache_data.clear()
            st.session_state.last_imported_external_ids = new_ext_ids  # informative only
        else:
            st.error("No rows found.")

# ---- Trades ----
with tab_trades:
    st.subheader("Trades (legs)")
    df = _load_trades()
    if df.empty:
        st.info("No trades yet.")
    else:
        # Attach tags
        tagmap = _fetch_tags_for(df["id"].tolist())
        df["tags"] = df["id"].map(lambda i: ", ".join(sorted(tagmap.get(i, []))))

        rename_map = {
            "external_trade_id": "Trade ID",
            "entry_ts_est": "Entry (EST)",
            "exit_ts_est": "Exit (EST)",
            "pnl_net": "PnL (Net)",
        }

        # Checkbox column (default False always)
        if "selected" not in df.columns:
            df.insert(0, "selected", False)

        df_display = df.rename(columns=rename_map)

        view_cols = [
            "selected", "Trade ID", "symbol", "side",
            "Entry (EST)", "Exit (EST)", "qty", "pnl_gross", "fees", "PnL (Net)",
            "r_multiple", "review_status", "tags"
        ]
        view_cols = [c for c in view_cols if c in df_display.columns]

        edited = st.data_editor(
            df_display[view_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "selected": st.column_config.CheckboxColumn("✓"),
                "pnl_gross": st.column_config.NumberColumn("PnL (Gross)", step=0.01, format="%.2f"),
                "fees": st.column_config.NumberColumn("Fees", step=0.01, format="%.2f"),
                "PnL (Net)": st.column_config.NumberColumn("PnL (Net)", step=0.01, format="%.2f"),
                "r_multiple": st.column_config.NumberColumn("R Multiple", step=0.01),
                "review_status": st.column_config.SelectboxColumn("Review Status", options=["unreviewed", "flagged", "reviewed"]),
                "tags": st.column_config.TextColumn("Tags (read-only)"),
            },
            disabled=[c for c in view_cols if c not in ("selected", "r_multiple", "review_status")],
            num_rows="fixed",
        )

        # Map back to original and bring internal id for persistence
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
        # Bulk actions: comments, tags, grouping & UN-grouping
        selected_ids = edited_back.loc[edited_back.get("selected", False) == True, "id"].tolist()
        st.write(f"Selected: {len(selected_ids)}")

        with st.form("bulk_actions", clear_on_submit=True):
            comment = st.text_area("Add comment (markdown)")
            tag_str = st.text_input("Add tags (comma-separated)")

            st.markdown("**Grouping**")
            gmode = st.radio(
                "Group action",
                ["None", "Add to existing", "Create new (or auto)", "Remove from existing", "Remove from all groups"],
                horizontal=False
            )

            existing = None
            new_group = None
            notes = st.text_input("Group notes (optional, supports #hashtags e.g. #ON-Continuation #BaseHit)")

            # Inputs by mode
            if gmode == "Add to existing" or gmode == "Remove from existing":
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
                    st.info("No groups yet — choose 'Create new (or auto)'.")
            elif gmode == "Create new (or auto)":
                new_group = st.text_input("New group name (leave blank to auto-name)")

            do = st.form_submit_button("Apply")

        if do and selected_ids:
            # comments/tags
            if comment:
                _save_comments(selected_ids, comment)
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
            if tags:
                _save_tags(selected_ids, tags)

            # grouping actions
            if gmode == "Add to existing" and existing:
                _add_to_group(selected_ids, group_id=existing, notes=notes if notes else None)
                st.success("Added to group")
            elif gmode == "Create new (or auto)":
                gid = _add_to_group(selected_ids, new_group_name=(new_group if new_group is not None else None), notes=notes if notes else None)
                st.success(f"Created/used group: {gid[:6] if gid else ''}")
            elif gmode == "Remove from existing" and existing:
                _remove_from_group(selected_ids, group_id=existing, remove_from_all=False)
                st.success("Removed from selected group")
            elif gmode == "Remove from all groups":
                _remove_from_group(selected_ids, group_id=None, remove_from_all=True)
                st.success("Removed from all groups")

            st.cache_data.clear()
            st.rerun()

# ---- Groups (collapsed + hashtag filters + details) ----
with tab_groups:
    st.subheader("Groups (collapsed positions)")

    groups, mem_df = _fetch_all_groups_with_members()
    if not groups:
        st.info("No groups yet. Create them from the Trades tab after importing.")
    else:
        # Build rollup
        roll = _rollup_by_group(mem_df)

        # join names/notes
        name_map = {g["id"]: g["name"] for g in groups}
        notes_map = {g["id"]: g.get("notes") or "" for g in groups}
        if not roll.empty:
            roll["name"] = roll["group_id"].map(name_map)
            roll["notes"] = roll["group_id"].map(notes_map)

            # Hashtag extraction for filters
            all_hashtags = set()
            for n in roll["notes"].fillna(""):
                for h in _extract_hashtags(n):
                    all_hashtags.add(h)
            all_hashtags = sorted(all_hashtags)

            c1, c2, c3 = st.columns([1, 1, 3])
            with c1:
                day = st.date_input("Filter by day (first entry)", value=None)
            with c2:
                tag_filter = st.multiselect("Filter by #hashtag", options=all_hashtags, default=[])

            rshow = roll.copy()
            if day is not None:
                try:
                    rshow = rshow[rshow["first_entry"].dt.date == day]
                except Exception:
                    pass
            if tag_filter:
                # keep rows where notes contain ALL selected hashtags
                def _has_all_tags(text):
                    tags = set(_extract_hashtags(text or ""))
                    return all(t in tags for t in tag_filter)
                mask = rshow["notes"].apply(_has_all_tags)
                rshow = rshow[mask]

            show_cols = [
                "name","symbol","side","first_entry","last_exit",
                "legs","total_qty","vwap_entry","vwap_exit","pnl_net_sum","notes"
            ]
            show_cols = [c for c in show_cols if c in rshow.columns]

            st.markdown("**Collapsed (1 row per group)**")
            st.dataframe(
                rshow[show_cols].sort_values(["first_entry"], ascending=[False]),
                use_container_width=True
            )

            st.divider()

            # DETAILS for a selected group
            st.markdown("**Group details**")
            label_to_id = {f"{name_map.get(g['id'],'(unnamed)')} ({g['id'][:6]})": g["id"] for g in groups}
            choice = st.selectbox("Pick a group", list(label_to_id.keys()))
            gid = label_to_id[choice]

            gdf = mem_df[mem_df["group_id"] == gid].copy()
            if gdf.empty

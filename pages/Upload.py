#!/usr/bin/env python3
# pages/00_Upload.py

import io
import re
import hashlib

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Upload Trades (EST)", layout="wide")

# ===== CONFIG =====
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
    if (
        len(parts) == 2
        and len(parts[1]) == 6
        and parts[1][3] == ":"
        and (parts[1][0] in "+-")
    ):
        s = parts[0]
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _synth_id(row: dict) -> str:
    key = "|".join(
        [
            str(row.get("contractname", "")).strip().upper(),
            str(row.get("enteredat", "")).strip(),
            str(row.get("exitedat", "")).strip(),
            str(row.get("entryprice", "")).strip(),
            str(row.get("exitprice", "")).strip(),
            str(row.get("size", "")).strip(),
        ]
    )
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
        sep=None,  # auto-detect comma or tab
        engine="python",
        dtype=str,
        keep_default_na=False,  # keep empty cells as "", not NaN
    )
    original_headers = list(df.columns)
    norm_headers = [_clean_header(h) for h in original_headers]
    df.columns = norm_headers

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
        "tradeday": "tradeday",
        "tradeduration": "tradeduration",
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
        "canonical_mapped_pairs": sorted(
            [(k, v) for k, v in canonical_cols.items()]
        ),
        "row_count": len(rows),
    }
    return rows, debug


# ---------- insert-or-update (manual upsert keyed by user_id + external_trade_id) ----------
def _insert_or_update_trade(payload: dict):
    q = (
        sb.table("tj_trades")
        .select("id", count="exact")
        .eq("external_trade_id", payload["external_trade_id"])
        .eq("user_id", payload["user_id"])
        .limit(1)
    )
    res = q.execute()
    existing = res.data or []
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
            side = (
                "long"
                if t in ("long", "buy", "b")
                else ("short" if t in ("short", "sell", "s") else None)
            )

            qty_val = r.get("size")
            symbol = (r.get("contractname") or "").strip().upper() or None

            f1 = _as_float(r.get("fees")) or 0.0
            f2 = _as_float(r.get("commissions")) or 0.0
            fees_val = float(f1) + float(f2)

            entry_iso = _to_iso_est(r.get("enteredat"))
            exit_iso = _to_iso_est(r.get("exitedat"))
            if not entry_iso:
                raise ValueError(f"Missing/Bad EnteredAt: {r.get('enteredat')}")

            payload = {
                "user_id": USER_ID,  # REQUIRED
                "external_trade_id": ext_id,
                "symbol": symbol,
                "side": side,
                "entry_ts_est": entry_iso,
                "exit_ts_est": exit_iso,  # may be None for open trades
                "entry_px": _as_float(r.get("entryprice")),
                "exit_px": _as_float(r.get("exitprice")),
                "qty": _as_float(qty_val),
                "pnl_gross": _as_float(r.get("pnl")),
                "fees": fees_val,
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
                errs.append(
                    {
                        "id": r.get("id"),
                        "enteredat": r.get("enteredat"),
                        "exitedat": r.get("exitedat"),
                        "error": str(e),
                    }
                )

    return inserted, skipped, errs, samples, inserted_external_ids, inserted_trade_ids


# ===== Session state defaults (shared across pages) =====
if "just_imported" not in st.session_state:
    st.session_state.just_imported = False
if "last_imported_external_ids" not in st.session_state:
    st.session_state.last_imported_external_ids = []
if "auto_select_after_upload" not in st.session_state:
    st.session_state.auto_select_after_upload = False  # used on Trading Performance page

# ===== UI =====
st.title("Upload Trades (EST)")
st.subheader("Upload CSV")

up = st.file_uploader("Drop your trade export", type=["csv"])

if up and not st.session_state.just_imported:
    rows, dbg = _read_csv_to_rows(up.read())
    st.caption("Original headers: " + ", ".join(dbg["original_headers"]))
    st.caption("Normalized headers: " + ", ".join(dbg["normalized_headers"]))
    st.caption(
        "Canonical mapped pairs: "
        + ", ".join(
            [f"{pair[1]} -> {pair[0]}" for pair in dbg["canonical_mapped_pairs"]]
        )
    )
    st.caption(f"Row count detected: {dbg['row_count']}")
    if rows:
        (
            ins,
            skip,
            errs,
            samples,
            new_ext_ids,
            new_trade_ids,
        ) = _upsert_trades_from_rows(rows)
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
        st.session_state.last_imported_external_ids = new_ext_ids
        st.session_state.just_imported = True
        st.info("Now go to **Trading Performance** to tag/group and review trades.")
    else:
        st.error("No rows found.")
elif st.session_state.just_imported:
    st.info("Upload complete. Go to **Trading Performance** to work with the new trades.")
    if up is None:
        st.session_state.just_imported = False

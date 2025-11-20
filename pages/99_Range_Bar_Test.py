#!/usr/bin/env python3
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
from supabase import create_client
import streamlit.components.v1 as components

st.set_page_config(page_title="Range Bar Test", layout="wide")

# =========================
# Supabase client
# =========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# Helpers
# =========================
def _load_latest_session_from_es_30m(max_rows: int = 2000):
    """Load recent es_30m and compute latest trade_day (18:00 ET roll)."""
    resp = (
        sb.table("es_30m")
          .select("*")
          .order("time", desc=True)
          .limit(max_rows)
          .execute()
    )
    df = pd.DataFrame(resp.data)
    if df.empty:
        return None, None

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["time_et"] = df["time"].dt.tz_convert("US/Eastern")

    et = df["time_et"]
    midnight = et.dt.floor("D")
    roll = (et.dt.hour >= 18).astype("int64")
    df["trade_day"] = midnight + pd.to_timedelta(roll, unit="D")
    df["trade_date"] = df["trade_day"].dt.date

    latest_td = df["trade_day"].max()
    latest_session = df[df["trade_day"] == latest_td].copy().sort_values("time_et")
    return latest_session, latest_td


def _load_prior_rth_range(trade_date: dt.date):
    """
    From es_trade_day_summary, load prior day's RTH Hi / Lo.
    Columns in DB:
      "RTH Hi", "RTH Lo"
    """
    try:
        resp = (
            sb.table("es_trade_day_summary")
              .select('trade_date,"RTH Hi","RTH Lo"')
              .lte("trade_date", trade_date.isoformat())
              .order("trade_date", desc=True)
              .limit(5)
              .execute()
        )
        df = pd.DataFrame(resp.data)
        if df.empty:
            return np.nan, np.nan, None

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        # Prior = latest strictly less than current trade_date, if available
        prior_mask = df["trade_date"] < trade_date
        if prior_mask.any():
            prior_row = df.loc[prior_mask].sort_values("trade_date").iloc[-1]
        else:
            # fallback: use the last row as "prior"
            prior_row = df.sort_values("trade_date").iloc[-1]

        prior_date = prior_row["trade_date"]
        rth_hi = prior_row['RTH Hi']
        rth_lo = prior_row['RTH Lo']
        return rth_hi, rth_lo, prior_date
    except Exception as e:
        st.warning(f"Could not load es_trade_day_summary: {e}")
        return np.nan, np.nan, None


def _normalize_span(prior_hi, prior_lo, sess_hi, sess_lo, last_price):
    """
    Determine global min/max so we can map all levels to 0–100% width.
    Returns (global_min, global_max).
    """
    vals = [v for v in [prior_hi, prior_lo, sess_hi, sess_lo, last_price] if pd.notna(v)]
    if not vals:
        return 0.0, 1.0
    gmin = min(vals)
    gmax = max(vals)
    if gmax <= gmin:
        gmax = gmin + 1.0
    return gmin, gmax


def _pct(x, gmin, gmax):
    if pd.isna(x):
        return None
    return (x - gmin) / (gmax - gmin) * 100.0


# =========================
# Load data
# =========================
st.title("Range Bar Test — Prior RTH vs Current Session")

session_df, latest_td = _load_latest_session_from_es_30m()
if session_df is None or latest_td is None or session_df.empty:
    st.error("No data from es_30m to test with.")
    st.stop()

trade_date = latest_td.date()
sess_hi = session_df["high"].max()
sess_lo = session_df["low"].min()
last_price = session_df["close"].iloc[-1]

prior_rth_hi, prior_rth_lo, prior_trade_date = _load_prior_rth_range(trade_date)

col_info, col_main = st.columns([1, 2])

with col_info:
    st.markdown("**Data used for test**")
    st.write(
        {
            "current_trade_date": str(trade_date),
            "prior_trade_date": str(prior_trade_date),
            "prior_RTH_Hi": float(prior_rth_hi) if pd.notna(prior_rth_hi) else None,
            "prior_RTH_Lo": float(prior_rth_lo) if pd.notna(prior_rth_lo) else None,
            "current_session_Hi": float(sess_hi) if pd.notna(sess_hi) else None,
            "current_session_Lo": float(sess_lo) if pd.notna(sess_lo) else None,
            "last_price": float(last_price) if pd.notna(last_price) else None,
        }
    )

with col_main:
    st.markdown(
        "These are different **horizontal range visuals** showing how the **current session** "
        "relates to the **prior RTH range**."
    )

gmin, gmax = _normalize_span(prior_rth_hi, prior_rth_lo, sess_hi, sess_lo, last_price)

p_lo_pct = _pct(prior_rth_lo, gmin, gmax)
p_hi_pct = _pct(prior_rth_hi, gmin, gmax)
s_lo_pct = _pct(sess_lo, gmin, gmax)
s_hi_pct = _pct(sess_hi, gmin, gmax)
last_pct = _pct(last_price, gmin, gmax)


# =========================
# Variant 1: Overlay box (prior hollow + current fill + last marker)
# =========================
st.subheader("Variant 1 — Overlay Box (prior hollow, current filled, last marker)")

if any(pd.isna(v) for v in [prior_rth_hi, prior_rth_lo, sess_hi, sess_lo, last_price]):
    st.info("Missing values — cannot render this variant.")
else:
    html_v1 = f"""
    <div style="font-size:0.85rem; margin-bottom:6px;">
      Prior RTH: {prior_rth_lo:.2f} – {prior_rth_hi:.2f} &nbsp; | &nbsp;
      Current: {sess_lo:.2f} – {sess_hi:.2f} &nbsp; | &nbsp;
      Last: {last_price:.2f}
    </div>
    <div style="position:relative; height:90px; padding:14px 4px 10px 4px;">
      <!-- base track -->
      <div style="position:absolute; left:0; right:0; top:40px; height:8px;
                  background-color:#E5E7EB; border-radius:999px;">
      </div>

      <!-- prior RTH hollow box -->
      <div style="position:absolute;
                  top:36px;
                  left:{p_lo_pct:.2f}%;
                  width:{(p_hi_pct - p_lo_pct):.2f}%;
                  height:16px;
                  border:2px solid #6B7280;
                  border-radius:8px;
                  background-color:rgba(255,255,255,0.9);">
      </div>

      <!-- current session solid box (overlay) -->
      <div style="position:absolute;
                  top:38px;
                  left:{s_lo_pct:.2f}%;
                  width:{(s_hi_pct - s_lo_pct):.2f}%;
                  height:12px;
                  border-radius:6px;
                  background-color:rgba(59,130,246,0.55);">
      </div>

      <!-- LAST price marker -->
      <div style="position:absolute;
                  left:{last_pct:.2f}%;
                  top:28px;
                  width:2px;
                  height:30px;
                  background-color:#111827;">
      </div>
      <div style="position:absolute;
                  left:{last_pct:.2f}%;
                  top:20px;
                  transform:translateX(-50%);
                  font-size:0.75rem;
                  background-color:#F9FAFB;
                  padding:2px 4px;
                  border-radius:4px;
                  border:1px solid #D1D5DB;">
        Last {last_price:.2f}
      </div>

      <!-- prior RTH labels above -->
      <div style="position:absolute;
                  left:{p_lo_pct:.2f}%;
                  top:6px;
                  transform:translateX(-50%);
                  font-size:0.7rem;
                  color:#DC2626;">
        RTH Lo<br>{prior_rth_lo:.2f}
      </div>
      <div style="position:absolute;
                  left:{p_hi_pct:.2f}%;
                  top:6px;
                  transform:translateX(-50%);
                  font-size:0.7rem;
                  color:#16A34A;">
        RTH Hi<br>{prior_rth_hi:.2f}
      </div>

      <!-- current session labels below -->
      <div style="position:absolute;
                  left:{s_lo_pct:.2f}%;
                  bottom:0px;
                  transform:translateX(-50%);
                  font-size:0.7rem;
                  color:#DC2626;">
        Sess Lo<br>{sess_lo:.2f}
      </div>
      <div style="position:absolute;
                  left:{s_hi_pct:.2f}%;
                  bottom:0px;
                  transform:translateX(-50%);
                  font-size:0.7rem;
                  color:#16A34A;">
        Sess Hi<br>{sess_hi:.2f}
      </div>
    </div>
    """
    components.html(html_v1, height=130)


# =========================
# Variant 2: Zone bar (below / inside / above + last bubble)
# =========================
st.subheader("Variant 2 — Zone Bar (below / inside / above prior range)")

if any(pd.isna(v) for v in [prior_rth_hi, prior_rth_lo, last_price]):
    st.info("Missing values — cannot render this variant.")
else:
    html_v2 = f"""
    <div style="font-size:0.85rem; margin-bottom:6px;">
      Zone: red = below prior low, gray = inside prior RTH, green = above prior high.
      Last marked as a bubble.
    </div>
    <div style="position:relative; height:70px; padding:10px 4px 6px 4px;">
      <div style="position:absolute; left:0; right:0; top:26px; height:12px; border-radius:999px; overflow:hidden;">
        <!-- left/red (below low) -->
        <div style="position:absolute; left:0; width:{p_lo_pct:.2f}%; height:100%; background-color:#FCA5A5;"></div>
        <!-- mid/gray (inside prior range) -->
        <div style="position:absolute; left:{p_lo_pct:.2f}%; width:{(p_hi_pct - p_lo_pct):.2f}%; height:100%; background-color:#E5E7EB;"></div>
        <!-- right/green (above high) -->
        <div style="position:absolute; left:{p_hi_pct:.2f}%; width:{(100.0 - p_hi_pct):.2f}%; height:100%; background-color:#BBF7D0;"></div>
      </div>

      <!-- last bubble -->
      <div style="position:absolute;
                  top:20px;
                  left:{last_pct:.2f}%;
                  transform:translateX(-50%);
                  width:18px;
                  height:18px;
                  border-radius:999px;
                  background-color:#111827;">
      </div>
      <div style="position:absolute;
                  top:6px;
                  left:{last_pct:.2f}%;
                  transform:translateX(-50%);
                  font-size:0.75rem;">
        {last_price:.1f}
      </div>
    </div>
    """
    components.html(html_v2, height=90)


# =========================
# Variant 3: Double track (prior vs current as two thin lines)
# =========================
st.subheader("Variant 3 — Double Track (top = prior RTH, bottom = current session)")

if any(pd.isna(v) for v in [prior_rth_hi, prior_rth_lo, sess_hi, sess_lo]):
    st.info("Missing values — cannot render this variant.")
else:
    html_v3 = f"""
    <div style="font-size:0.85rem; margin-bottom:6px;">
      Top track: prior RTH. Bottom track: current session. Last price marker on bottom.
    </div>
    <div style="position:relative; height:80px; padding:8px 4px 10px 4px;">
      <!-- prior track -->
      <div style="position:absolute; left:0; right:0; top:26px; height:3px; background-color:#E5E7EB;"></div>
      <div style="position:absolute;
                  top:24px;
                  left:{p_lo_pct:.2f}%;
                  width:{(p_hi_pct - p_lo_pct):.2f}%;
                  height:7px;
                  background-color:#6B7280;">
      </div>

      <!-- current track -->
      <div style="position:absolute; left:0; right:0; top:50px; height:3px; background-color:#E5E7EB;"></div>
      <div style="position:absolute;
                  top:48px;
                  left:{s_lo_pct:.2f}%;
                  width:{(s_hi_pct - s_lo_pct):.2f}%;
                  height:7px;
                  background-color:#3B82F6;">
      </div>

      <!-- last marker on bottom -->
      <div style="position:absolute;
                  left:{last_pct:.2f}%;
                  top:42px;
                  width:2px;
                  height:20px;
                  background-color:#111827;">
      </div>
    </div>
    """
    components.html(html_v3, height=100)


# =========================
# Variant 4: Horizontal "candlestick" for current session,
# with ghosted prior range in background
# =========================
st.subheader("Variant 4 — Horizontal Candlestick for Current Session (with prior ghost)")

if any(pd.isna(v) for v in [prior_rth_hi, prior_rth_lo, sess_hi, sess_lo, last_price]):
    st.info("Missing values — cannot render this variant.")
else:
    html_v4 = f"""
    <div style="font-size:0.85rem; margin-bottom:6px;">
      Gray ghost = prior RTH. Blue bar = current session range. Black tick = last.
    </div>
    <div style="position:relative; height:70px; padding:10px 4px;">
      <!-- prior ghost band -->
      <div style="position:absolute;
                  top:28px;
                  left:{p_lo_pct:.2f}%;
                  width:{(p_hi_pct - p_lo_pct):.2f}%;
                  height:10px;
                  border-radius:5px;
                  background-color:rgba(148,163,184,0.4);">
      </div>

      <!-- current horizontal 'candle' -->
      <div style="position:absolute;
                  top:30px;
                  left:{s_lo_pct:.2f}%;
                  width:{(s_hi_pct - s_lo_pct):.2f}%;
                  height:6px;
                  border-radius:3px;
                  background-color:#3B82F6;">
      </div>

      <!-- last 'tick' -->
      <div style="position:absolute;
                  top:24px;
                  left:{last_pct:.2f}%;
                  width:2px;
                  height:18px;
                  background-color:#111827;">
      </div>
    </div>
    """
    components.html(html_v4, height=90)


# =========================
# Variant 5: Minimal labels only (no fancy bar, just aligned markers)
# =========================
st.subheader("Variant 5 — Minimal (labels aligned on a simple line)")

if any(pd.isna(v) for v in [prior_rth_hi, prior_rth_lo, sess_hi, sess_lo, last_price]):
    st.info("Missing values — cannot render this variant.")
else:
    html_v5 = f"""
    <div style="position:relative; height:80px; padding:12px 4px;">
      <div style="position:absolute; left:0; right:0; top:40px; height:1px; background-color:#D1D5DB;"></div>

      <!-- prior low / high -->
      <div style="position:absolute; top:22px; left:{p_lo_pct:.2f}%; transform:translateX(-50%);
                  font-size:0.75rem; color:#DC2626;">
        pLo {prior_rth_lo:.1f}
      </div>
      <div style="position:absolute; top:22px; left:{p_hi_pct:.2f}%; transform:translateX(-50%);
                  font-size:0.75rem; color:#16A34A;">
        pHi {prior_rth_hi:.1f}
      </div>

      <!-- session low / high -->
      <div style="position:absolute; top:46px; left:{s_lo_pct:.2f}%; transform:translateX(-50%);
                  font-size:0.75rem; color:#DC2626;">
        Lo {sess_lo:.1f}
      </div>
      <div style="position:absolute; top:46px; left:{s_hi_pct:.2f}%; transform:translateX(-50%);
                  font-size:0.75rem; color:#16A34A;">
        Hi {sess_hi:.1f}
      </div>

      <!-- last -->
      <div style="position:absolute; top:32px; left:{last_pct:.2f}%;
                  width:2px; height:16px; background-color:#111827;"></div>
      <div style="position:absolute; top:60px; left:{last_pct:.2f}%; transform:translateX(-50%);
                  font-size:0.75rem;">
        Last {last_price:.1f}
      </div>
    </div>
    """
    components.html(html_v5, height=100)

#!/usr/bin/env python3
"""
QuantVault SaaS Platform - Minimal Frontend

Design: single-page with expandable strategy details
Navigation: top bar links only
No sidebar, no radio, no form (avoids TextInput JS bug)
"""

import os
import sys
import hashlib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import MASTER_KEY, is_configured
from saas_platform.database.supabase_client import (
    get_public_strategies,
    get_strategy_equity_curve,
    create_user,
    get_user_by_username,
    get_user_by_id,
    update_user_api_keys,
    create_subscription,
    get_user_subscriptions,
    deactivate_subscription,
    get_user_orders,
)
from saas_platform.web_frontend.crypto_utils import encrypt_api_key

CACHE_TTL = 300

st.set_page_config(page_title="QuantVault", page_icon="◆", layout="wide")

for k, v in {'logged_in': False, 'user_id': None, 'username': '', 'lang': 'en', 'view': None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _get_strategies():
    return get_public_strategies()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _get_equity(sid, bt):
    return get_strategy_equity_curve(sid, is_backtest=bt, limit=10000)


def _prep(records):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df['nav_value'] = pd.to_numeric(df['nav_value'], errors='coerce')
    df = df.dropna(subset=['date', 'nav_value']).drop_duplicates(subset=['date'], keep='last')
    return df.sort_values('date').reset_index(drop=True)


def _downsample(df, n=400):
    if len(df) <= n:
        return df
    step = len(df) // n
    out = df.iloc[::step].copy()
    out = pd.concat([out, df.iloc[-1:]], ignore_index=True)
    return out.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)


def _chart(sid, live_start=None):
    bt_df = _prep(_get_equity(sid, True))
    lv_df = _prep(_get_equity(sid, False))

    if live_start:
        ls = pd.Timestamp(str(live_start)[:10])
        bt_df = bt_df[bt_df['date'] < ls] if not bt_df.empty else bt_df

    if bt_df.empty and lv_df.empty:
        st.info("No chart data")
        return

    bt_p = _downsample(bt_df) if not bt_df.empty else pd.DataFrame()
    lv_p = _downsample(lv_df) if not lv_df.empty else pd.DataFrame()

    dates = []
    if not bt_df.empty:
        dates.extend(bt_df['date'].tolist())
    if not lv_df.empty:
        dates.extend(lv_df['date'].tolist())

    xmin, xmax = min(dates), max(dates)
    now = pd.Timestamp.now(tz=xmax.tzinfo) if xmax.tzinfo else pd.Timestamp.now()
    if xmax < now:
        xmax = now

    fig = go.Figure()

    if not bt_df.empty:
        fig.add_trace(go.Scatter(
            x=bt_p['date'], y=bt_p['nav_value'],
            mode='lines', name='Backtest',
            line=dict(color='#6366f1', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>NAV: %{y:.2f}<extra></extra>',
        ))

    if not lv_df.empty:
        fig.add_trace(go.Scatter(
            x=lv_p['date'], y=lv_p['nav_value'],
            mode='lines', name='Live',
            line=dict(color='#22c55e', width=2.5),
            hovertemplate='%{x|%Y-%m-%d}<br>NAV: %{y:.2f}<extra></extra>',
        ))

    if live_start:
        ls_str = str(live_start)[:10]
        fig.add_shape(type="line", x0=ls_str, x1=ls_str, y0=0, y1=1, yref="paper",
                      line=dict(color="#f59e0b", width=1.5, dash="dash"))
        fig.add_annotation(x=ls_str, y=1.02, yref="paper", text="LIVE ▶",
                           showarrow=False, font=dict(size=10, color="#f59e0b"), xanchor="left")

    fig.update_layout(
        template='plotly_dark', height=400,
        margin=dict(l=50, r=20, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right', font=dict(size=11)),
        xaxis=dict(gridcolor='rgba(99,102,241,0.06)', range=[xmin, xmax], type='date'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.06)', side='right'),
    )

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})


def _pos_label(pos):
    if pos is None:
        return '—'
    pct = f"{abs(pos)*100:.0f}%"
    if pos > 0.01:
        return f"Long {pct}"
    elif pos < -0.01:
        return f"Short {pct}"
    return "Flat"


# ── Top Bar ──

c1, c2, c3 = st.columns([2, 3, 2])
with c1:
    if st.button("◆ QuantVault"):
        st.session_state.view = None
        st.rerun()
with c3:
    lc, ac = st.columns(2)
    with lc:
        if st.button("中文" if st.session_state.lang == 'en' else "EN"):
            st.session_state.lang = 'zh' if st.session_state.lang == 'en' else 'en'
            st.rerun()
    with ac:
        if st.session_state.logged_in:
            if st.button("Sign Out"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.username = ''
                st.rerun()
        else:
            if st.button("Sign In"):
                st.session_state.view = 'login'
                st.rerun()

st.divider()

# ── Router ──

view = st.session_state.view

if view == 'login':
    st.subheader("Sign In")
    u = st.text_input("Username", key="login_u")
    p = st.text_input("Password", type="password", key="login_p")
    if st.button("Login", type="primary"):
        if not u or not p:
            st.error("Please fill in all fields")
        else:
            user = get_user_by_username(u)
            if not user:
                st.error("User not found")
            elif user.get('password_hash') != hashlib.sha256(p.encode()).hexdigest():
                st.error("Incorrect password")
            else:
                st.session_state.logged_in = True
                st.session_state.user_id = user['id']
                st.session_state.username = u
                st.session_state.view = None
                st.success(f"Welcome back, {u}!")
                st.rerun()
    if st.button("Don't have an account? Sign Up"):
        st.session_state.view = 'register'
        st.rerun()

elif view == 'register':
    st.subheader("Sign Up")
    u = st.text_input("Username", key="reg_u")
    p = st.text_input("Password", type="password", key="reg_p")
    p2 = st.text_input("Confirm Password", type="password", key="reg_p2")
    if st.button("Register", type="primary"):
        if not u or not p:
            st.error("Please fill in all fields")
        elif p != p2:
            st.error("Passwords do not match")
        elif len(p) < 6:
            st.error("Min 6 characters")
        elif get_user_by_username(u):
            st.error("Username already exists")
        else:
            try:
                create_user({'username': u, 'password_hash': hashlib.sha256(p.encode()).hexdigest(), 'exchange': 'binance'})
                st.success("Account created! Please sign in.")
                st.session_state.view = 'login'
                st.rerun()
            except Exception as ex:
                st.error(f"Failed: {ex}")
    if st.button("Already have an account? Sign In"):
        st.session_state.view = 'login'
        st.rerun()

elif view == 'dashboard':
    if not st.session_state.logged_in:
        st.warning("Please sign in first")
        st.session_state.view = 'login'
        st.rerun()

    uid = st.session_state.user_id
    user = get_user_by_id(uid) if uid else None

    st.subheader("📊 Dashboard")
    tab1, tab2, tab3 = st.tabs(["API Key", "Subscriptions", "Positions"])

    with tab1:
        st.info("🔒 Your API key is encrypted with AES-256. Grant only Futures Trading permission, disable withdrawals.")
        has_api = bool(user and user.get('encrypted_api_key'))
        k = st.text_input("API Key", type="password", key="api_k")
        s = st.text_input("API Secret", type="password", key="api_s")
        if st.button("Update" if has_api else "Save", type="primary"):
            if not k or not s:
                st.error("Provide both API Key and Secret")
            elif not MASTER_KEY:
                st.error("MASTER_KEY not configured")
            else:
                try:
                    update_user_api_keys(uid, encrypt_api_key(k), encrypt_api_key(s), 'binance')
                    st.success("API Key saved!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed: {ex}")
        if has_api:
            st.success("✅ API Key bound")

    with tab2:
        subs = [x for x in get_user_subscriptions(uid) if x.get('is_active')]
        smap = {x['id']: x for x in _get_strategies()}
        if subs:
            for sub in subs:
                strat = smap.get(sub['strategy_id'], {})
                sn = strat.get('name', sub['strategy_id'][:8])
                cap = sub.get('allocated_capital_usdt', 0)
                cc1, cc2, cc3 = st.columns([3, 2, 1])
                cc1.markdown(f"**{sn}**")
                cc2.markdown(f"Capital: **{cap:.0f} USDT**")
                if cc3.button("Unsub", key=f"us_{sub['id']}"):
                    deactivate_subscription(sub['id'])
                    st.success("Unsubscribed")
                    st.rerun()

        st.divider()
        strats = _get_strategies()
        if strats:
            opts = {f"{x['name']} ({x.get('target_symbol', '')})": x['id'] for x in strats}
            sel = st.selectbox("Strategy", list(opts.keys()), key="sub_sel")
            amt = st.number_input("Capital (USDT)", min_value=100, max_value=1000000, value=1000, step=100, key="sub_amt")
            if st.button("Subscribe", type="primary"):
                sid = opts[sel]
                if any(x['strategy_id'] == sid for x in subs):
                    st.warning("Already subscribed")
                else:
                    try:
                        create_subscription(uid, sid, amt)
                        st.success("Subscribed!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Failed: {ex}")

    with tab3:
        subs = [x for x in get_user_subscriptions(uid) if x.get('is_active')]
        if not subs:
            st.info("No active subscriptions")
        else:
            for sub in subs:
                strat = smap.get(sub['strategy_id'], {})
                pos = strat.get('current_target_position', 0)
                cap = sub.get('allocated_capital_usdt', 0)
                cc1, cc2, cc3 = st.columns(3)
                cc1.markdown(f"**{strat.get('name', sub['strategy_id'][:8])}**")
                cc2.markdown(f"Capital: {cap:.0f} USDT")
                cc3.markdown(f"Position: {pos*100:.0f}% | Notional: {cap*(pos or 0):.0f}")

elif view and view.startswith('strat_'):
    sid = view[6:]
    strats = _get_strategies()
    s = next((x for x in strats if x['id'] == sid), None)
    if not s:
        st.error("Strategy not found")
        st.session_state.view = None
        st.rerun()

    if st.button("← Back"):
        st.session_state.view = None
        st.rerun()

    name = s.get('name', 'N/A')
    status = s.get('status', '')
    pos = s.get('current_target_position')
    live_start = s.get('live_start_date')
    symbol = s.get('target_symbol', '')
    badge = "🟢 LIVE" if status == 'LIVE' else "🟡 PAPER"

    st.markdown(f"## {name}  {badge}")
    info = [symbol]
    if live_start:
        info.append(f"Live since {str(live_start)[:10]}")
    st.caption('  ·  '.join(info))

    st.markdown("**📊 Backtest Performance**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Sharpe", f"{s.get('backtest_sharpe') or 0:.2f}")
    bt_ann = s.get('backtest_annualized_return')
    m2.metric("Ann. Return", f"{bt_ann:.1%}" if bt_ann else '—')
    bt_mdd = s.get('backtest_max_drawdown')
    m3.metric("Max DD", f"{bt_mdd:.1%}" if bt_mdd else '—')

    lv_df = _prep(_get_equity(s['id'], False))
    if not lv_df.empty:
        st.markdown("**🟢 Live Performance**")
        first_nav = lv_df['nav_value'].iloc[0]
        last_nav = lv_df['nav_value'].iloc[-1]
        ret = (last_nav - first_nav) / first_nav if first_nav > 0 else 0
        dur = (lv_df['date'].iloc[-1] - lv_df['date'].iloc[0]).days + 1
        l1, l2, l3 = st.columns(3)
        l1.metric("Total Return", f"{ret:.1%}", delta=f"{ret:.1%}", delta_color="normal" if ret >= 0 else "inverse")
        l2.metric("Duration", f"{dur} days")
        l3.metric("Today Signal", _pos_label(pos))

    _chart(s['id'], live_start)

    st.divider()
    st.markdown("### 🤖 Auto Copy Trade")
    st.info("Bind your Binance API key to auto-copy this strategy's trades. Grant only Futures Trading permission, disable withdrawals.")

    if st.session_state.logged_in:
        user = get_user_by_id(st.session_state.user_id)
        if user and user.get('encrypted_api_key'):
            st.success("✅ API Key bound")
            amt = st.number_input("Capital (USDT)", min_value=100, max_value=1000000, value=1000, step=100, key=f"alloc_{sid}")
            if st.button("Confirm Subscription", type="primary"):
                try:
                    create_subscription(st.session_state.user_id, sid, amt)
                    st.success("Subscribed!")
                except Exception as ex:
                    st.error(f"Failed: {ex}")
        else:
            st.warning("Please bind your Binance API key first.")
            if st.button("Go to Dashboard"):
                st.session_state.view = 'dashboard'
                st.rerun()
    else:
        st.warning("Please sign in first")
        if st.button("Sign In"):
            st.session_state.view = 'login'
            st.rerun()

else:
    # ── Home: Strategy Cards ──
    if not is_configured():
        st.error("⚠️ Database Not Connected")
    else:
        strats = _get_strategies()
        if not strats:
            st.warning("No strategies available yet")
        else:
            st.markdown("## 🏠 Strategy Overview")

            if st.session_state.logged_in:
                st.caption(f"👤 Signed in as **{st.session_state.username}** | [Dashboard](#)")
                if st.button("📊 Go to Dashboard"):
                    st.session_state.view = 'dashboard'
                    st.rerun()

            for s in strats:
                name = s.get('name', 'N/A')
                status = s.get('status', '')
                pos = s.get('current_target_position')
                live_start = s.get('live_start_date')
                symbol = s.get('target_symbol', '')
                badge = "🟢 LIVE" if status == 'LIVE' else "🟡 PAPER"

                with st.container():
                    cc1, cc2 = st.columns([4, 1])
                    with cc1:
                        st.markdown(f"### {name}  {badge}")
                        info = [symbol]
                        if live_start:
                            info.append(f"Live since {str(live_start)[:10]}")
                        st.caption('  ·  '.join(info))
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Sharpe", f"{s.get('backtest_sharpe') or 0:.2f}")
                        bt_ann = s.get('backtest_annualized_return')
                        m2.metric("Ann. Return", f"{bt_ann:.1%}" if bt_ann else '—')
                        bt_mdd = s.get('backtest_max_drawdown')
                        m3.metric("Max DD", f"{bt_mdd:.1%}" if bt_mdd else '—')
                    with cc2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("View Details →", key=f"vd_{s['id']}", type='primary'):
                            st.session_state.view = f"strat_{s['id']}"
                            st.rerun()
                st.divider()

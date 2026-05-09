#!/usr/bin/env python3
"""
QuantVault SaaS Platform - Frontend
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

st.set_page_config(page_title="QuantVault", page_icon="\u25c6", layout="wide")

for k, v in {'logged_in': False, 'user_id': None, 'username': '', 'lang': 'en', 'view': None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

TEXT = {
    'en': {
        'back': '\u2190 Back',
        'sign_in': 'Sign In',
        'sign_out': 'Sign Out',
        'login_title': 'Sign In',
        'login_btn': 'Login',
        'register_title': 'Sign Up',
        'register_btn': 'Register',
        'username': 'Username',
        'password': 'Password',
        'confirm_pwd': 'Confirm Password',
        'no_account': "Don't have an account? Sign Up",
        'has_account': 'Already have an account? Sign In',
        'fill_fields': 'Please fill in all fields',
        'user_not_found': 'User not found',
        'wrong_password': 'Incorrect password',
        'welcome': 'Welcome back, {}!',
        'pwd_mismatch': 'Passwords do not match',
        'min_pwd': 'Min 6 characters',
        'user_exists': 'Username already exists',
        'account_created': 'Account created! Please sign in.',
        'strategy_overview': 'Strategy Overview',
        'signed_in_as': 'Signed in as **{}**',
        'go_dashboard': 'Go to Dashboard',
        'view_details': 'View Details \u2192',
        'live': 'LIVE',
        'paper': 'PAPER',
        'live_since': 'Live since {}',
        'backtest_perf': 'Backtest Performance',
        'sharpe': 'Sharpe',
        'ann_return': 'Ann. Return',
        'max_dd': 'Max DD',
        'live_perf': 'Live Performance',
        'total_return': 'Total Return',
        'duration': 'Duration',
        'days': '{} days',
        'today_signal': 'Today Signal',
        'long': 'Long {}',
        'short': 'Short {}',
        'flat': 'Flat',
        'auto_copy': 'Auto Copy Trade',
        'api_info': 'Bind your Binance API key to auto-copy trades. Grant only Futures Trading permission, disable withdrawals.',
        'api_bound': 'API Key bound',
        'subscribe': 'Subscribe to this strategy',
        'subscribed': 'Subscribed! Daily auto-trading is now active.',
        'sub_failed': 'Failed: {}',
        'execute_now': 'Execute Trade Now',
        'executing': 'Executing trade...',
        'trade_done': 'Done! Orders: {}, Errors: {}',
        'trade_failed': 'Trade failed: {}',
        'recent_trades': 'Recent Trade History',
        'no_trades': 'No trades yet',
        'step1_ip': 'Step 1: Add server IP **{}** to your Binance API whitelist.',
        'step2_api': 'Step 2: Enter your Binance API Key and Secret below.',
        'api_key': 'API Key',
        'api_secret': 'API Secret',
        'save_api': 'Save API Key',
        'provide_both': 'Provide both API Key and Secret',
        'no_master_key': 'MASTER_KEY not configured',
        'api_saved': 'API Key saved!',
        'api_save_failed': 'Failed: {}',
        'please_sign_in': 'Please sign in first (click Sign In at the top right)',
        'no_strategies': 'No strategies available yet',
        'db_not_connected': 'Database Not Connected',
        'strategy_not_found': 'Strategy not found',
        'lang_btn': '\u4e2d\u6587',
        'backtest_label': 'Backtest',
        'live_label': 'Live',
        'no_chart_data': 'No chart data',
        'ccxt_missing': 'ccxt library not installed. Please contact admin to install it.',
        'tab_api': 'API Key',
        'tab_subs': 'Subscriptions',
        'tab_pos': 'Positions',
        'unsub': 'Unsub',
        'unsub_done': 'Unsubscribed',
        'sub_active_info': 'You are subscribed. The system will auto-trade daily based on the signal.',
    },
    'zh': {
        'back': '\u2190 \u8fd4\u56de',
        'sign_in': '\u767b\u5f55',
        'sign_out': '\u9000\u51fa',
        'login_title': '\u767b\u5f55',
        'login_btn': '\u767b\u5f55',
        'register_title': '\u6ce8\u518c',
        'register_btn': '\u6ce8\u518c',
        'username': '\u7528\u6237\u540d',
        'password': '\u5bc6\u7801',
        'confirm_pwd': '\u786e\u8ba4\u5bc6\u7801',
        'no_account': '\u6ca1\u6709\u8d26\u53f7\uff1f\u53bb\u6ce8\u518c',
        'has_account': '\u5df2\u6709\u8d26\u53f7\uff1f\u53bb\u767b\u5f55',
        'fill_fields': '\u8bf7\u586b\u5199\u6240\u6709\u5b57\u6bb5',
        'user_not_found': '\u7528\u6237\u4e0d\u5b58\u5728',
        'wrong_password': '\u5bc6\u7801\u9519\u8bef',
        'welcome': '\u6b22\u8fce\u56de\u6765\uff0c{}\uff01',
        'pwd_mismatch': '\u4e24\u6b21\u5bc6\u7801\u4e0d\u4e00\u81f4',
        'min_pwd': '\u5bc6\u7801\u81f3\u5c116\u4f4d',
        'user_exists': '\u7528\u6237\u540d\u5df2\u5b58\u5728',
        'account_created': '\u6ce8\u518c\u6210\u529f\uff01\u8bf7\u767b\u5f55\u3002',
        'strategy_overview': '\u7b56\u7565\u603b\u89c8',
        'signed_in_as': '\u5df2\u767b\u5f55\u4e3a **{}**',
        'go_dashboard': '\u8fdb\u5165\u63a7\u5236\u53f0',
        'view_details': '\u67e5\u770b\u8be6\u60c5 \u2192',
        'live': '\u5b9e\u76d8',
        'paper': '\u6a21\u62df',
        'live_since': '\u5b9e\u76d8\u8d77\u59cb {}',
        'backtest_perf': '\u56de\u6d4b\u8868\u73b0',
        'sharpe': '\u590f\u666e\u6bd4\u7387',
        'ann_return': '\u5e74\u5316\u6536\u76ca',
        'max_dd': '\u6700\u5927\u56de\u64a4',
        'live_perf': '\u5b9e\u76d8\u8868\u73b0',
        'total_return': '\u7d2f\u8ba1\u6536\u76ca',
        'duration': '\u8fd0\u884c\u5929\u6570',
        'days': '{} \u5929',
        'today_signal': '\u4eca\u65e5\u4fe1\u53f7',
        'long': '\u505a\u591a {}',
        'short': '\u505a\u7a7a {}',
        'flat': '\u7a7a\u4ed3',
        'auto_copy': '\u81ea\u52a8\u8ddf\u5355',
        'api_info': '\u7ed1\u5b9a\u5e01\u5b89 API Key \u4ee5\u81ea\u52a8\u8ddf\u5355\u3002\u4ec5\u6388\u4e88\u5408\u7ea6\u4ea4\u6613\u6743\u9650\uff0c\u7981\u7528\u63d0\u73b0\u3002',
        'api_bound': 'API Key \u5df2\u7ed1\u5b9a',
        'subscribe': '\u8ba2\u9605\u6b64\u7b56\u7565',
        'subscribed': '\u5df2\u8ba2\u9605\uff01\u6bcf\u65e5\u81ea\u52a8\u8ddf\u5355\u5df2\u5f00\u542f\u3002',
        'sub_failed': '\u5931\u8d25: {}',
        'execute_now': '\u7acb\u5373\u6267\u884c\u4ea4\u6613',
        'executing': '\u6b63\u5728\u6267\u884c\u4ea4\u6613...',
        'trade_done': '\u5b8c\u6210\uff01\u8ba2\u5355\u6570: {}, \u9519\u8bef\u6570: {}',
        'trade_failed': '\u4ea4\u6613\u5931\u8d25: {}',
        'recent_trades': '\u6700\u8fd1\u4ea4\u6613\u8bb0\u5f55',
        'no_trades': '\u6682\u65e0\u4ea4\u6613\u8bb0\u5f55',
        'step1_ip': '\u7b2c\u4e00\u6b65\uff1a\u5c06\u670d\u52a1\u5668 IP **{}** \u6dfb\u52a0\u5230\u5e01\u5b89 API \u767d\u540d\u5355\u3002',
        'step2_api': '\u7b2c\u4e8c\u6b65\uff1a\u5728\u4e0b\u65b9\u8f93\u5165\u5e01\u5b89 API Key \u548c Secret\u3002',
        'api_key': 'API Key',
        'api_secret': 'API Secret',
        'save_api': '\u4fdd\u5b58 API Key',
        'provide_both': '\u8bf7\u540c\u65f6\u63d0\u4f9b API Key \u548c Secret',
        'no_master_key': 'MASTER_KEY \u672a\u914d\u7f6e',
        'api_saved': 'API Key \u5df2\u4fdd\u5b58\uff01',
        'api_save_failed': '\u4fdd\u5b58\u5931\u8d25: {}',
        'please_sign_in': '\u8bf7\u5148\u767b\u5f55\uff08\u70b9\u51fb\u53f3\u4e0a\u89d2\u767b\u5f55\uff09',
        'no_strategies': '\u6682\u65e0\u53ef\u7528\u7b56\u7565',
        'db_not_connected': '\u6570\u636e\u5e93\u672a\u8fde\u63a5',
        'strategy_not_found': '\u7b56\u7565\u672a\u627e\u5230',
        'lang_btn': 'EN',
        'backtest_label': '\u56de\u6d4b',
        'live_label': '\u5b9e\u76d8',
        'no_chart_data': '\u6682\u65e0\u56fe\u8868\u6570\u636e',
        'ccxt_missing': 'ccxt \u5e93\u672a\u5b89\u88c5\uff0c\u8bf7\u8054\u7cfb\u7ba1\u7406\u5458\u5b89\u88c5\u3002',
        'tab_api': 'API \u5bc6\u94a5',
        'tab_subs': '\u8ba2\u9605',
        'tab_pos': '\u6301\u4ed3',
        'unsub': '\u9000\u8ba2',
        'unsub_done': '\u5df2\u9000\u8ba2',
        'sub_active_info': '\u5df2\u8ba2\u9605\uff0c\u7cfb\u7edf\u5c06\u6bcf\u65e5\u81ea\u52a8\u8ddf\u5355\u3002',
    }
}


def _t(key, *args):
    txt = TEXT.get(st.session_state.lang, TEXT['en']).get(key, key)
    if args:
        txt = txt.format(*args)
    return txt


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _get_strategies():
    return get_public_strategies()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _get_equity(sid, bt):
    return get_strategy_equity_curve(sid, is_backtest=bt, limit=10000)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_server_ip():
    try:
        import requests as _req
        return _req.get('https://api.ipify.org', timeout=5).text.strip()
    except Exception:
        return os.environ.get('SAAS_SERVER_IP', 'N/A')


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
        st.info(_t('no_chart_data'))
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
            mode='lines', name=_t('backtest_label'),
            line=dict(color='#6366f1', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>NAV: %{y:.2f}<extra></extra>',
        ))

    if not lv_df.empty:
        fig.add_trace(go.Scatter(
            x=lv_p['date'], y=lv_p['nav_value'],
            mode='lines', name=_t('live_label'),
            line=dict(color='#22c55e', width=2.5),
            hovertemplate='%{x|%Y-%m-%d}<br>NAV: %{y:.2f}<extra></extra>',
        ))

    if live_start:
        ls_str = str(live_start)[:10]
        fig.add_shape(type="line", x0=ls_str, x1=ls_str, y0=0, y1=1, yref="paper",
                      line=dict(color="#f59e0b", width=1.5, dash="dash"))
        fig.add_annotation(x=ls_str, y=1.02, yref="paper", text="LIVE \u25b6",
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
        return '\u2014'
    pct = f"{abs(pos)*100:.0f}%"
    if pos > 0.01:
        return _t('long', pct)
    elif pos < -0.01:
        return _t('short', pct)
    return _t('flat')


# ── Top Bar ──

c1, c2, c3 = st.columns([2, 3, 2])
with c1:
    if st.button("\u25c6 QuantVault"):
        st.session_state.view = None
        st.rerun()
with c3:
    lc, ac = st.columns(2)
    with lc:
        if st.button(_t('lang_btn'), key="lang_toggle_top"):
            st.session_state.lang = 'zh' if st.session_state.lang == 'en' else 'en'
            st.rerun()
    with ac:
        if st.session_state.logged_in:
            if st.button(_t('sign_out'), key="signout_top"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.session_state.username = ''
                st.rerun()
        else:
            if st.button(_t('sign_in'), key="signin_top"):
                st.session_state.view = 'login'
                st.rerun()

st.divider()

# ── Router ──

view = st.session_state.view

if view == 'login':
    st.subheader(_t('login_title'))
    u = st.text_input(_t('username'), key="login_u")
    p = st.text_input(_t('password'), type="password", key="login_p")
    if st.button(_t('login_btn'), type="primary", key="login_btn"):
        if not u or not p:
            st.error(_t('fill_fields'))
        else:
            user = get_user_by_username(u)
            if not user:
                st.error(_t('user_not_found'))
            elif user.get('password_hash') != hashlib.sha256(p.encode()).hexdigest():
                st.error(_t('wrong_password'))
            else:
                st.session_state.logged_in = True
                st.session_state.user_id = user['id']
                st.session_state.username = u
                st.session_state.view = None
                st.success(_t('welcome', u))
                st.rerun()
    if st.button(_t('no_account'), key="go_register"):
        st.session_state.view = 'register'
        st.rerun()

elif view == 'register':
    st.subheader(_t('register_title'))
    u = st.text_input(_t('username'), key="reg_u")
    p = st.text_input(_t('password'), type="password", key="reg_p")
    p2 = st.text_input(_t('confirm_pwd'), type="password", key="reg_p2")
    if st.button(_t('register_btn'), type="primary", key="reg_btn"):
        if not u or not p:
            st.error(_t('fill_fields'))
        elif p != p2:
            st.error(_t('pwd_mismatch'))
        elif len(p) < 6:
            st.error(_t('min_pwd'))
        elif get_user_by_username(u):
            st.error(_t('user_exists'))
        else:
            try:
                create_user({'username': u, 'password_hash': hashlib.sha256(p.encode()).hexdigest(), 'exchange': 'binance'})
                st.success(_t('account_created'))
                st.session_state.view = 'login'
                st.rerun()
            except Exception as ex:
                st.error(_t('api_save_failed', ex))
    if st.button(_t('has_account'), key="go_login"):
        st.session_state.view = 'login'
        st.rerun()

elif view == 'dashboard':
    if not st.session_state.logged_in:
        st.warning(_t('please_sign_in'))
        st.session_state.view = 'login'
        st.rerun()

    uid = st.session_state.user_id
    user = get_user_by_id(uid) if uid else None

    st.subheader("\U0001f4ca " + _t('go_dashboard').replace('\u2192', '').strip())
    tab1, tab2, tab3 = st.tabs([_t('tab_api'), _t('tab_subs'), _t('tab_pos')])

    with tab1:
        st.info("\U0001f512 " + _t('api_info'))
        has_api = bool(user and user.get('encrypted_api_key'))
        k = st.text_input(_t('api_key'), type="password", key="dash_api_k")
        s = st.text_input(_t('api_secret'), type="password", key="dash_api_s")
        if st.button(_t('save_api'), type="primary", key="dash_save_api"):
            if not k or not s:
                st.error(_t('provide_both'))
            elif not MASTER_KEY:
                st.error(_t('no_master_key'))
            else:
                try:
                    update_user_api_keys(uid, encrypt_api_key(k), encrypt_api_key(s), 'binance')
                    st.success(_t('api_saved'))
                    st.rerun()
                except Exception as ex:
                    st.error(_t('api_save_failed', ex))
        if has_api:
            st.success("\u2705 " + _t('api_bound'))

    with tab2:
        subs = [x for x in get_user_subscriptions(uid) if x.get('is_active')]
        smap = {x['id']: x for x in _get_strategies()}
        if subs:
            for sub in subs:
                strat = smap.get(sub['strategy_id'], {})
                sn = strat.get('name', sub['strategy_id'][:8])
                cc1, cc2 = st.columns([4, 1])
                cc1.markdown(f"**{sn}**")
                if cc2.button(_t('unsub'), key=f"us_{sub['id']}"):
                    deactivate_subscription(sub['id'])
                    st.success(_t('unsub_done'))
                    st.rerun()

    with tab3:
        subs = [x for x in get_user_subscriptions(uid) if x.get('is_active')]
        if not subs:
            st.info(_t('no_strategies'))
        else:
            for sub in subs:
                strat = smap.get(sub['strategy_id'], {})
                pos = strat.get('current_target_position', 0)
                cc1, cc2 = st.columns(2)
                cc1.markdown(f"**{strat.get('name', sub['strategy_id'][:8])}**")
                cc2.markdown(f"{_t('today_signal')}: {_pos_label(pos)}")

elif view and view.startswith('strat_'):
    sid = view[6:]
    strats = _get_strategies()
    s = next((x for x in strats if x['id'] == sid), None)
    if not s:
        st.error(_t('strategy_not_found'))
        st.session_state.view = None
        st.rerun()

    if st.button(_t('back'), key=f"back_{sid}"):
        st.session_state.view = None
        st.rerun()

    name = s.get('name', 'N/A')
    status = s.get('status', '')
    pos = s.get('current_target_position')
    live_start = s.get('live_start_date')
    symbol = s.get('target_symbol', '')
    badge = "\U0001f7e2 " + _t('live') if status == 'LIVE' else "\U0001f7e1 " + _t('paper')

    st.markdown(f"## {name}  {badge}")
    info = [symbol]
    if live_start:
        info.append(_t('live_since', str(live_start)[:10]))
    st.caption('  \u00b7  '.join(info))

    st.markdown(f"**\U0001f4ca {_t('backtest_perf')}**")
    m1, m2, m3 = st.columns(3)
    m1.metric(_t('sharpe'), f"{s.get('backtest_sharpe') or 0:.2f}")
    bt_ann = s.get('backtest_annualized_return')
    m2.metric(_t('ann_return'), f"{bt_ann:.1%}" if bt_ann else '\u2014')
    bt_mdd = s.get('backtest_max_drawdown')
    m3.metric(_t('max_dd'), f"{bt_mdd:.1%}" if bt_mdd else '\u2014')

    lv_df = _prep(_get_equity(s['id'], False))
    if not lv_df.empty:
        st.markdown(f"**\U0001f7e2 {_t('live_perf')}**")
        first_nav = lv_df['nav_value'].iloc[0]
        last_nav = lv_df['nav_value'].iloc[-1]
        ret = (last_nav - first_nav) / first_nav if first_nav > 0 else 0
        dur = (lv_df['date'].iloc[-1] - lv_df['date'].iloc[0]).days + 1
        l1, l2, l3 = st.columns(3)
        l1.metric(_t('total_return'), f"{ret:.1%}", delta=f"{ret:.1%}", delta_color="normal" if ret >= 0 else "inverse")
        l2.metric(_t('duration'), _t('days', dur))
        l3.metric(_t('today_signal'), _pos_label(pos))

    _chart(s['id'], live_start)

    st.divider()
    st.markdown(f"### \U0001f916 {_t('auto_copy')}")
    st.info(_t('api_info'))

    if st.session_state.logged_in:
        user = get_user_by_id(st.session_state.user_id)
        has_api = bool(user and user.get('encrypted_api_key'))

        if not has_api:
            server_ip = _get_server_ip()
            st.warning(_t('step1_ip', server_ip))
            st.info(_t('step2_api'))
            k = st.text_input(_t('api_key'), type="password", key=f"api_k_{sid}")
            sv = st.text_input(_t('api_secret'), type="password", key=f"api_s_{sid}")
            if st.button(_t('save_api'), type="primary", key=f"api_save_{sid}"):
                if not k or not sv:
                    st.error(_t('provide_both'))
                elif not MASTER_KEY:
                    st.error(_t('no_master_key'))
                else:
                    try:
                        update_user_api_keys(st.session_state.user_id, encrypt_api_key(k), encrypt_api_key(sv), 'binance')
                        st.success(_t('api_saved'))
                        st.rerun()
                    except Exception as ex:
                        st.error(_t('api_save_failed', ex))
        else:
            st.success("\u2705 " + _t('api_bound'))
            subs = [x for x in get_user_subscriptions(st.session_state.user_id) if x.get('is_active')]
            is_subbed = any(x['strategy_id'] == sid for x in subs)

            if not is_subbed:
                if st.button(_t('subscribe'), type="primary", key=f"sub_{sid}"):
                    try:
                        create_subscription(st.session_state.user_id, sid, 0)
                        st.success(_t('subscribed'))
                        st.rerun()
                    except Exception as ex:
                        st.error(_t('sub_failed', ex))
            else:
                st.info(_t('sub_active_info'))

                if st.button(_t('execute_now'), key=f"manual_{sid}"):
                    with st.spinner(_t('executing')):
                        try:
                            import ccxt as _ccxt
                            from saas_platform.production_engine.copy_trading_router import CopyTradingRouter
                            router = CopyTradingRouter(sandbox=False)
                            result = router.execute_single(st.session_state.user_id, sid)
                            if result.get('success'):
                                order = result.get('order')
                                if order:
                                    sandbox_tag = " (TESTNET)" if order.get('is_sandbox') else " (LIVE)"
                                    st.success(f"{_t('trade_done', 1, 0)}{sandbox_tag}")
                                    st.json({
                                        'side': order.get('side'),
                                        'amount': f"{order.get('amount', 0):.6f}",
                                        'price': f"{order.get('price', 0):.2f}",
                                        'fee': f"{order.get('fee', 0):.4f}",
                                        'balance_before': f"{order.get('balance_before', 0):.2f}",
                                        'balance_after': f"{order.get('balance_after', 0):.2f}",
                                        'position_before': f"{order.get('position_before', 0):.6f}",
                                        'position_after': f"{order.get('position_after', 0):.6f}",
                                    })
                                else:
                                    msg = result.get('message', 'No trade needed')
                                    st.info(msg)
                            else:
                                st.error(_t('trade_failed', result.get('error', 'Unknown error')))
                        except ImportError:
                            st.error(_t('ccxt_missing'))
                        except Exception as ex:
                            st.error(_t('trade_failed', ex))

                st.markdown(f"**{_t('recent_trades')}**")
                orders = get_user_orders(st.session_state.user_id, 20)
                if orders:
                    for o in orders:
                        ts = str(o.get('created_at', ''))[:19]
                        sym = o.get('symbol', '')
                        side = o.get('side', '')
                        amt = o.get('amount', 0)
                        px = o.get('price', 0)
                        fee = o.get('fee', 0)
                        status = o.get('status', '')
                        bal_b = o.get('balance_before', 0)
                        bal_a = o.get('balance_after', 0)
                        pos_b = o.get('position_before', 0)
                        pos_a = o.get('position_after', 0)
                        tgt = o.get('target_position', 0)
                        sandbox = o.get('is_sandbox', True)
                        ex_id = o.get('exchange_order_id', '')

                        if status == 'SKIPPED':
                            st.markdown(f"⏭️ `{ts}` | {sym} | SKIPPED | Target: {tgt*100:.0f}%")
                            continue

                        mode = "TESTNET" if sandbox else "LIVE"
                        side_icon = "🟢" if side == "buy" else "🔴" if side == "sell" else "⚠️"
                        status_icon = "✅" if status == "FILLED" else "❌" if status == "FAILED" else "⏳"

                        detail = (
                            f"{status_icon} `{ts}` | {side_icon} **{side.upper()}** | {sym} | [{mode}]\n"
                            f"- Amount: `{amt:.6f}` @ `{px:.2f}` | Fee: `{fee:.4f}`\n"
                        )
                        if bal_b or bal_a:
                            detail += f"- Balance: `{bal_b:.2f}` → `{bal_a:.2f}` USDT\n"
                        if pos_b or pos_a:
                            detail += f"- Position: `{pos_b:.6f}` → `{pos_a:.6f}`\n"
                        if tgt:
                            detail += f"- Target: `{tgt*100:.0f}%`\n"
                        if ex_id:
                            detail += f"- Order ID: `{ex_id}`\n"
                        if o.get('error_message'):
                            detail += f"- Error: `{o['error_message'][:100]}`\n"
                        st.markdown(detail)
                else:
                    st.caption(_t('no_trades'))
    else:
        st.warning(_t('please_sign_in'))

else:
    if not is_configured():
        st.error("\u26a0\ufe0f " + _t('db_not_connected'))
    else:
        strats = _get_strategies()
        if not strats:
            st.warning(_t('no_strategies'))
        else:
            st.markdown(f"## \U0001f3e0 {_t('strategy_overview')}")

            if st.session_state.logged_in:
                st.caption(f"\U0001f464 {_t('signed_in_as', st.session_state.username)}")
                if st.button(_t('go_dashboard'), key="home_dash"):
                    st.session_state.view = 'dashboard'
                    st.rerun()

            for s in strats:
                name = s.get('name', 'N/A')
                status = s.get('status', '')
                pos = s.get('current_target_position')
                live_start = s.get('live_start_date')
                symbol = s.get('target_symbol', '')
                badge = "\U0001f7e2 " + _t('live') if status == 'LIVE' else "\U0001f7e1 " + _t('paper')

                with st.container():
                    cc1, cc2 = st.columns([4, 1])
                    with cc1:
                        st.markdown(f"### {name}  {badge}")
                        info = [symbol]
                        if live_start:
                            info.append(_t('live_since', str(live_start)[:10]))
                        st.caption('  \u00b7  '.join(info))
                        m1, m2, m3 = st.columns(3)
                        m1.metric(_t('sharpe'), f"{s.get('backtest_sharpe') or 0:.2f}")
                        bt_ann = s.get('backtest_annualized_return')
                        m2.metric(_t('ann_return'), f"{bt_ann:.1%}" if bt_ann else '\u2014')
                        bt_mdd = s.get('backtest_max_drawdown')
                        m3.metric(_t('max_dd'), f"{bt_mdd:.1%}" if bt_mdd else '\u2014')
                    with cc2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button(_t('view_details'), key=f"vd_{s['id']}", type='primary'):
                            st.session_state.view = f"strat_{s['id']}"
                            st.rerun()
                st.divider()

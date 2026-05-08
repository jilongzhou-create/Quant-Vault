#!/usr/bin/env python3
"""
QuantVault SaaS Frontend

- Theme: .streamlit/config.toml (zero custom CSS)
- Charts: Plotly
- Navigation: streamlit radio
- i18n: EN/ZH bilingual, default EN
"""

import os
import sys
import hashlib

import streamlit as st
import pandas as pd
import numpy as np

CACHE_TTL = 300

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _cached_get_strategies():
    return get_public_strategies()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _cached_get_equity(strategy_id, is_backtest):
    return get_strategy_equity_curve(strategy_id, is_backtest=is_backtest, limit=10000)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import MASTER_KEY, is_configured, get_config_summary
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
from saas_platform.web_frontend.crypto_utils import encrypt_api_key, decrypt_api_key

import plotly.graph_objects as go
from datetime import datetime, timedelta

T = {
    'en': {
        'discover': 'Discover', 'dashboard': 'Dashboard', 'orders': 'Orders',
        'login': 'Sign In', 'register': 'Sign Up', 'logout': 'Sign Out',
        'lang': '中文',
        'subtitle': 'Algorithmic Strategy & Copy Trading Platform',
        'no_strategies': 'No strategies available yet',
        'sharpe': 'Sharpe', 'ann_ret': 'Ann. Return', 'mdd': 'Max DD', 'position': 'Position',
        'backtest': 'Backtest', 'live': 'Live', 'no_data': 'No chart data',
        'copy_trade': 'Copy Trade',
        'db_err': 'Database Not Connected',
        'username': 'Username', 'password': 'Password', 'confirm_pw': 'Confirm Password',
        'email': 'Email',
        'fill_all': 'Please fill in all fields', 'not_found': 'User not found',
        'wrong_pw': 'Incorrect password', 'welcome': 'Welcome back, {}!',
        'pw_mismatch': 'Passwords do not match', 'pw_min': 'Min 6 characters',
        'user_exists': 'Username already exists',
        'reg_ok': 'Account created! Please sign in.', 'reg_fail': 'Failed: {}',
        'api_title': 'Binance API Key',
        'api_note': 'Encrypted with AES-256. Never stored in plaintext.',
        'api_tip': 'Grant only Futures Trading permission. Disable withdrawals.',
        'api_key': 'API Key', 'api_secret': 'API Secret',
        'save': 'Save', 'update': 'Update',
        'fill_api': 'Provide both API Key and Secret',
        'no_master': 'MASTER_KEY not configured',
        'api_ok': 'API Key saved!', 'api_fail': 'Failed: {}', 'api_bound': 'API Key bound',
        'subs_title': 'Subscriptions', 'active_subs': 'Active', 'capital': 'Capital',
        'unsub': 'Unsubscribe', 'unsubed': 'Unsubscribed',
        'new_sub': 'New Subscription', 'select_strat': 'Strategy',
        'alloc': 'Capital (USDT)', 'confirm': 'Confirm',
        'already': 'Already subscribed', 'sub_ok': 'Subscribed!', 'sub_fail': 'Failed: {}',
        'no_sub': 'No strategies available',
        'pos_title': 'Positions', 'no_pos': 'No active subscriptions',
        'notional': 'Notional',
        'orders_title': 'Order History', 'no_orders': 'No orders yet',
        'please_login': 'Please sign in first',
        'bt_stats': 'Backtest Performance', 'live_stats': 'Live Performance',
        'total_return': 'Total Return', 'duration': 'Duration',
        'signal': 'Signal', 'today_signal': 'Today Signal',
        'days': 'days', 'long': 'Long', 'short': 'Short', 'flat': 'Flat',
    },
    'zh': {
        'discover': '策略发现', 'dashboard': '控制台', 'orders': '订单',
        'login': '登录', 'register': '注册', 'logout': '退出',
        'lang': 'EN',
        'subtitle': '量化策略跟单平台',
        'no_strategies': '暂无上线策略',
        'sharpe': '夏普率', 'ann_ret': '年化收益', 'mdd': '最大回撤', 'position': '仓位',
        'backtest': '回测', 'live': '实盘', 'no_data': '暂无数据',
        'copy_trade': '跟单',
        'db_err': '数据库未连接',
        'username': '用户名', 'password': '密码', 'confirm_pw': '确认密码',
        'email': '邮箱',
        'fill_all': '请填写所有字段', 'not_found': '用户不存在',
        'wrong_pw': '密码错误', 'welcome': '欢迎回来，{}！',
        'pw_mismatch': '密码不一致', 'pw_min': '至少6位',
        'user_exists': '用户名已存在',
        'reg_ok': '注册成功！请登录', 'reg_fail': '失败: {}',
        'api_title': 'Binance API Key',
        'api_note': 'AES-256加密存储，绝不存明文',
        'api_tip': '仅授予合约交易权限，禁止提币',
        'api_key': 'API Key', 'api_secret': 'API Secret',
        'save': '保存', 'update': '更新',
        'fill_api': '请填写完整的 Key 和 Secret',
        'no_master': 'MASTER_KEY 未配置',
        'api_ok': 'API Key 已保存', 'api_fail': '失败: {}', 'api_bound': '已绑定 API Key',
        'subs_title': '订阅', 'active_subs': '当前订阅', 'capital': '资金',
        'unsub': '退订', 'unsubed': '已退订',
        'new_sub': '新订阅', 'select_strat': '策略',
        'alloc': '资金 (USDT)', 'confirm': '确认',
        'already': '已订阅', 'sub_ok': '订阅成功！', 'sub_fail': '失败: {}',
        'no_sub': '暂无可订阅策略',
        'pos_title': '持仓', 'no_pos': '暂无活跃订阅',
        'notional': '名义敞口',
        'orders_title': '订单记录', 'no_orders': '暂无订单',
        'please_login': '请先登录',
        'bt_stats': '回测表现', 'live_stats': '实盘表现',
        'total_return': '总收益', 'duration': '运行时长',
        'signal': '信号', 'today_signal': '当日信号',
        'days': '天', 'long': '做多', 'short': '做空', 'flat': '空仓',
    },
}


def t(key):
    return T.get(st.session_state.get('lang', 'en'), T['en']).get(key, key)


st.set_page_config(page_title="QuantVault", page_icon="◆", layout="wide")


def _init():
    for k, v in {'logged_in': False, 'user_id': None, 'username': '', 'lang': 'en'}.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Navigation ──

def _nav():
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        st.markdown(f"### ◆ QuantVault")
        st.caption(t('subtitle'))
    with c3:
        if st.button(t('lang'), key='lang_btn'):
            st.session_state.lang = 'zh' if st.session_state.lang == 'en' else 'en'
            st.rerun()

    if st.session_state.logged_in:
        tabs = [t('discover'), t('dashboard'), t('orders'), t('logout')]
    else:
        tabs = [t('discover'), t('login'), t('register')]

    page = st.radio("Navigation", tabs, horizontal=True, label_visibility='collapsed', key='page_nav')

    if page == t('logout'):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = ''
        st.rerun()

    return page


# ── Chart ──

def _prepare_chart_data(records):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df['nav_value'] = pd.to_numeric(df['nav_value'], errors='coerce')
    df = df.dropna(subset=['date', 'nav_value'])
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _downsample(df, max_points=500):
    if len(df) <= max_points:
        return df
    step = len(df) // max_points
    sampled = df.iloc[::step].copy()
    last = df.iloc[-1:]
    sampled = pd.concat([sampled, last], ignore_index=True)
    sampled = sampled.drop_duplicates(subset=['date'], keep='last').sort_values('date').reset_index(drop=True)
    return sampled


def _chart(strategy_id, live_start_date=None):
    bt = _cached_get_equity(strategy_id, is_backtest=True)
    lv = _cached_get_equity(strategy_id, is_backtest=False)

    bt_df = _prepare_chart_data(bt)
    lv_df = _prepare_chart_data(lv)

    if live_start_date:
        ls = pd.Timestamp(live_start_date)
        if bt_df is not None and not bt_df.empty:
            bt_df = bt_df[bt_df['date'] < ls]

    bt_count = len(bt_df)
    lv_count = len(lv_df)

    if bt_count == 0 and lv_count == 0:
        st.info(t('no_data'))
        return

    bt_plot = _downsample(bt_df) if bt_count > 0 else pd.DataFrame()
    lv_plot = _downsample(lv_df) if lv_count > 0 else pd.DataFrame()

    all_dates = []
    if bt_count:
        all_dates.extend(bt_df['date'].tolist())
    if lv_count:
        all_dates.extend(lv_df['date'].tolist())

    x_min = min(all_dates)
    x_max = max(all_dates)
    today = pd.Timestamp.now(tz=x_max.tzinfo) if x_max.tzinfo else pd.Timestamp.now()
    if x_max < today:
        x_max = today

    fig = go.Figure()

    if bt_count:
        fig.add_trace(go.Scatter(
            x=bt_plot['date'].tolist(),
            y=bt_plot['nav_value'].tolist(),
            mode='lines',
            name=t('backtest'),
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.05)',
            hovertemplate='%{x|%Y-%m-%d}<br>NAV: %{y:.2f}<extra></extra>',
            connectgaps=True,
        ))

    if lv_count:
        fig.add_trace(go.Scatter(
            x=lv_plot['date'].tolist(),
            y=lv_plot['nav_value'].tolist(),
            mode='lines',
            name=t('live'),
            line=dict(color='#22c55e', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(34,197,94,0.05)',
            hovertemplate='%{x|%Y-%m-%d}<br>NAV: %{y:.2f}<extra></extra>',
            connectgaps=True,
        ))

    if live_start_date:
        ls_str = str(live_start_date)[:10]
        fig.add_shape(
            type="line",
            x0=ls_str, x1=ls_str,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=ls_str, y=1.02, yref="paper",
            text="LIVE ▶",
            showarrow=False,
            font=dict(size=10, color="#f59e0b"),
            xanchor="left",
        )

    fig.update_layout(
        template='plotly_dark',
        height=420,
        margin=dict(l=60, r=30, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            gridcolor='rgba(99,102,241,0.06)',
            range=[x_min, x_max],
            rangeslider=dict(visible=True, thickness=0.05),
            type='date',
        ),
        yaxis=dict(
            gridcolor='rgba(99,102,241,0.06)',
            side='right',
            tickfont=dict(size=11),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    })


def _calc_live_stats(lv_df):
    if lv_df is None or lv_df.empty:
        return None
    first_nav = lv_df['nav_value'].iloc[0]
    last_nav = lv_df['nav_value'].iloc[-1]
    total_return = (last_nav - first_nav) / first_nav if first_nav > 0 else 0
    first_date = lv_df['date'].iloc[0]
    last_date = lv_df['date'].iloc[-1]
    duration_days = (last_date - first_date).days + 1
    return {
        'total_return': total_return,
        'duration_days': duration_days,
        'first_date': first_date,
        'last_date': last_date,
    }


def _position_label(pos, lang='en'):
    if pos is None:
        return '—'
    pct = f"{abs(pos)*100:.0f}%"
    if pos > 0.01:
        direction = t('long') if lang == 'zh' else 'Long'
        return f"{direction} {pct}"
    elif pos < -0.01:
        direction = t('short') if lang == 'zh' else 'Short'
        return f"{direction} {pct}"
    else:
        return t('flat') if lang == 'zh' else 'Flat'


# ── Discover ──

def _discover():
    if not is_configured():
        st.error(f"⚠️ {t('db_err')}")
        st.markdown(f"1. Create project at [Supabase](https://supabase.com)\n"
                    f"2. Set `SAAS_SUPABASE_URL` / `SAAS_SUPABASE_KEY` / `MASTER_KEY` in `.env`\n"
                    f"3. Run `schema.sql` in Supabase SQL Editor\n"
                    f"4. Restart Streamlit")
        for k, v in get_config_summary().items():
            st.markdown(f"- `{k}`: `{v}`")
        return

    strategies = _cached_get_strategies()
    if not strategies:
        st.warning(t('no_strategies'))
        return

    for s in strategies:
        name = s.get('name', 'N/A')
        status = s.get('status', '')
        badge = "🟢 LIVE" if status == 'LIVE' else "🟡 PAPER"
        pos = s.get('current_target_position')
        live_start = s.get('live_start_date')

        c1, c2 = st.columns([4, 1])
        with c1:
            st.subheader(f"{name}  {badge}")
            info_parts = [s.get('target_asset', ''), s.get('target_symbol', '')]
            if live_start:
                info_parts.append(f"Live since {live_start[:10]}")
            st.caption(' · '.join(p for p in info_parts if p))
        with c2:
            if st.session_state.logged_in:
                if st.button(t('copy_trade'), key=f"ct_{s['id']}", type='primary'):
                    st.session_state.page_nav = t('dashboard')
                    st.rerun()

        bt_sharpe = s.get('backtest_sharpe') or 0
        bt_ann = s.get('backtest_annualized_return')
        bt_mdd = s.get('backtest_max_drawdown')

        st.markdown(f"**📊 {t('bt_stats')}**")
        m1, m2, m3 = st.columns(3)
        m1.metric(t('sharpe'), f"{bt_sharpe:.2f}")
        m2.metric(t('ann_ret'), f"{bt_ann:.1%}" if bt_ann else '—')
        m3.metric(t('mdd'), f"{bt_mdd:.1%}" if bt_mdd else '—')

        lv = _cached_get_equity(s['id'], is_backtest=False)
        lv_df = _prepare_chart_data(lv)
        live_stats = _calc_live_stats(lv_df)

        if live_stats:
            st.markdown(f"**🟢 {t('live_stats')}**")
            l1, l2, l3 = st.columns(3)
            l1.metric(t('total_return'), f"{live_stats['total_return']:.1%}",
                      delta=f"{live_stats['total_return']:.1%}",
                      delta_color="normal" if live_stats['total_return'] >= 0 else "inverse")
            l2.metric(t('duration'), f"{live_stats['duration_days']} {t('days')}")
            l3.metric(t('today_signal'), _position_label(pos, st.session_state.lang))

        _chart(s['id'], live_start_date=live_start)
        st.divider()


# ── Auth ──

def _login():
    st.subheader(t('login'))
    with st.form("login"):
        u = st.text_input(t('username'))
        p = st.text_input(t('password'), type='password')
        if st.form_submit_button(t('login'), type='primary'):
            if not u or not p:
                st.error(t('fill_all')); return
            user = get_user_by_username(u)
            if not user:
                st.error(t('not_found')); return
            if user.get('password_hash') != hashlib.sha256(p.encode()).hexdigest():
                st.error(t('wrong_pw')); return
            st.session_state.logged_in = True
            st.session_state.user_id = user['id']
            st.session_state.username = u
            st.success(t('welcome').format(u))
            st.rerun()


def _register():
    st.subheader(t('register'))
    with st.form("reg"):
        u = st.text_input(t('username'))
        e = st.text_input(t('email'))
        p = st.text_input(t('password'), type='password')
        p2 = st.text_input(t('confirm_pw'), type='password')
        if st.form_submit_button(t('register'), type='primary'):
            if not u or not e or not p:
                st.error(t('fill_all')); return
            if p != p2:
                st.error(t('pw_mismatch')); return
            if len(p) < 6:
                st.error(t('pw_min')); return
            if get_user_by_username(u):
                st.error(t('user_exists')); return
            try:
                create_user({'username': u, 'email': e, 'password_hash': hashlib.sha256(p.encode()).hexdigest(), 'exchange': 'binance'})
                st.success(t('reg_ok'))
                st.rerun()
            except Exception as ex:
                st.error(t('reg_fail').format(ex))


# ── Dashboard ──

def _dashboard():
    if not st.session_state.logged_in:
        st.warning(t('please_login')); return

    uid = st.session_state.user_id
    user = get_user_by_id(uid) if uid else None

    tab_api, tab_sub, tab_pos = st.tabs([t('api_title'), t('subs_title'), t('pos_title')])

    with tab_api:
        st.markdown(f"**{t('api_note')}**")
        st.info(t('api_tip'))
        has = bool(user and user.get('encrypted_api_key'))
        with st.form("api"):
            k = st.text_input(t('api_key'), type='password')
            s = st.text_input(t('api_secret'), type='password')
            if st.form_submit_button(t('update') if has else t('save'), type='primary'):
                if not k or not s:
                    st.error(t('fill_api')); return
                if not MASTER_KEY:
                    st.error(t('no_master')); return
                try:
                    update_user_api_keys(uid, encrypt_api_key(k), encrypt_api_key(s), 'binance')
                    st.success(t('api_ok')); st.rerun()
                except Exception as ex:
                    st.error(t('api_fail').format(ex))
        if has:
            st.success(t('api_bound'))

    with tab_sub:
        subs = [s for s in get_user_subscriptions(uid) if s.get('is_active')]
        if subs:
            st.markdown(f"**{t('active_subs')}**")
            smap = {s['id']: s for s in _cached_get_strategies()}
            for sub in subs:
                strat = smap.get(sub['strategy_id'], {})
                sn = strat.get('name', sub['strategy_id'][:8])
                c1, c2, c3 = st.columns([3, 2, 1])
                c1.markdown(f"**{sn}**")
                c2.markdown(f"{t('capital')}: **{sub.get('allocated_capital_usdt', 0):.0f} USDT**")
                if c3.button(t('unsub'), key=f"us_{sub['id']}"):
                    deactivate_subscription(sub['id'])
                    st.success(t('unsubed')); st.rerun()

        st.divider()
        st.markdown(f"**{t('new_sub')}**")
        strats = _cached_get_strategies()
        if not strats:
            st.info(t('no_sub')); return
        opts = {f"{s['name']} ({s.get('target_symbol', '')})": s['id'] for s in strats}
        with st.form("sub"):
            sel = st.selectbox(t('select_strat'), list(opts.keys()))
            amt = st.number_input(t('alloc'), min_value=100, max_value=1000000, value=1000, step=100)
            if st.form_submit_button(t('confirm'), type='primary'):
                sid = opts[sel]
                if any(s['strategy_id'] == sid for s in subs):
                    st.warning(t('already')); return
                try:
                    create_subscription(uid, sid, amt)
                    st.success(t('sub_ok')); st.rerun()
                except Exception as ex:
                    st.error(t('sub_fail').format(ex))

    with tab_pos:
        subs = [s for s in get_user_subscriptions(uid) if s.get('is_active')]
        if not subs:
            st.info(t('no_pos')); return
        smap = {s['id']: s for s in _cached_get_strategies()}
        for sub in subs:
            strat = smap.get(sub['strategy_id'], {})
            pos = strat.get('current_target_position', 0)
            cap = sub.get('allocated_capital_usdt', 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**{strat.get('name', sub['strategy_id'][:8])}**")
            c2.markdown(f"{t('capital')}: {cap:.0f} USDT")
            c3.markdown(f"{t('position')}: {pos*100:.0f}%" if pos else f"{t('position')}: —")
            c4.markdown(f"{t('notional')}: {cap * (pos or 0):.0f} USDT")


# ── Orders ──

def _orders():
    if not st.session_state.logged_in:
        st.warning(t('please_login')); return
    st.subheader(t('orders_title'))
    orders = get_user_orders(st.session_state.user_id, 100)
    if not orders:
        st.info(t('no_orders')); return
    df = pd.DataFrame(orders)
    cols = [c for c in ['created_at', 'symbol', 'side', 'amount', 'price', 'fee', 'status'] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


# ── Main ──

page = _nav()

if page == t('discover'):
    _discover()
elif page == t('login'):
    _login()
elif page == t('register'):
    _register()
elif page == t('dashboard'):
    _dashboard()
elif page == t('orders'):
    _orders()

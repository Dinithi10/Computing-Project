"""
RiskLens — Sri Lanka × Malaysia Sovereign Risk Intelligence
Professional financial-research dashboard inspired by Risk.net layout.
Supports light & dark themes, with axis-labelled charts and 7-day forecasts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from utils.auth import sign_in, sign_up
from utils.data_api import (
    COUNTRIES, INDICATORS, EXCHANGE_RATE_KEY,
    load_indicator, expand_to_frequency, build_indicator_panel, filter_year_range,
)
from utils.forecast import forecast, risk_score
from utils.ml_models import prepare_dataset, benchmark_regressors
from utils.report import build_report

YEAR_MIN, YEAR_MAX = 2000, 2026
BRAND_NAME = "RiskLens"
BRAND_MARK = "◈"

st.set_page_config(
    page_title=f"{BRAND_NAME} — Sri Lanka × Malaysia Risk Intelligence",
    page_icon=BRAND_MARK,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Theme system
# ──────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

THEMES = {
    "Light": {
        "bg": "#FFFFFF", "panel": "#F8FAFC", "ink": "#0B1F3A",
        "muted": "#475569", "border": "#CBD5E1", "grid": "#D1D5DB",
        "navy": "#0B1F3A", "navy2": "#13315C", "accent": "#B8860B",
        "lk": "#0B5394", "my": "#A4161A",
        "line": "#0B1F3A", "fillalpha": "rgba(11,31,58,0.10)",
    },
    "Dark": {
        "bg": "#0B1220", "panel": "#0F1A2E", "ink": "#F1F5F9",
        "muted": "#CBD5E1", "border": "#334155", "grid": "#243049",
        "navy": "#0B1F3A", "navy2": "#1E3A5F", "accent": "#E0B84A",
        "lk": "#5FA8E6", "my": "#F26B6F",
        "line": "#5FA8E6", "fillalpha": "rgba(95,168,230,0.18)",
    },
}

def axis_label_for(indicator: str) -> str:
    m = {
        "GDP (current US$)": "GDP (current US$)",
        "Inflation (CPI %)": "Inflation rate (%)",
        "Unemployment Rate (%)": "Unemployment rate (%)",
        "Real Interest Rate (%)": "Real interest rate (%)",
        "Wage & Salaried Workers (%)": "Wage & salaried workers (% of employment)",
        "Total Natural Resources Rents (% GDP)": "Natural resources rents (% of GDP)",
        "Infrastructure (Logistics Performance Index)": "Logistics Performance Index (1–5)",
        EXCHANGE_RATE_KEY: "Exchange rate (LCU per USD)",
    }
    return m.get(indicator, "Value")

def x_label_for(freq: str) -> str:
    return {"D": "Date (daily)", "M": "Month", "Y": "Year"}.get(freq, "Date")

T = THEMES[st.session_state.theme]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, .stApp {{
    font-family: 'Inter', system-ui, sans-serif;
    color: {T['ink']};
}}
.stApp, .main {{ background: {T['bg']} !important; }}
.block-container {{ padding-top: 0 !important; padding-left: 2rem; padding-right: 2rem; max-width: 1400px; }}

/* Hide Streamlit's default header so our topbar sits at the very top */
header[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}

.topbar {{
    background: {T['navy']};
    margin: 0 -2rem 1.6rem -2rem;
    padding: 1.1rem 2rem;
    border-bottom: 3px solid {T['accent']};
    display: flex; align-items: center; justify-content: space-between;
    min-height: 76px;
}}
.topbar .brand {{ display:flex; align-items:center; gap:0.9rem;
    font-family:'Source Serif 4', serif; color:#FFFFFF; }}
.topbar .brand .mark {{
    width:42px; height:42px; border:1.5px solid {T['accent']};
    display:flex; align-items:center; justify-content:center;
    color:{T['accent']}; font-weight:700; font-size:1.4rem;
    border-radius:2px;
}}
.topbar .brand .name {{ font-size:1.55rem; font-weight:700; letter-spacing:0.4px; color:#FFFFFF; }}
.topbar .brand .tag  {{ font-size:0.78rem; color:#E5E7EB; font-family:'Inter',sans-serif;
    letter-spacing:1.6px; text-transform:uppercase; margin-top:2px; }}

section[data-testid="stSidebar"] {{ background:{T['panel']} !important;
    border-right:1px solid {T['border']}; }}
section[data-testid="stSidebar"] > div {{ color:{T['ink']} !important; }}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{ color:{T['ink']} !important; }}
section[data-testid="stSidebar"] h3 {{ color:{T['accent']} !important;
    font-size:0.78rem;letter-spacing:1.2px;text-transform:uppercase;margin-top:1rem; }}
/* Keep button text white (overrides the sidebar-wide ink color rule) */
section[data-testid="stSidebar"] .stButton>button,
section[data-testid="stSidebar"] .stButton>button * {{ color:#FFFFFF !important; }}
section[data-testid="stSidebar"] .stDownloadButton>button,
section[data-testid="stSidebar"] .stDownloadButton>button * {{ color:#FFFFFF !important; }}
.stButton>button, .stButton>button *,
.stFormSubmitButton>button, .stFormSubmitButton>button * {{ color:#FFFFFF !important; }}

.page-h {{ border-bottom:1px solid {T['border']}; padding-bottom:0.6rem; margin-bottom:1.2rem; }}
.page-h h1 {{ font-family:'Source Serif 4', serif; color:{T['ink']}; font-size:1.7rem;
    margin:0; font-weight:700; }}
.page-h p {{ color:{T['muted']}; margin:0.2rem 0 0 0; font-size:0.88rem; }}

.kpi {{ background:{T['panel']}; border:1px solid {T['border']};
    border-left:3px solid {T['navy']}; border-radius:4px; padding:0.85rem 1rem; }}
.kpi h4 {{ color:{T['muted']}; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:1px; margin:0 0 0.35rem 0; font-weight:600; }}
.kpi .v {{ color:{T['ink']}; font-size:1.45rem; font-weight:700; font-feature-settings:'tnum'; }}
.kpi.gold {{ border-left-color:{T['accent']}; }}
.kpi.lk   {{ border-left-color:{T['lk']}; }}
.kpi.my   {{ border-left-color:{T['my']}; }}

.pill {{ display:inline-block; padding:0.35rem 0.9rem; border-radius:2px; font-weight:600;
    font-size:0.78rem; color:white; letter-spacing:0.5px; text-transform:uppercase; }}

.stButton>button {{ background:{T['navy']}; color:white; border:0; border-radius:3px;
    padding:0.55rem 1.3rem; font-weight:600; letter-spacing:0.3px; }}
.stButton>button:hover {{ background:{T['navy2']}; }}
.stDownloadButton>button {{ background:{T['accent']}; color:#FFFFFF; border:0;
    border-radius:3px; padding:0.55rem 1.3rem; font-weight:700; }}

#MainMenu, footer {{ visibility:hidden; }}

/* Section heading above charts (e.g. "7-Day Forecast") with breathing room */
.chart-heading {{
    font-family:'Source Serif 4', serif; color:{T['ink']};
    font-size:1.05rem; font-weight:700;
    margin: 1.4rem 0 1.1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {T['border']};
}}

.hero {{
    background: linear-gradient(135deg, {T['navy']} 0%, {T['navy2']} 100%);
    color: white; padding: 4rem 3rem; border-radius: 4px;
    border-left: 4px solid {T['accent']}; margin-bottom: 2rem;
}}
.hero h1 {{ font-family:'Source Serif 4', serif; font-size:2.6rem; margin:0 0 0.6rem 0; font-weight:700; color:#FFFFFF; }}
.hero .sub {{ color:{T['accent']}; font-size:0.85rem; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:1rem; }}
.hero p {{ color:#E5E7EB; font-size:1.05rem; max-width:680px; line-height:1.6; }}
.feat {{ background:{T['panel']}; border:1px solid {T['border']}; padding:1.2rem;
    border-top:3px solid {T['navy']}; }}
.feat h4 {{ color:{T['ink']}; margin:0 0 0.4rem 0; font-size:1rem; font-weight:700; }}
.feat p {{ color:{T['muted']}; margin:0; font-size:0.85rem; line-height:1.5; }}

/* DataFrames readable in both themes */
[data-testid="stDataFrame"], [data-testid="stDataFrame"] * {{
    color:{T['ink']} !important;
}}
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] thead, [data-testid="stDataFrame"] thead * {{
    background:{T['panel']} !important;
    color:{T['ink']} !important;
    border-color:{T['border']} !important;
    font-weight:700 !important;
}}
[data-testid="stDataFrame"] [role="row"],
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="rowheader"] {{
    background:{T['bg']} !important;
    color:{T['ink']} !important;
    border-color:{T['border']} !important;
}}
[data-testid="stDataFrame"] [data-testid="stTable"] {{ background:{T['bg']} !important; }}
/* Table (st.table) */
[data-testid="stTable"], [data-testid="stTable"] * {{
    color:{T['ink']} !important; background:{T['bg']} !important;
    border-color:{T['border']} !important;
}}
[data-testid="stTable"] thead tr th {{ background:{T['panel']} !important; }}

/* Selectbox / inputs — force theme-aware colors so they're readable in both modes */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [data-baseweb="input"] input {{
    background:{T['panel']} !important;
    color:{T['ink']} !important;
    border-color:{T['border']} !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] [data-baseweb="input"] * {{
    color:{T['ink']} !important;
}}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {{
    background:{T['panel']} !important; color:{T['ink']} !important;
}}
/* Global text inputs (login form, etc.) — readable in both themes */
[data-baseweb="input"] > div, [data-baseweb="input"] input,
.stTextInput > div > div, .stTextInput input {{
    background:{T['panel']} !important;
    color:{T['ink']} !important;
    border-color:{T['border']} !important;
}}
.stTextInput label, .stSelectbox label, .stRadio label {{ color:{T['ink']} !important; }}
/* Tabs (Sign in / Create account) */
.stTabs [data-baseweb="tab"] {{ color:{T['muted']} !important; }}
.stTabs [aria-selected="true"] {{ color:{T['ink']} !important; }}

/* ─── Form inputs: ensure WHITE background + dark text in light mode
       (Streamlit ships with a translucent grey fill that washes out values) */
[data-baseweb="input"], [data-baseweb="input"] > div,
[data-baseweb="input"] input,
.stTextInput > div > div, .stTextInput input,
input[type="text"], input[type="email"], input[type="password"] {{
    background:{T['bg']} !important;
    color:{T['ink']} !important;
    -webkit-text-fill-color:{T['ink']} !important;
    border:1px solid {T['border']} !important;
    caret-color:{T['ink']} !important;
}}
input::placeholder {{ color:{T['muted']} !important; opacity:0.85 !important; }}

/* Password reveal (eye) button — Streamlit renders it inside the input
   wrapper; in light mode the icon inherits white-on-white. Force visible. */
[data-baseweb="input"] button,
.stTextInput button {{
    background:{T['bg']} !important;
    border-color:{T['border']} !important;
}}
[data-baseweb="input"] button svg,
.stTextInput button svg,
[data-baseweb="input"] button svg * {{
    fill:{T['ink']} !important;
    stroke:{T['ink']} !important;
    color:{T['ink']} !important;
    opacity:1 !important;
}}

/* ─── Alerts (info / success / warning / error) — readable in BOTH themes.
       Streamlit's defaults render pale text on pale tints in light mode. */
div[data-testid="stAlert"], div[data-testid="stAlertContainer"],
div[data-testid="stAlert"] *, div[data-testid="stAlertContainer"] * {{
    color:{T['ink']} !important;
    -webkit-text-fill-color:{T['ink']} !important;
    font-weight:500 !important;
}}
div[data-testid="stAlert"] {{
    border:1px solid {T['border']} !important;
    border-radius:4px !important;
}}
/* Per-status accent backgrounds tuned for light mode */
div[data-testid="stAlert"][kind="info"],
div[data-testid="stAlertContainer"][kind="info"] {{
    background:{'#E0F2FE' if st.session_state.theme == 'Light' else '#0F2A44'} !important;
    border-left:4px solid #0284C7 !important;
}}
div[data-testid="stAlert"][kind="success"],
div[data-testid="stAlertContainer"][kind="success"] {{
    background:{'#DCFCE7' if st.session_state.theme == 'Light' else '#0F2A1E'} !important;
    border-left:4px solid #16A34A !important;
}}
div[data-testid="stAlert"][kind="warning"],
div[data-testid="stAlertContainer"][kind="warning"] {{
    background:{'#FEF3C7' if st.session_state.theme == 'Light' else '#3A2E10'} !important;
    border-left:4px solid #D97706 !important;
}}
div[data-testid="stAlert"][kind="error"],
div[data-testid="stAlertContainer"][kind="error"] {{
    background:{'#FEE2E2' if st.session_state.theme == 'Light' else '#3A1414'} !important;
    border-left:4px solid #DC2626 !important;
}}

/* ─── HTML tables (st.table) — fully theme-aware (used in place of stDataFrame
       for the Dashboard / Forecast tables, since the canvas-based dataframe
       ignores CSS in light mode and renders dark on dark). */
.risk-table {{
    width:100%;
    border-collapse:collapse;
    font-size:0.88rem;
    background:{T['bg']};
    border:1px solid {T['border']};
    border-radius:4px;
    overflow:hidden;
    margin: 0.4rem 0 1rem 0;
}}
.risk-table thead th {{
    background:{T['navy']};
    color:#FFFFFF !important;
    text-align:left;
    padding:0.7rem 0.9rem;
    font-weight:600;
    letter-spacing:0.4px;
    font-size:0.78rem;
    text-transform:uppercase;
    border-bottom:2px solid {T['accent']};
}}
.risk-table tbody td {{
    padding:0.6rem 0.9rem;
    color:{T['ink']} !important;
    border-bottom:1px solid {T['border']};
    font-feature-settings:'tnum';
}}
.risk-table tbody tr:nth-child(even) td {{ background:{T['panel']}; }}
.risk-table tbody tr:hover td {{ background:{'#FEF3C7' if st.session_state.theme == 'Light' else '#1E2A44'}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Theme-aware Plotly layout — every text element has explicit color so light
# mode never inherits dark text on white (and vice versa).
PLOTLY_LAYOUT = dict(
    template="plotly_white" if st.session_state.theme == "Light" else "plotly_dark",
    font=dict(family="Inter, sans-serif", color=T['ink'], size=12),
    paper_bgcolor=T['bg'], plot_bgcolor=T['bg'],
    margin=dict(l=10, r=10, t=70, b=60),
    title=dict(font=dict(color=T['ink'], size=15), x=0, xanchor="left", pad=dict(b=18)),
    legend=dict(font=dict(color=T['ink'])),
    xaxis=dict(
        showgrid=True, gridcolor=T['grid'], zeroline=False,
        linecolor=T['border'], tickfont=dict(color=T['ink']),
        title_font=dict(size=12, color=T['ink']),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=T['grid'], zeroline=False,
        linecolor=T['border'], tickfont=dict(color=T['ink']),
        title_font=dict(size=12, color=T['ink']),
    ),
)

NAVY, NAVY_2, ACCENT, LK, MY, INK, MUTED, LINE = (
    T['navy'], T['navy2'], T['accent'], T['lk'], T['my'], T['ink'], T['muted'], T['line'])

# ──────────────────────────────────────────────────────────────────────
# Auth state
# ──────────────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "show_login" not in st.session_state:
    st.session_state.show_login = False


def render_topbar():
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="mark">{BRAND_MARK}</div>
            <div>
              <div class="name">{BRAND_NAME}</div>
              <div class="tag">Risk · Research · Real-Time Macro Intelligence</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def theme_toggle(key_suffix: str = ""):
    cols = st.columns([6, 1])
    with cols[1]:
        new = st.selectbox(
            "Theme", ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            key=f"theme_select_{key_suffix}", label_visibility="collapsed",
        )
        if new != st.session_state.theme:
            st.session_state.theme = new
            st.rerun()


def render_landing():
    render_topbar()
    theme_toggle("landing")
    st.markdown(
        f"""
        <div class="hero">
          <div class="sub">Financial Risk Prediction System</div>
          <h1>Sri Lanka × Malaysia<br/>Sovereign Risk Intelligence</h1>
          <p>A real-time analytical platform that streams live macroeconomic data from
          the World Bank and the European Central Bank, generates 7-day forecasts using
          four production-grade models, and benchmarks sovereign-crisis probability
          across two emerging-market economies — built for analysts, researchers and
          policy desks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, title, body in [
        (c1, "Live Data Pipeline",
         "Every chart and metric is pulled live from the World Bank Open Data API and the European Central Bank — no static datasets."),
        (c2, "7-Day Forecast Engine",
         "Linear Regression, Random Forest, Prophet and ARIMA produce 7-day projections with R² and RMSE validation on a hold-out split."),
        (c3, "Sovereign Risk Score",
         "A composite low / medium / high signal derived from forecast volatility and projected change against historical baselines."),
    ]:
        with col:
            st.markdown(f"<div class='feat'><h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)

    st.write("")
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown("---")
        if not st.session_state.show_login:
            if st.button("🔐  Login to access dashboard", use_container_width=True):
                st.session_state.show_login = True
                st.rerun()
        else:
            tab_in, tab_up = st.tabs(["Sign in", "Create account"])
            with tab_in:
                with st.form("signin"):
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    if st.form_submit_button("Sign in", use_container_width=True):
                        ok, info = sign_in(u, p)
                        if ok:
                            st.session_state.user = info
                            st.rerun()
                        else:
                            st.error(info)
            with tab_up:
                with st.form("signup"):
                    fn = st.text_input("Full name")
                    em = st.text_input("Email")
                    u2 = st.text_input("Username", key="su_u")
                    p2 = st.text_input("Password (≥6 chars)", type="password", key="su_p")
                    if st.form_submit_button("Create account", use_container_width=True):
                        ok, msg = sign_up(u2, fn, em, p2)
                        (st.success if ok else st.error)(msg)


if not st.session_state.user:
    render_landing()
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# Authenticated app
# ──────────────────────────────────────────────────────────────────────
user = st.session_state.user
render_topbar()

with st.sidebar:
    st.markdown(
        f"<div style='font-family:Source Serif 4,serif;color:{T['accent']};"
        f"font-weight:700;font-size:1.35rem;'>{BRAND_MARK} {BRAND_NAME}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Risk Intelligence Workspace")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Model Predictions",
         "Country Comparison", "Forecast & Risk", "Report Download"],
        label_visibility="collapsed",
    )

    st.markdown("### Appearance")
    new_theme = st.radio("Theme", ["Light", "Dark"],
                        index=0 if st.session_state.theme == "Light" else 1,
                        horizontal=True, key="theme_radio")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("### Workspace")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    iso3 = COUNTRIES[country_name]["iso3"]

    indicator = st.selectbox(
        "Indicator",
        [EXCHANGE_RATE_KEY] + list(INDICATORS.keys()),
        index=1,
    )
    freq_label = st.radio("History granularity", ["Daily", "Monthly", "Yearly"],
                          horizontal=True, index=2)
    freq_code = {"Daily": "D", "Monthly": "M", "Yearly": "Y"}[freq_label]

    st.markdown("### Account")
    st.caption(f"{user['full_name']}\n\n{user['email']}")
    if st.button("Sign out", use_container_width=True):
        st.session_state.user = None
        st.session_state.show_login = False
        st.rerun()


def kpi(label, value, klass=""):
    st.markdown(f"<div class='kpi {klass}'><h4>{label}</h4><div class='v'>{value}</div></div>", unsafe_allow_html=True)


def fmt(v):
    if v is None or pd.isna(v): return "—"
    a = abs(v)
    if a >= 1e12: return f"{v/1e12:,.2f} T"
    if a >= 1e9:  return f"{v/1e9:,.2f} B"
    if a >= 1e6:  return f"{v/1e6:,.2f} M"
    if a >= 1e3:  return f"{v:,.2f}"
    return f"{v:,.4f}"


def page_header(title, sub):
    st.markdown(f"<div class='page-h'><h1>{title}</h1><p>{sub}</p></div>", unsafe_allow_html=True)


def section_heading(text: str):
    st.markdown(f"<div class='chart-heading'>{text}</div>", unsafe_allow_html=True)


def render_table(df: pd.DataFrame):
    """Render a DataFrame as a themed HTML table (CSS-styleable in light mode,
    unlike st.dataframe which uses a canvas grid that ignores CSS)."""
    df = df.copy()
    # Pretty-print numerics; keep strings as-is
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.4f}")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].map(lambda v: "—" if pd.isna(v) else f"{v:,}")
    html = df.to_html(index=False, classes="risk-table", border=0, escape=False)
    st.markdown(html, unsafe_allow_html=True)


def line_chart(traces, title, height=440, x_title="Date", y_title="Value"):
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["y"], name=t["name"], mode=t.get("mode", "lines"),
            line=dict(color=t["color"], width=t.get("width", 2.2),
                      dash=t.get("dash", "solid")),
            fill=t.get("fill"), fillcolor=t.get("fillcolor"),
            marker=t.get("marker"),
        ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=dict(text=title, font=dict(color=T['ink'], size=15),
                   x=0, xanchor="left", y=0.98, yanchor="top"),
        height=height + 40,
        margin=dict(l=10, r=10, t=110, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0,
                    font=dict(color=T['ink']), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────────────────────────────
if page == "Dashboard":
    page_header(
        f"Macro Dashboard — {country_name}",
        f"Live overview · {indicator} · {freq_label} granularity · {YEAR_MIN}–{YEAR_MAX}",
    )
    with st.spinner("Fetching real-time data..."):
        raw = load_indicator(iso3, indicator)

    if raw.empty:
        st.warning("No data returned by the live API.")
    else:
        series = filter_year_range(expand_to_frequency(raw, freq_code), YEAR_MIN, YEAR_MAX)
        if series.empty:
            st.warning("No observations in the 2000–2026 window.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Latest value", fmt(series['value'].iloc[-1]), "gold")
            with c2: kpi("Period high", fmt(series['value'].max()))
            with c3: kpi("Period low", fmt(series['value'].min()))
            with c4: kpi("Observations", f"{len(series):,}")

            line_chart(
                [{"x": series["date"], "y": series["value"], "name": indicator,
                  "color": LINE, "fill": "tozeroy",
                  "fillcolor": T['fillalpha']}],
                f"{indicator} — {country_name} ({freq_label})",
                x_title=x_label_for(freq_code),
                y_title=axis_label_for(indicator),
            )

            section_heading("Indicator coverage snapshot")
            with st.spinner("Loading indicator panel..."):
                panel = build_indicator_panel(iso3)
                panel = panel[(panel["year"] >= YEAR_MIN) & (panel["year"] <= YEAR_MAX)].tail(15)
                panel["year"] = panel["year"].astype(int).astype(str)
            render_table(panel)


elif page == "Model Predictions":
    page_header(
        f"Model Benchmark — {country_name}",
        f"Linear Regression · XGBoost · Neural Network — target: {indicator}",
    )
    if indicator in INDICATORS:
        target = indicator
    else:
        target = "Inflation (CPI %)"
        st.caption(f"Exchange Rate is not part of the regression panel — using **{target}** as the target.")

    if st.button("▶ Train all models", type="primary"):
        with st.spinner("Training Linear / XGBoost / Neural Network..."):
            try:
                X, y_reg, y_cls, feats, df = prepare_dataset(iso3, target)
                reg = benchmark_regressors(X, y_reg, feats)
                st.session_state["ml_results"] = {
                    "reg": reg, "feats": feats,
                    "target": target, "rows": len(df),
                }
            except Exception as e:
                st.error(f"Training failed: {e}")

    res = st.session_state.get("ml_results")
    if res and res.get("target") == target:
        st.caption(f"Training rows: {res['rows']} · Features: {len(res['feats'])}")

        section_heading("Regression metrics")
        cols = st.columns(3)
        for i, (name, m) in enumerate(res["reg"].items()):
            with cols[i]:
                st.markdown(f"<div class='kpi gold'><h4>{name}</h4>"
                            f"<div class='v'>R² {m['r2']:.3f}</div>"
                            f"<div style='color:{MUTED};font-size:0.8rem;'>RMSE {m['rmse']:.3f}</div>"
                            f"</div>", unsafe_allow_html=True)

        section_heading(f"Predicted vs Actual (test set) — {target}")
        fig = go.Figure()
        colors_map = {"Linear Regression": LINE, "XGBoost": ACCENT, "Neural Network": LK}
        for name, m in res["reg"].items():
            fig.add_trace(go.Scatter(
                x=m["y_true"], y=m["y_pred"], mode="markers", name=name,
                marker=dict(size=10, color=colors_map[name], opacity=0.85,
                            line=dict(color=T['bg'], width=1)),
            ))
        lo = min(min(m["y_true"].min() for m in res["reg"].values()),
                 min(m["y_pred"].min() for m in res["reg"].values()))
        hi = max(max(m["y_true"].max() for m in res["reg"].values()),
                 max(m["y_pred"].max() for m in res["reg"].values()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color=MUTED, dash="dash"), showlegend=False))
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.04, font=dict(color=T['ink'])),
        )
        fig.update_xaxes(title_text=f"Actual — {target}")
        fig.update_yaxes(title_text=f"Predicted — {target}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a target indicator and click **Train all models** to run the benchmark.")


elif page == "Country Comparison":
    page_header(
        "Sri Lanka 🇱🇰 vs Malaysia 🇲🇾",
        f"Side-by-side trajectory · {indicator} · {freq_label}",
    )
    lk = my = None
    with st.spinner("Loading both countries from the live World Bank API..."):
        try:
            lk_raw = load_indicator("LKA", indicator)
            my_raw = load_indicator("MYS", indicator)
            lk = filter_year_range(expand_to_frequency(lk_raw, freq_code), YEAR_MIN, YEAR_MAX)
            my = filter_year_range(expand_to_frequency(my_raw, freq_code), YEAR_MIN, YEAR_MAX)
        except Exception as e:
            st.error(f"Live data fetch failed: {e}. Please retry — the World Bank API "
                     f"sometimes throttles requests.")
            st.stop()

    if lk is None or my is None or lk.empty or my.empty:
        st.warning("Insufficient data for one of the countries — try another indicator "
                   "or refresh in a minute.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Sri Lanka — latest", fmt(lk['value'].iloc[-1]), "lk")
        with c2: kpi("Malaysia — latest", fmt(my['value'].iloc[-1]), "my")
        with c3: kpi("Δ (LK − MY)", fmt(lk['value'].iloc[-1] - my['value'].iloc[-1]))
        ratio = lk['value'].iloc[-1] / my['value'].iloc[-1] if my['value'].iloc[-1] else None
        with c4: kpi("Ratio LK / MY", f"{ratio:.3f}" if ratio else "—")

        line_chart(
            [
                {"x": lk["date"], "y": lk["value"], "name": "Sri Lanka", "color": LK, "width": 2.6},
                {"x": my["date"], "y": my["value"], "name": "Malaysia",  "color": MY, "width": 2.6},
            ],
            f"{indicator} — Sri Lanka vs Malaysia ({YEAR_MIN}–{YEAR_MAX})",
            height=480,
            x_title=x_label_for(freq_code),
            y_title=axis_label_for(indicator),
        )


elif page == "Forecast & Risk":
    page_header(
        f"7-Day Forecast & Sovereign Risk — {country_name}",
        f"Model-driven projection from today onwards (7 days) · {indicator}",
    )
    model_name = st.selectbox(
        "Forecast model",
        ["Linear Regression", "Random Forest", "Prophet", "ARIMA"],
        index=2,
    )
    with st.spinner("Building real-time 7-day forecast..."):
        raw = load_indicator(iso3, indicator)
        if raw.empty:
            st.warning("No data."); st.stop()
        daily = expand_to_frequency(raw, "D")
        result = forecast(daily, model_name, horizon=7)

    if "error" in result:
        st.error(result["error"]); st.stop()

    future = result["future"]
    level, score, color = risk_score(daily, future)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Model", model_name, "gold")
    with c2: kpi("R² (test)", f"{result['r2']:.3f}")
    with c3: kpi("RMSE (test)", f"{result['rmse']:,.4f}")
    with c4:
        st.markdown(f"<div class='kpi'><h4>Risk · next 7d</h4>"
                    f"<span class='pill' style='background:{color};'>{level}</span></div>",
                    unsafe_allow_html=True)

    today_ts = pd.Timestamp(pd.Timestamp.today().date())
    section_heading(f"7-Day Forecast — {indicator} (from {today_ts:%d %b %Y})")

    recent = daily[daily["date"] <= today_ts].tail(60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["value"], name="Historical",
                             mode="lines+markers",
                             line=dict(color=LINE, width=2.6),
                             marker=dict(size=4, color=LINE)))
    if "lower" in future.columns:
        fig.add_trace(go.Scatter(
            x=list(future["date"]) + list(future["date"][::-1]),
            y=list(future["upper"]) + list(future["lower"][::-1]),
            fill="toself", fillcolor="rgba(184,134,11,0.18)" if st.session_state.theme == "Light" else "rgba(224,184,74,0.22)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence band", hoverinfo="skip",
        ))
    if len(recent):
        fig.add_trace(go.Scatter(
            x=[recent["date"].iloc[-1], future["date"].iloc[0]],
            y=[recent["value"].iloc[-1], future["value"].iloc[0]],
            mode="lines", line=dict(color=ACCENT, width=2.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=future["date"], y=future["value"], name="7-day forecast",
                             mode="lines+markers",
                             line=dict(color=ACCENT, width=3.2, dash="dot"),
                             marker=dict(size=10, color=ACCENT,
                                         line=dict(color=T['bg'], width=1))))
    # No Plotly title here — the section_heading above provides spacing/separation
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, font=dict(color=T['ink'])),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=axis_label_for(indicator))
    st.plotly_chart(fig, use_container_width=True)

    section_heading("Forecast values (next 7 days from today)")
    table = future.copy()
    table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
    for c in ("value", "lower", "upper"):
        if c in table.columns:
            table[c] = table[c].round(4)
    render_table(table)


elif page == "Report Download":
    page_header(
        "Report Download",
        "Generate a publication-ready PDF for the current country and indicator.",
    )
    st.write(f"**Country:** {country_name}  ·  **Indicator:** {indicator}")
    if st.button("Build PDF report", type="primary"):
        with st.spinner("Compiling report from live data..."):
            raw = load_indicator(iso3, indicator)
            series = expand_to_frequency(raw, freq_code)
            daily = expand_to_frequency(raw, "D")
            r = forecast(daily, "Prophet", 7)
            level, score, color = risk_score(daily, r["future"]) if "future" in r else ("—", 0, "")

            # ── Build chart images for the PDF (via kaleido) ────────────────
            def _fig_to_png(fig) -> bytes:
                fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Helvetica", color="#0B1F3A", size=11),
                    margin=dict(l=50, r=20, t=50, b=50),
                    height=420, width=900,
                )
                try:
                    return fig.to_image(format="png", scale=2)
                except Exception:
                    return b""

            hist_fig = go.Figure()
            hist_fig.add_trace(go.Scatter(
                x=series["date"], y=series["value"], mode="lines",
                line=dict(color="#0B1F3A", width=2.4),
                fill="tozeroy", fillcolor="rgba(11,31,58,0.10)",
                name=indicator,
            ))
            hist_fig.update_layout(
                title=f"{indicator} — {country_name} ({freq_label})",
                xaxis_title=x_label_for(freq_code),
                yaxis_title=axis_label_for(indicator),
            )
            hist_png = _fig_to_png(hist_fig)

            fc_png = b""
            if "future" in r:
                fut = r["future"]
                recent = daily.tail(60)
                fc_fig = go.Figure()
                fc_fig.add_trace(go.Scatter(
                    x=recent["date"], y=recent["value"], name="Historical",
                    mode="lines", line=dict(color="#0B1F3A", width=2.2),
                ))
                if "lower" in fut.columns:
                    fc_fig.add_trace(go.Scatter(
                        x=list(fut["date"]) + list(fut["date"][::-1]),
                        y=list(fut["upper"]) + list(fut["lower"][::-1]),
                        fill="toself", fillcolor="rgba(184,134,11,0.18)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Confidence band", hoverinfo="skip",
                    ))
                fc_fig.add_trace(go.Scatter(
                    x=fut["date"], y=fut["value"], name="7-day forecast",
                    mode="lines+markers",
                    line=dict(color="#B8860B", width=3, dash="dot"),
                    marker=dict(size=8, color="#B8860B"),
                ))
                fc_fig.update_layout(
                    title=f"7-Day Forecast — {indicator}",
                    xaxis_title="Date", yaxis_title=axis_label_for(indicator),
                    legend=dict(orientation="h", y=1.12, x=0),
                )
                fc_png = _fig_to_png(fc_fig)

            sections = [
                {"title": "Executive Summary",
                 "paragraphs": [
                     f"This report covers <b>{indicator}</b> for <b>{country_name}</b>, "
                     f"using live data from the World Bank and the European Central Bank. "
                     f"All figures are generated at request time and reflect the latest "
                     f"observations available as of {datetime.utcnow():%d %B %Y}.",
                     f"The 7-day Prophet forecast classifies sovereign risk as "
                     f"<b>{level}</b> (composite score {score:.3f}).",
                 ],
                 "table": None},
                {"title": "Recent observations",
                 "paragraphs": [],
                 "images": [hist_png] if hist_png else [],
                 "table": [["Date", "Value"]] + [
                     [d.strftime("%Y-%m-%d"), f"{v:,.4f}"]
                     for d, v in series.tail(10).itertuples(index=False)
                 ]},
                {"title": "Forecast (next 7 days)",
                 "paragraphs": [
                     f"Test-set R² {r.get('r2', 0):.3f} · RMSE {r.get('rmse', 0):.4f}.",
                 ],
                 "images": [fc_png] if fc_png else [],
                 "table": [["Date", "Forecast"]] + [
                     [d.strftime("%Y-%m-%d"), f"{v:,.4f}"]
                     for d, v in r["future"][["date", "value"]].itertuples(index=False)
                 ]},
            ]
            pdf = build_report({"country": country_name, "indicator": indicator}, sections)

        st.success("Report generated.")
        st.download_button(
            "⬇ Download PDF",
            data=pdf,
            file_name=f"{BRAND_NAME}_{iso3}_{datetime.utcnow():%Y%m%d}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

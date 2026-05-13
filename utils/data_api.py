"""Real-time data fetchers for Sri Lanka & Malaysia only.
Sources: World Bank Open Data API (annual macro) + Frankfurter / ECB (daily FX).
No bundled datasets — every call hits the live API.
"""
from __future__ import annotations
import requests
import pandas as pd
import streamlit as st
from datetime import date, timedelta

WB_BASE = "https://api.worldbank.org/v2"
FX_BASE = "https://api.frankfurter.app"

# Project scope: Sri Lanka vs Malaysia only
COUNTRIES = {
    "Sri Lanka": {"iso3": "LKA", "iso2": "LK", "currency": "LKR", "flag": "🇱🇰"},
    "Malaysia":  {"iso3": "MYS", "iso2": "MY", "currency": "MYR", "flag": "🇲🇾"},
}

# Indicator code map (World Bank)
INDICATORS = {
    "GDP (current US$)":                              "NY.GDP.MKTP.CD",
    "Inflation (CPI %)":                              "FP.CPI.TOTL.ZG",
    "Infrastructure (Logistics Performance Index)":   "LP.LPI.INFR.XQ",
    "Real Interest Rate (%)":                         "FR.INR.RINR",
    "Total Natural Resources Rents (% GDP)":          "NY.GDP.TOTL.RT.ZS",
    "Unemployment Rate (%)":                          "SL.UEM.TOTL.ZS",
    "Wage & Salaried Workers (%)":                    "SL.EMP.WORK.ZS",
}
EXCHANGE_RATE_KEY = "Exchange Rate (USD)"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_wb_indicator(country_iso3: str, indicator_code: str) -> pd.DataFrame:
    """Fetch annual time-series from the World Bank API with retries."""
    url = f"{WB_BASE}/country/{country_iso3}/indicator/{indicator_code}"
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(
                url,
                params={"format": "json", "per_page": 20000},
                timeout=(10, 60),
            )
            r.raise_for_status()
            payload = r.json()
            if len(payload) < 2 or payload[1] is None:
                return pd.DataFrame(columns=["date", "value"])
            df = pd.DataFrame(payload[1]).dropna(subset=["value"])
            df["date"] = pd.to_datetime(df["date"] + "-12-31")
            return df[["date", "value"]].sort_values("date").reset_index(drop=True)
        except Exception as e:
            last_err = e
            continue
    # Final fallback: empty frame instead of crashing the app
    print(f"[wb] fetch failed for {country_iso3}/{indicator_code}: {last_err}")
    return pd.DataFrame(columns=["date", "value"])


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fx_daily(country_iso3: str, years_back: int = 25) -> pd.DataFrame:
    """Daily USD → local-currency rate. Frankfurter for MYR; World Bank annual fallback for LKR."""
    cur = next((c["currency"] for c in COUNTRIES.values() if c["iso3"] == country_iso3), None)
    if cur is None:
        return pd.DataFrame(columns=["date", "value"])
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    try:
        r = requests.get(f"{FX_BASE}/{start}..{end}", params={"from": "USD", "to": cur}, timeout=30)
        if r.status_code == 200:
            rates = r.json().get("rates", {})
            rows = [(pd.to_datetime(d), v[cur]) for d, v in rates.items() if cur in v]
            if rows:
                return pd.DataFrame(rows, columns=["date", "value"]).sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    # Fallback: World Bank official exchange rate (LCU per US$, annual)
    return fetch_wb_indicator(country_iso3, "PA.NUS.FCRF")


def filter_year_range(df: pd.DataFrame, start_year: int = 2000, end_year: int = 2026) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    m = (df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)
    return df.loc[m].reset_index(drop=True)


def load_indicator(iso3: str, label: str) -> pd.DataFrame:
    """Single entry-point that picks the right live source."""
    if label == EXCHANGE_RATE_KEY:
        return fetch_fx_daily(iso3)
    return fetch_wb_indicator(iso3, INDICATORS[label])


def expand_to_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample to D / M / Y. Annual data is interpolated and forward-filled to today."""
    if df.empty:
        return df
    s = df.set_index("date")["value"].sort_index()
    today = pd.Timestamp(date.today())
    if s.index.max() < today:
        s.loc[today] = s.iloc[-1]
        s = s.sort_index()
    if freq == "Y":
        out = s.resample("YE").last().dropna()
    elif freq == "M":
        out = s.resample("D").interpolate("linear").ffill().resample("ME").last().dropna()
    else:
        out = s.resample("D").interpolate("linear").ffill().dropna()
    return out.reset_index().rename(columns={"index": "date"})


def build_indicator_panel(iso3: str) -> pd.DataFrame:
    """Wide annual panel of every indicator for a country (used by ML page)."""
    frames = []
    for label, code in INDICATORS.items():
        df = fetch_wb_indicator(iso3, code).rename(columns={"value": label})
        df["year"] = df["date"].dt.year
        frames.append(df[["year", label]])
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on="year", how="outer")
    return panel.sort_values("year").reset_index(drop=True)

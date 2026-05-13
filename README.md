# SovereignIQ — Sri Lanka × Malaysia Risk Intelligence

A real-time sovereign-risk research dashboard built in Streamlit, inspired by the
layout of professional finance publications such as Risk.net.

## Features

- **Landing page** with project framing and a sign-in gate
- **Local SQLite + bcrypt authentication** (no third-party auth services)
- **Sidebar navigation** with 6 pages:
  - Dashboard
  - Economic Indicators
  - Model Predictions (Linear · XGBoost · Neural Net · Logistic crisis classifier)
  - Country Comparison (Sri Lanka vs Malaysia)
  - Forecast & Risk (7-day Prophet / ARIMA / RF / Linear forecast + risk pill)
  - Report Download (publication-ready PDF)
- **Real-time data only** — no bundled CSVs:
  - World Bank Open Data API for macro indicators
  - European Central Bank (via Frankfurter) for daily FX
- **7-day forecast horizon** computed from today's date on every refresh
- **Risk classifier** with Low / Medium / High labels

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

First-run users can create an account from the sign-in screen. Credentials are
stored in `users.db` (SQLite) using bcrypt-hashed passwords.

## Indicators tracked

GDP · Exchange Rate (USD) · Inflation · Infrastructure (LPI) · Real Interest Rate
· Total Natural Resources Rents · Unemployment Rate · Wage & Salaried Workers.

## Disclaimer

For analytical and educational purposes. Not investment advice.

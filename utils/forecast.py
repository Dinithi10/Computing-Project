"""Forecast models with R² and RMSE."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def _features(dates: pd.Series) -> np.ndarray:
    t = (dates - dates.min()).dt.days.values.reshape(-1, 1)
    return np.hstack([t, np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25)])


def _eval_split(y_true, y_pred):
    return float(r2_score(y_true, y_pred)), float(np.sqrt(mean_squared_error(y_true, y_pred)))


def forecast(df: pd.DataFrame, model_name: str, horizon: int = 7) -> dict:
    """df: columns=[date,value] daily. Returns dict with future, metrics, model name."""
    df = df.dropna().sort_values("date").reset_index(drop=True)
    if len(df) < 30:
        return {"error": "Not enough data points to forecast (need ≥ 30)."}

    # last 20% as test
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    today = pd.Timestamp(pd.Timestamp.today().date())
    future_dates = pd.date_range(today + pd.Timedelta(days=1), periods=horizon, freq="D")

    fitted_test = None
    future_values = None
    lower = upper = None

    if model_name == "Linear Regression":
        X_tr, y_tr = _features(train["date"]), train["value"].values
        X_te = _features(test["date"])
        m = LinearRegression().fit(X_tr, y_tr)
        fitted_test = m.predict(X_te)
        full = pd.concat([train, test])
        m2 = LinearRegression().fit(_features(full["date"]), full["value"].values)
        future_values = m2.predict(_features(pd.Series(future_dates)))

    elif model_name == "Random Forest":
        X_tr, y_tr = _features(train["date"]), train["value"].values
        m = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_tr, y_tr)
        fitted_test = m.predict(_features(test["date"]))
        full = pd.concat([train, test])
        m2 = RandomForestRegressor(n_estimators=400, random_state=42).fit(
            _features(full["date"]), full["value"].values
        )
        future_values = m2.predict(_features(pd.Series(future_dates)))

    elif model_name == "Prophet":
        from prophet import Prophet
        p = Prophet(daily_seasonality=False, yearly_seasonality=True)
        p.fit(train.rename(columns={"date": "ds", "value": "y"}))
        fitted_test = p.predict(test[["date"]].rename(columns={"date": "ds"}))["yhat"].values
        p2 = Prophet(daily_seasonality=False, yearly_seasonality=True)
        p2.fit(df.rename(columns={"date": "ds", "value": "y"}))
        fc = p2.predict(pd.DataFrame({"ds": future_dates}))
        future_values = fc["yhat"].values
        lower, upper = fc["yhat_lower"].values, fc["yhat_upper"].values

    elif model_name == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA
        m = ARIMA(train["value"].values, order=(2, 1, 2)).fit()
        fitted_test = m.forecast(steps=len(test))
        m2 = ARIMA(df["value"].values, order=(2, 1, 2)).fit()
        fc = m2.get_forecast(steps=horizon)
        future_values = fc.predicted_mean
        ci = fc.conf_int(alpha=0.2)
        lower, upper = ci[:, 0], ci[:, 1]
    else:
        return {"error": f"Unknown model {model_name}"}

    r2, rmse = _eval_split(test["value"].values, fitted_test)
    fut_df = pd.DataFrame({"date": future_dates, "value": future_values})
    if lower is not None:
        fut_df["lower"] = lower
        fut_df["upper"] = upper
    return {"model": model_name, "r2": r2, "rmse": rmse, "future": fut_df, "test": test, "fitted_test": fitted_test}


def risk_score(df: pd.DataFrame, future: pd.DataFrame) -> tuple[str, float, str]:
    """Risk based on volatility + projected change."""
    hist = df["value"].pct_change().dropna()
    vol = float(hist.std())
    proj_change = float((future["value"].iloc[-1] - df["value"].iloc[-1]) / abs(df["value"].iloc[-1] + 1e-9))
    score = abs(proj_change) + vol
    if score < 0.02:
        level, color = "Low", "#16a34a"
    elif score < 0.08:
        level, color = "Medium", "#f59e0b"
    else:
        level, color = "High", "#dc2626"
    return level, score, color

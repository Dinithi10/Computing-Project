"""Crisis-prediction & ML benchmarking for the Model Predictions page.
Trains 4 models on World Bank annual indicators per country and labels
each year as 'crisis' / 'no crisis' from inflation + unemployment + FX-stress proxies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, confusion_matrix,
)
from xgboost import XGBRegressor, XGBClassifier

from .data_api import build_indicator_panel, INDICATORS


def _label_crisis(panel: pd.DataFrame) -> pd.Series:
    """Heuristic crisis label: inflation > 10% OR unemployment > 8%."""
    infl = panel.get("Inflation (CPI %)", pd.Series(dtype=float))
    unemp = panel.get("Unemployment Rate (%)", pd.Series(dtype=float))
    return ((infl.fillna(0) > 10) | (unemp.fillna(0) > 8)).astype(int)


def prepare_dataset(iso3: str, target_label: str):
    """Returns (X, y_reg, y_cls, feature_names, panel)."""
    panel = build_indicator_panel(iso3).dropna(how="all")
    panel = panel.interpolate(limit_direction="both")
    if target_label not in panel.columns:
        raise ValueError(f"Target {target_label} unavailable for {iso3}")
    feats = [c for c in INDICATORS.keys() if c != target_label and c in panel.columns]
    df = panel.dropna(subset=feats + [target_label]).reset_index(drop=True)
    X = df[feats].values
    y_reg = df[target_label].values
    y_cls = _label_crisis(df).values
    return X, y_reg, y_cls, feats, df


def benchmark_regressors(X, y, feature_names):
    """Train Linear Regression, XGBoost, Neural Network on regression target."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    out = {}

    lr = LinearRegression().fit(Xtr, ytr)
    pr = lr.predict(Xte)
    out["Linear Regression"] = {
        "r2": r2_score(yte, pr),
        "rmse": float(np.sqrt(mean_squared_error(yte, pr))),
        "importance": dict(zip(feature_names, np.abs(lr.coef_))),
        "y_true": yte, "y_pred": pr,
    }

    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       random_state=42, verbosity=0).fit(Xtr, ytr)
    pr = xgb.predict(Xte)
    out["XGBoost"] = {
        "r2": r2_score(yte, pr),
        "rmse": float(np.sqrt(mean_squared_error(yte, pr))),
        "importance": dict(zip(feature_names, xgb.feature_importances_)),
        "y_true": yte, "y_pred": pr,
    }

    nn = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                      random_state=42).fit(Xtr_s, ytr)
    pr = nn.predict(Xte_s)
    # permutation-based importance
    base = mean_squared_error(yte, pr)
    imps = {}
    for i, name in enumerate(feature_names):
        Xp = Xte_s.copy()
        np.random.default_rng(0).shuffle(Xp[:, i])
        imps[name] = max(0.0, mean_squared_error(yte, nn.predict(Xp)) - base)
    out["Neural Network"] = {
        "r2": r2_score(yte, pr),
        "rmse": float(np.sqrt(mean_squared_error(yte, pr))),
        "importance": imps,
        "y_true": yte, "y_pred": pr,
    }
    return out


def crisis_classifier(X, y_cls, feature_names):
    """Logistic Regression: crisis (yes/no)."""
    if len(set(y_cls)) < 2:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y_cls, test_size=0.25, random_state=42, stratify=y_cls)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return {
        "accuracy": accuracy_score(yte, pred),
        "confusion": confusion_matrix(yte, pred, labels=[0, 1]),
        "importance": dict(zip(feature_names, np.abs(clf.coef_[0]))),
        "y_true": yte, "y_pred": pred,
    }

from __future__ import annotations
import os, warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class ImpactResult:
    """Result container for causal impact analysis."""
    summary: pd.DataFrame
    series: pd.DataFrame
    model_info: Dict[str, Any]

    def __repr__(self) -> str:
        return f"ImpactResult\n\n{self.summary.to_string(index=False)}"


# -----------------------
# Helpers
# -----------------------
def _ensure_datetime_index(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime")
    return df.sort_index()


def _ols_fit(y: pd.Series, X: pd.DataFrame):
    X_ = np.c_[np.ones(len(X)), X.values]
    beta = np.linalg.pinv(X_.T @ X_) @ (X_.T @ y.values)
    return beta


# -----------------------
# Public API
# -----------------------
def dataloader(
    data_path: str,
    *,
    date_col: Optional[str] = None,
    y_col: Optional[str] = None,
    control_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load CSV/Parquet and return time-indexed DataFrame."""
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)
    df = _ensure_datetime_index(df, date_col)

    if control_cols:
        keep = [c for c in control_cols if c in df.columns]
        if y_col and y_col not in keep:
            keep = [y_col] + keep
        df = df[keep]
    if y_col and y_col in df.columns:
        cols = [y_col] + [c for c in df.columns if c != y_col]
        df = df[cols]
    return df


def syn_generate(control_df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """Simple PCA via SVD to reduce controls."""
    Z = (control_df - control_df.mean()) / (control_df.std(ddof=1) + 1e-12)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    comps = Z @ Vt[:k].T
    cols = [f"PC{i+1}" for i in range(k)]
    return pd.DataFrame(comps, index=control_df.index, columns=cols)


def assumption_val(control_df: pd.DataFrame, y: pd.Series, timepoint=None) -> pd.DataFrame:
    """Basic diagnostics: missingness, correlation, collinearity."""
    df = control_df.join(y.rename("y"))
    rows = [{"check": "missing_rate_y", "value": df["y"].isna().mean()}]
    for c in control_df.columns:
        rows.append({"check": f"missing_rate_{c}", "value": df[c].isna().mean()})
    if timepoint:
        pre = df[df.index <= pd.to_datetime(timepoint)]
        corr = pre.drop("y", axis=1).corrwith(pre["y"])
    else:
        corr = df.drop("y", axis=1).corrwith(df["y"])
    for c, v in corr.items():
        rows.append({"check": f"corr_y_{c}", "value": v})
    return pd.DataFrame(rows)


def causal_impact(y: pd.Series, X: pd.DataFrame, timepoint) -> ImpactResult:
    """Estimate impact using pycausalimpact if available, else fallback OLS."""
    try:
        from causalimpact import CausalImpact
        df = pd.concat([y, X], axis=1)
        pre = [df.index.min(), pd.to_datetime(timepoint)]
        post = [pd.to_datetime(timepoint) + pd.Timedelta(1, "s"), df.index.max()]
        ci = CausalImpact(df, pre, post)
        summary = pd.DataFrame({
            "average_effect": [ci.summary_data.loc["post", "average"]],
            "p_value": [ci.p_value],
        })
        return ImpactResult(summary, ci.inferences, {"backend": "pycausalimpact"})
    except Exception:
        warnings.warn("Falling back to OLS.")
        cutoff = pd.to_datetime(timepoint)
        beta = _ols_fit(y[y.index <= cutoff], X[X.index <= cutoff])
        yhat = np.c_[np.ones(len(X)), X.values] @ beta
        eff = y.values - yhat
        series = pd.DataFrame({"y": y, "yhat": yhat, "point_effect": eff}, index=y.index)
        summary = pd.DataFrame({"average_effect": [series.loc[y.index > cutoff, "point_effect"].mean()]})
        return ImpactResult(summary, series, {"backend": "ols"})


def casual_impact(y: pd.Series, X: pd.DataFrame, timepoint):
    return causal_impact(y, X, timepoint)


def run_demo(n: int = 200, k: int = 5, seed: int = 42, treatment_shift: float = 1.0):
    """Generate synthetic data and run full pipeline."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    X = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=[f"x{i}" for i in range(k)])
    y = 0.5 * X["x0"] - 0.3 * X["x1"] + rng.normal(size=n)
    tp = idx[n // 2]
    y[idx > tp] += treatment_shift
    summary, series, model = causal_impact(y, syn_generate(X), tp).summary, causal_impact(y, syn_generate(X), tp).series, causal_impact(y, syn_generate(X), tp).model_info
    return summary, series, model

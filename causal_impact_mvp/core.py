from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd


# -----------------------
# Utilities / result type
# -----------------------
@dataclass
class ImpactResult:
    """Container for causal impact results."""
    summary: pd.DataFrame         # one-row summary
    series: pd.DataFrame          # pointwise & cumulative effects (post period)
    model_info: Dict[str, Any]    # backend + diagnostics

    def __repr__(self) -> str:
        s = self.summary.to_string(index=False)
        return f"ImpactResult\n\n{s}\n\nUse .series for pointwise effects and .model_info for details."


def _ensure_datetime_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    if date_col is not None and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or specify date_col to set it.")
    return df.sort_index()


def _split_pre_post(y: pd.Series, timepoint) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("y must be indexed by DatetimeIndex")
    cutoff = pd.to_datetime(timepoint)
    y_pre = y[y.index <= cutoff]
    y_post = y[y.index > cutoff]
    if len(y_pre) < 10 or len(y_post) < 1:
        warnings.warn("Very few observations in pre or post period; results may be unstable.")
    return y_pre, y_post


def _ols_fit_predict(y: pd.Series, X: pd.DataFrame):
    # Adds intercept, solves via pseudo-inverse
    X_ = np.c_[np.ones((len(X), 1)), X.values]
    y_ = y.values.reshape(-1, 1)
    beta = np.linalg.pinv(X_.T @ X_) @ (X_.T @ y_)
    yhat = (X_ @ beta).ravel()
    resid = y.values - yhat
    return yhat, resid, beta


# ---------------
# Public API (4x)
# ---------------

def dataloader(
    data_path: str,
    *,
    date_col: Optional[str] = None,
    y_col: Optional[str] = None,
    control_cols: Optional[Sequence[str]] = None,
    parse_dates: bool = True,
    infer_datetime_format: bool = True,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Load CSV or Parquet into a time-indexed DataFrame.

    Parameters
    ----------
    data_path : str
        Path to CSV/Parquet file.
    date_col : str, optional
        Column to be parsed as datetime and set as index.
    y_col : str, optional
        Name of outcome column; moved to first column if provided.
    control_cols : list[str], optional
        Subset of control columns to keep (others dropped). If None, keep all.
    """
    lower = data_path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(
            data_path,
            parse_dates=[date_col] if (parse_dates and date_col) else None,
            infer_datetime_format=infer_datetime_format,
            **read_kwargs,
        )
    elif lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(data_path, **read_kwargs)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

    df = _ensure_datetime_index(df, date_col)

    if control_cols is not None:
        keep = [c for c in control_cols if c in df.columns]
        if y_col and y_col in df.columns and y_col not in keep:
            keep = [y_col] + keep
        df = df[keep]

    if y_col and y_col in df.columns:
        cols = [y_col] + [c for c in df.columns if c != y_col]
        df = df[cols]

    return df


def syn_generate(control_vars_df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Generate a low-dimensional synthetic control matrix X via PCA (NumPy/SVD).
    Returns a DataFrame with columns PC1..PCk and the same DatetimeIndex.

    Parameters
    ----------
    control_vars_df : DataFrame with DatetimeIndex (controls only)
    n_components : int, default 3
    """
    if not isinstance(control_vars_df.index, pd.DatetimeIndex):
        raise ValueError("control_vars_df must be time-indexed (DatetimeIndex)")
    if control_vars_df.shape[1] == 0:
        raise ValueError("control_vars_df has no columns.")

    # Standardize (z-score) without sklearn
    Z = (control_vars_df.values - control_vars_df.values.mean(axis=0)) / (
        control_vars_df.values.std(axis=0, ddof=1) + 1e-12
    )

    # PCA via SVD of standardized data
    # Z = U S V^T ; principal components = Z @ V_k
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    k = int(max(1, min(n_components, Vt.shape[0])))
    Vk = Vt[:k].T  # p x k
    comps = Z @ Vk  # n x k

    cols = [f"PC{i+1}" for i in range(k)]
    X = pd.DataFrame(comps, index=control_vars_df.index, columns=cols)
    # Store explained variance ratio as metadata
    var = (S ** 2) / (Z.shape[0] - 1)
    evr = (var / var.sum())[:k].tolist()
    X.attrs["explained_variance_ratio_"] = evr
    return X


def assumption_val(control_vars_df: pd.DataFrame, y: pd.Series, timepoint=None) -> pd.DataFrame:
    """
    Basic diagnostics for causal-impact assumptions.

    - Missingness of y and controls
    - Correlation of controls with y (pre-period if timepoint provided)
    - Collinearity indicator via condition number of controls
    - Pre/Post means of y if timepoint provided

    Returns
    -------
    DataFrame with rows of simple checks.
    """
    if not isinstance(control_vars_df.index, pd.DatetimeIndex) or not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("Inputs must be time-indexed (DatetimeIndex)")

    df = control_vars_df.join(y.rename("y"), how="inner")
    miss = df.isna().mean()

    # Pre-period slice for correlations (if timepoint provided)
    if timepoint is not None:
        cutoff = pd.to_datetime(timepoint)
        pre = df[df.index <= cutoff]
    else:
        pre = df

    corr = pre.drop(columns=["y"]).corrwith(pre["y"])

    # Collinearity via condition number
    X = pre.drop(columns=["y"]).values
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        cond_num = (s.max() / s.min()) if s.min() > 0 else np.inf
    except Exception:
        cond_num = np.inf

    rows = []
    rows.append({"check": "missing_rate_y", "value": float(miss.get("y", 0.0))})
    for c in control_vars_df.columns:
        rows.append({"check": f"missing_rate_{c}", "value": float(miss.get(c, 0.0))})
    for c, v in corr.items():
        rows.append({"check": f"corr_y_{c}", "value": float(v) if pd.notna(v) else np.nan})
    rows.append({"check": "condition_number_controls", "value": float(cond_num)})

    if timepoint is not None:
        y_pre, y_post = _split_pre_post(df["y"], timepoint)
        rows.append({"check": "y_pre_mean", "value": float(y_pre.mean())})
        rows.append({"check": "y_post_mean", "value": float(y_post.mean())})

    return pd.DataFrame(rows)


def causal_impact(y: pd.Series, X: pd.DataFrame, timepoint) -> ImpactResult:
    """
    Causal Impact–style analysis.

    Tries to use `pycausalimpact` if available (Bayesian structural time series).
    Otherwise falls back to a simple OLS-on-pre-period counterfactual forecast.

    Parameters
    ----------
    y : pd.Series with DatetimeIndex (outcome)
    X : pd.DataFrame with DatetimeIndex (controls, aligned to y)
    timepoint : timestamp-like cutoff between pre and post
    """
    if not isinstance(y.index, pd.DatetimeIndex) or not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("y and X must be time-indexed (DatetimeIndex)")

    data = X.join(y.rename("y"), how="inner").dropna()
    y = data["y"]
    X = data.drop(columns=["y"])

    # Try pycausalimpact if installed
    try:
        from causalimpact import CausalImpact  # type: ignore

        df_ci = pd.concat([y, X], axis=1)
        pre_period = [df_ci.index.min(), pd.to_datetime(timepoint)]
        post_period = [pd.to_datetime(timepoint) + pd.Timedelta(seconds=1), df_ci.index.max()]

        ci = CausalImpact(df_ci, pre_period, post_period)

        series = pd.DataFrame(
            {
                "y": y,
                "point_effect": ci.inferences["point_effect"],
                "point_effect_lower": ci.inferences["point_effect_lower"],
                "point_effect_upper": ci.inferences["point_effect_upper"],
                "cumulative_effect": ci.inferences["cumulative_effect"],
                "cumulative_effect_lower": ci.inferences["cumulative_effect_lower"],
                "cumulative_effect_upper": ci.inferences["cumulative_effect_upper"],
            },
            index=df_ci.index,
        )

        summary = pd.DataFrame(
            {
                "average_effect": [ci.summary_data.loc["post", "average"]],
                "average_effect_lower": [ci.summary_data.loc["post", "lower"]],
                "average_effect_upper": [ci.summary_data.loc["post", "upper"]],
                "p_value": [ci.p_value],
            }
        )

        return ImpactResult(summary=summary, series=series, model_info={"backend": "pycausalimpact"})

    except Exception as e:
        warnings.warn(
            "pycausalimpact not available or failed; falling back to simple OLS counterfactual.\n"
            f"Reason: {e}"
        )

    # OLS fallback
    cutoff = pd.to_datetime(timepoint)
    mask_pre = X.index <= cutoff
    X_pre, y_pre = X.loc[mask_pre], y.loc[mask_pre]
    X_post, y_post = X.loc[~mask_pre], y.loc[~mask_pre]

    # Fit on pre
    yhat_pre, resid_pre, beta = _ols_fit_predict(y_pre, X_pre)

    # Forecast post with fixed beta
    Xp_ = np.c_[np.ones((len(X_post), 1)), X_post.values]
    yhat_post = (Xp_ @ beta).ravel()

    # Residual std from pre period
    dof = max(1, len(y_pre) - (X_pre.shape[1] + 1))
    sigma = float(np.sqrt((resid_pre @ resid_pre) / dof)) if len(resid_pre) > 2 else float(np.std(resid_pre))
    z = 1.96

    idx_post = X_post.index
    point_eff = y_post.values - yhat_post
    se = np.full_like(point_eff, sigma, dtype=float)
    lower = point_eff - z * se
    upper = point_eff + z * se

    series = pd.DataFrame(
        {
            "y": y_post.values,
            "yhat": yhat_post,
            "point_effect": point_eff,
            "point_effect_lower": lower,
            "point_effect_upper": upper,
        },
        index=idx_post,
    )

    cumulative = series["point_effect"].cumsum()
    cum_se = np.sqrt(np.arange(1, len(series) + 1)) * sigma
    cum_lower = cumulative - z * cum_se
    cum_upper = cumulative + z * cum_se

    series["cumulative_effect"] = cumulative
    series["cumulative_effect_lower"] = cum_lower
    series["cumulative_effect_upper"] = cum_upper

    summary = pd.DataFrame(
        {
            "average_effect": [float(series["point_effect"].mean())],
            "average_effect_lower": [float(series["point_effect_lower"].mean())],
            "average_effect_upper": [float(series["point_effect_upper"].mean())],
            "p_value": [np.nan],  # not computed in fallback
        }
    )

    model_info = {"backend": "ols_fallback", "beta": beta.ravel().tolist(), "sigma": sigma}
    return ImpactResult(summary=summary, series=series, model_info=model_info)


# Typo-tolerant alias
def casual_impact(y: pd.Series, X: pd.DataFrame, timepoint):
    return causal_impact(y, X, timepoint)


# -----------------------
# Demo (generates test data)
# -----------------------
def run_demo(
    *,
    generate_data: bool = True,
    n: int = 220,
    k: int = 5,
    seed: int = 42,
    treatment_shift: float = 1.0,
    data_path: str | None = None,
    date_col: str = "date",
    y_col: str = "kpi",
    n_components: int = 3,
    out_dir: str = "./_demo_data",
    csv_name: str = "toy_metric.csv",
    save_outputs: bool = True,
):
    """
    Demo runner for the causal-impact MVP.

    Default (generate_data=True): synthesizes a dataset, writes CSV to out_dir,
    and runs PCA synthetic controls, diagnostics, and causal impact.
    Returns (summary_df, series_df, model_info).
    """
    os.makedirs(out_dir, exist_ok=True)

    if generate_data:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        X_raw = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=[f"x{i+1}" for i in range(k)])

        signal = 0.6 * X_raw["x1"] - 0.35 * X_raw["x2"] + 0.15 * X_raw["x3"]
        y = signal + rng.normal(scale=0.6, size=n)

        timepoint = idx[n // 2]
        y[idx > timepoint] = y[idx > timepoint] + treatment_shift

        df = pd.concat([y.rename(y_col), X_raw], axis=1)
        df.insert(0, date_col, df.index)
        csv_path = os.path.join(out_dir, csv_name)
        df.to_csv(csv_path, index=False)
    else:
        if not data_path:
            raise ValueError("When generate_data=False, provide data_path to a CSV.")
        csv_path = data_path
        # Infer midpoint timepoint from data for convenience
        tmp = dataloader(csv_path, date_col=date_col, y_col=y_col)
        timepoint = tmp.index[len(tmp) // 2]

    loaded = dataloader(csv_path, date_col=date_col, y_col=y_col)
    y_series = loaded[y_col]
    controls = loaded.drop(columns=[y_col])

    X = syn_generate(controls, n_components=n_components)
    # Diagnostics are computed but not returned separately to keep API small
    _ = assumption_val(X, y_series, timepoint=timepoint)

    res = causal_impact(y_series, X, timepoint=timepoint)

    if save_outputs:
        res.series.to_csv(os.path.join(out_dir, "impact_series.csv"))
        res.summary.to_csv(os.path.join(out_dir, "impact_summary.csv"), index=False)

    return res.summary, res.series, res.model_info


if __name__ == "__main__":
    s, ser, m = run_demo()
    print("=== IMPACT SUMMARY ===")
    print(s.to_string(index=False))
    print("\nFirst 10 post-period rows:")
    print(ser.head(10).to_string())
    print("\nBackend:", m.get("backend"))

# run_demo.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from causal_impact_mvp import (
    dataloader,
    syn_generate,
    assumption_val,
    causal_impact,
    casual_impact,  # alias demo
)

# ----------------------
# 1) make synthetic data
# ----------------------
def make_toy_data(n=220, k=5, seed=42, treatment_shift=1.0):
    """
    Creates a toy panel: controls X (k columns) and outcome y.
    Outcome is linear in a couple of controls + noise.
    After the midpoint time, we add a treatment effect (shift).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    # controls
    X = pd.DataFrame(rng.normal(size=(n, k)), index=idx, columns=[f"x{i+1}" for i in range(k)])

    # build outcome
    signal = 0.6 * X["x1"] - 0.35 * X["x2"] + 0.15 * X["x3"]
    y = signal + rng.normal(scale=0.6, size=n)

    # treatment effect after timepoint
    tp = idx[n // 2]
    y[idx > tp] = y[idx > tp] + treatment_shift

    df = pd.concat([y.rename("kpi"), X], axis=1)
    df.insert(0, "date", df.index)  # explicit date col for CSV
    return df, tp


def main():
    out_dir = "./_demo_data"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "toy_metric.csv")

    df, timepoint = make_toy_data()
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic data to: {csv_path}")
    print(f"Intervention timepoint: {timepoint}")

    # ----------------------
    # 2) load via dataloader
    # ----------------------
    loaded = dataloader(
        csv_path,
        date_col="date",
        y_col="kpi",
        # keep only some controls to mimic real life selection
        control_cols=["kpi", "x1", "x2", "x3", "x4", "x5"],
    )
    print("\nLoaded head():")
    print(loaded.head())

    # Split y and controls
    y = loaded["kpi"]
    controls = loaded.drop(columns=["kpi"])

    # ---------------------------------------
    # 3) build synthetic controls via PCA/SVD
    # ---------------------------------------
    X = syn_generate(controls, n_components=3)
    print("\nExplained variance ratio (first 3 PCs):", X.attrs.get("explained_variance_ratio_"))

    # -----------------------------
    # 4) run simple diagnostics
    # -----------------------------
    diag = assumption_val(X, y, timepoint=timepoint)
    print("\nDiagnostics (head):")
    print(diag.head(12).to_string(index=False))

    # -------------------------------------
    # 5) run causal impact end-to-end
    # -------------------------------------
    res = causal_impact(y, X, timepoint=timepoint)
    print("\n=== IMPACT SUMMARY ===")
    print(res)  # pretty __repr__

    print("\nPointwise effects (first 10 rows of post period):")
    print(res.series.head(10).to_string())

    # Alias demo (should be identical)
    res_alias = casual_impact(y, X, timepoint=timepoint)
    assert np.allclose(
        res.series["point_effect"].values,
        res_alias.series["point_effect"].values,
        equal_nan=True,
    ), "Alias mismatch"
    print("\nAlias check passed: `casual_impact` == `causal_impact`")

    # Optional: save outputs
    res.series.to_csv(os.path.join(out_dir, "impact_series.csv"))
    res.summary.to_csv(os.path.join(out_dir, "impact_summary.csv"), index=False)
    print(f"\nSaved outputs to {out_dir}/impact_series.csv and {out_dir}/impact_summary.csv")


if __name__ == "__main__":
    main()

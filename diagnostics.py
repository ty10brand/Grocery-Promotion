

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
PRED_PATH = Path("reports") / "predictions_test_xgb.csv"
DATA_PATH = Path("Grocery_data.csv")  # to get category labels
FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_ERR_CSV = Path("reports") / "diagnostics_error_summary.csv"

FIG_SCATTER = FIG_DIR / "09_actual_vs_pred_scatter.png"
FIG_ERR_WEEK = FIG_DIR / "10_error_by_week.png"
FIG_ERR_CAT = FIG_DIR / "11_error_by_category.png"


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Wrote: {path.resolve()}")


# -----------------------------
# Load predictions
# -----------------------------
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing {PRED_PATH}. Run train_xgb.py first.")

pred = pd.read_csv(PRED_PATH)

# Choose model prediction column
pred_col = "Pred_XGB"
if pred_col not in pred.columns:
    raise ValueError(f"Expected column {pred_col} not found in {PRED_PATH}.")

pred["error"] = pred[pred_col] - pred["Actual_Units"]
pred["abs_error"] = pred["error"].abs()
pred["sq_error"] = pred["error"] ** 2

# Attach categories (optional but recommended)
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)[["Product_ID", "Week", "Product_Category"]]
    pred = pred.merge(df, on=["Product_ID", "Week"], how="left")
else:
    pred["Product_Category"] = "Unknown"


# -----------------------------
# Summary table
# -----------------------------
summary = {
    "MAE": float(pred["abs_error"].mean()),
    "RMSE": float(np.sqrt(pred["sq_error"].mean())),
    "Mean_Error": float(pred["error"].mean()),
    "Median_Abs_Error": float(pred["abs_error"].median()),
    "P90_Abs_Error": float(pred["abs_error"].quantile(0.90)),
}
print("\nOverall diagnostics:")
for k, v in summary.items():
    print(f"  {k}: {v:,.3f}")

# Error by week and by category
err_by_week = pred.groupby("Week").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
    MeanError=("error", "mean"),
    N=("error", "size")
).reset_index()

err_by_cat = pred.groupby("Product_Category").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
    MeanError=("error", "mean"),
    N=("error", "size")
).reset_index().sort_values("MAE", ascending=False)

# Save combined summary
err_by_week.to_csv(Path("reports") / "error_by_week.csv", index=False)
err_by_cat.to_csv(Path("reports") / "error_by_category.csv", index=False)

pd.DataFrame([summary]).to_csv(OUT_ERR_CSV, index=False)
print(f"\n[CSV] Wrote: {OUT_ERR_CSV.resolve()}")
print(f"[CSV] Wrote: {(Path('reports') / 'error_by_week.csv').resolve()}")
print(f"[CSV] Wrote: {(Path('reports') / 'error_by_category.csv').resolve()}")


# -----------------------------
# Plots
# -----------------------------
# 1) Actual vs predicted scatter
plt.figure()
plt.scatter(pred["Actual_Units"], pred[pred_col], alpha=0.25)
plt.title("Actual vs Predicted Units (XGBoost, Weeks 43–52)")
plt.xlabel("Actual Units")
plt.ylabel("Predicted Units")
savefig(FIG_SCATTER)

# 2) Error by week
plt.figure()
plt.plot(err_by_week["Week"], err_by_week["MAE"], marker="o")
plt.title("MAE by Week (Weeks 43–52)")
plt.xlabel("Week")
plt.ylabel("MAE")
savefig(FIG_ERR_WEEK)

# 3) Error by category (top 10 worst)
top = err_by_cat.head(10).copy()
plt.figure()
plt.bar(top["Product_Category"].astype(str), top["MAE"].values)
plt.title("Top 10 Categories by MAE (Higher = Harder to Predict)")
plt.xlabel("Category")
plt.ylabel("MAE")
plt.xticks(rotation=30, ha="right")
savefig(FIG_ERR_CAT)


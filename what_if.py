

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import load


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("Grocery_data.csv")
MODEL_PATH = Path("reports") / "xgb_pipeline.joblib"
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = REPORT_DIR / "what_if_results.csv"
OUT_FIG = FIG_DIR / "08_what_if_lift_curve.png"

# Pick a default scenario (you can change these)
DEFAULT_PRODUCT_ID = 150
DEFAULT_WEEK = 52

# Discount scenarios to test (0 means no promo)
DISCOUNT_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20]


# -----------------------------
# Load data + model
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing {DATA_PATH}. Put Grocery_data.csv in the repo root.")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing {MODEL_PATH}. Run train_xgb.py first to create reports/xgb_pipeline.joblib."
    )

df = pd.read_csv(DATA_PATH).sort_values(["Product_ID", "Week"]).reset_index(drop=True)
pipe = load(MODEL_PATH)


# -----------------------------
# Feature engineering (must match training)
# -----------------------------
TARGET = "Total_Sales_Volume"

df["discount_pct"] = ((df["Price"] - df["Promoted_Price"]) / df["Price"]).clip(lower=0).fillna(0)
df["week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)

df["sales_lag_1"] = df.groupby("Product_ID")[TARGET].shift(1)
df["sales_lag_2"] = df.groupby("Product_ID")[TARGET].shift(2)
df["sales_roll4_mean"] = (
    df.groupby("Product_ID")[TARGET]
      .shift(1)
      .rolling(4, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

X_COLS = [
    "Product_ID",
    "Product_Category",
    "Week",
    "Promotion_Flag",
    "Price",
    "Promoted_Price",
    "discount_pct",
    "Store_Traffic",
    "Promoted_Complementary_Sales",
    "Non_Promoted_Complementary_Sales",
    "Promoted_Substitute_Sales",
    "Non_Promoted_Substitute_Sales",
    "Promoted_Unrelated_Sales",
    "Non_Promoted_Unrelated_Sales",
    "Store_Profit",
    "Weekday_Indicator",
    "week_sin",
    "week_cos",
    "sales_lag_1",
    "sales_lag_2",
    "sales_roll4_mean",
]


def pick_base_row(product_id: int, week: int) -> pd.Series:
    """Pick a realistic row to simulate from. If exact week not found, fallback to nearest available week."""
    sub = df[df["Product_ID"] == product_id].copy()
    if sub.empty:
        raise ValueError(f"Product_ID {product_id} not found in dataset.")

    if (sub["Week"] == week).any():
        row = sub.loc[sub["Week"] == week].iloc[0]
        return row

    # fallback: nearest week for that product
    sub["week_dist"] = (sub["Week"] - week).abs()
    row = sub.sort_values("week_dist").iloc[0]
    return row


def simulate_row(base_row: pd.Series, discount: float) -> dict:
    """
    Create a scenario row by adjusting promo variables.
    discount = 0.0 => no promo
    discount > 0 => promo with promoted price = price*(1-discount)
    """
    price = float(base_row["Price"])
    promo_flag = 1 if discount > 0 else 0
    promoted_price = price * (1 - discount) if promo_flag == 1 else price

    scenario = base_row.copy()
    scenario["Promotion_Flag"] = promo_flag
    scenario["Promoted_Price"] = promoted_price
    scenario["discount_pct"] = discount

    # NOTE: the dataset has separate columns for promoted/non-promoted related sales.
    # In a true what-if engine, we'd model those too. Here we hold them constant from the base row.
    return scenario.to_dict()


# -----------------------------
# Run scenario
# -----------------------------
product_id = DEFAULT_PRODUCT_ID
week = DEFAULT_WEEK

base_row = pick_base_row(product_id, week)
base_dict = base_row.to_dict()

rows = []
for d in DISCOUNT_LEVELS:
    scenario_dict = simulate_row(base_row, d)
    X_one = pd.DataFrame([scenario_dict])[X_COLS]

    # model outputs log1p(units) then we invert
    pred_log = pipe.predict(X_one)[0]
    pred_units = float(np.expm1(pred_log))

    rows.append({
        "Product_ID": int(product_id),
        "Week": int(base_row["Week"]),  # actual week used (may be nearest fallback)
        "Scenario_DiscountPct": d,
        "Scenario_PromoFlag": int(1 if d > 0 else 0),
        "Price": float(base_dict["Price"]),
        "Promoted_Price": float(scenario_dict["Promoted_Price"]),
        "Pred_Units": pred_units
    })

out = pd.DataFrame(rows).sort_values("Scenario_DiscountPct")

# Compute lift vs no-promo scenario
baseline_units = float(out.loc[out["Scenario_DiscountPct"] == 0.0, "Pred_Units"].iloc[0])
out["Lift_Units_vs_NoPromo"] = out["Pred_Units"] - baseline_units
out["Lift_Pct_vs_NoPromo"] = np.where(
    baseline_units > 0,
    (out["Pred_Units"] - baseline_units) / baseline_units * 100,
    np.nan
)

out.to_csv(OUT_CSV, index=False)
print(f"[CSV] Wrote: {OUT_CSV.resolve()}")

# Plot lift curve
plt.figure()
plt.plot(out["Scenario_DiscountPct"] * 100, out["Pred_Units"], marker="o")
plt.title(f"What-If: Predicted Units vs Discount Depth (Product {product_id})")
plt.xlabel("Discount (%)")
plt.ylabel("Predicted Weekly Units")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
plt.close()
print(f"[FIG] Wrote: {OUT_FIG.resolve()}")

print("\nPreview:")
print(out[["Scenario_DiscountPct", "Pred_Units", "Lift_Units_vs_NoPromo", "Lift_Pct_vs_NoPromo"]])



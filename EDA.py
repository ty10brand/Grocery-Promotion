

"""
EDA.py — Grocery Promotion Data
Exploratory data analysis + charts saved to reports/figures/

Run:
  python EDA.py

Expected file:
  Grocery_data.csv (in the same folder as this script)

Outputs:
  reports/figures/*.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("Grocery_data.csv")
FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 50)


# -----------------------------
# Helpers
# -----------------------------
def savefig(name: str) -> None:
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {out}")


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# -----------------------------
# Load
# -----------------------------
print_header("LOAD DATA")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Put Grocery_data.csv in the same folder as EDA.py, or edit DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded {DATA_PATH} | shape={df.shape}")
print(df.head(5))

# -----------------------------
# Basic checks
# -----------------------------
print_header("BASIC QUALITY CHECKS")
print("Columns:")
print(df.columns.tolist())

missing = df.isna().mean().sort_values(ascending=False)
print("\nMissingness (fraction):")
print(missing[missing > 0].head(30) if (missing > 0).any() else "No missing values ✅")

print("\nDtypes:")
print(df.dtypes)

print("\nUnique counts:")
for c in ["Product_ID", "Product_Category", "Week", "Promotion_Flag", "Weekday_Indicator"]:
    if c in df.columns:
        print(f"  {c}: {df[c].nunique()} unique")

# Expected row count sanity: 200 products x 52 weeks = 10400
print("\nSanity checks:")
if df["Product_ID"].nunique() == 200 and df["Week"].nunique() == 52 and df.shape[0] == 200 * 52:
    print("  Looks like 200 products x 52 weeks ✅")
else:
    print("  Row/product/week counts differ from 200x52 (may still be fine).")


# -----------------------------
# Feature engineering for EDA
# -----------------------------
print_header("DERIVED FEATURES")
df["discount_pct"] = (df["Price"] - df["Promoted_Price"]) / df["Price"]
df["discount_pct"] = df["discount_pct"].clip(lower=0).fillna(0)

# Promo bucket for lift curves
bins = [-0.001, 0.00, 0.05, 0.10, 0.15, 0.20, 1.0]
labels = ["0%", "0–5%", "5–10%", "10–15%", "15–20%", "20%+"]
df["discount_bucket"] = pd.cut(df["discount_pct"], bins=bins, labels=labels)

print(df[["Promotion_Flag", "Price", "Promoted_Price", "discount_pct", "discount_bucket"]].head(8))


# -----------------------------
# Weekly constant checks (traffic/profit constant per week)
# -----------------------------
print_header("WEEKLY CONSTANT CHECKS (Traffic / Profit)")
weekly_traffic_var = df.groupby("Week")["Store_Traffic"].nunique().describe()
weekly_profit_var = df.groupby("Week")["Store_Profit"].nunique().describe()

print("Store_Traffic unique-count per week (should usually be 1 if constant within a week):")
print(weekly_traffic_var)

print("\nStore_Profit unique-count per week (should usually be 1 if constant within a week):")
print(weekly_profit_var)

# If you want to flag weeks that violate the constant assumption:
traffic_bad = df.groupby("Week")["Store_Traffic"].nunique()
profit_bad = df.groupby("Week")["Store_Profit"].nunique()
bad_weeks = sorted(set(traffic_bad[traffic_bad > 1].index).union(set(profit_bad[profit_bad > 1].index)))
print(f"\nWeeks with >1 unique Store_Traffic or Store_Profit value: {bad_weeks if bad_weeks else 'None ✅'}")


# -----------------------------
# Summary tables: promo vs non-promo
# -----------------------------
print_header("PROMO IMPACT SUMMARY (Units/Revenue)")
summary = (
    df.groupby("Promotion_Flag")[["Total_Sales_Volume", "Total_Sales_Revenue", "discount_pct"]]
      .agg(["mean", "median", "std"])
)
print(summary)

# Lift in units (mean)
promo_mean = df.loc[df["Promotion_Flag"] == 1, "Total_Sales_Volume"].mean()
nonpromo_mean = df.loc[df["Promotion_Flag"] == 0, "Total_Sales_Volume"].mean()
lift_pct = (promo_mean - nonpromo_mean) / nonpromo_mean * 100
print(f"\nMean units non-promo: {nonpromo_mean:,.1f}")
print(f"Mean units promo    : {promo_mean:,.1f}")
print(f"Estimated promo lift: {lift_pct:.2f}% (simple mean difference)")

# Promo lift by category
lift_by_cat = (
    df.groupby(["Product_Category", "Promotion_Flag"])["Total_Sales_Volume"]
      .mean()
      .unstack("Promotion_Flag")
      .rename(columns={0: "mean_units_nonpromo", 1: "mean_units_promo"})
)
lift_by_cat["lift_pct"] = (lift_by_cat["mean_units_promo"] - lift_by_cat["mean_units_nonpromo"]) / lift_by_cat["mean_units_nonpromo"] * 100
lift_by_cat = lift_by_cat.sort_values("lift_pct", ascending=False)
print_header("PROMO LIFT BY CATEGORY (Mean Units)")
print(lift_by_cat.round(2))


# -----------------------------
# Charts
# -----------------------------
print_header("CHARTS (saved to reports/figures/)")

# 1) Distribution of sales volume by promo flag
plt.figure()
for flag, sub in df.groupby("Promotion_Flag"):
    plt.hist(sub["Total_Sales_Volume"], bins=40, alpha=0.6, label=f"Promo={flag}")
plt.title("Total Sales Volume Distribution: Promo vs Non-Promo")
plt.xlabel("Weekly Units Sold")
plt.ylabel("Count")
plt.legend()
savefig("01_sales_volume_distribution_promo_vs_nonpromo.png")

# 2) Average weekly units by week (overall) and promo share
weekly = df.groupby("Week").agg(
    mean_units=("Total_Sales_Volume", "mean"),
    promo_rate=("Promotion_Flag", "mean"),
    mean_traffic=("Store_Traffic", "mean")
).reset_index()

plt.figure()
plt.plot(weekly["Week"], weekly["mean_units"])
plt.title("Average Weekly Units Sold (All Products)")
plt.xlabel("Week")
plt.ylabel("Mean Units")
savefig("02_weekly_mean_units.png")

plt.figure()
plt.plot(weekly["Week"], weekly["promo_rate"])
plt.title("Promotion Rate by Week (Share of product-weeks on promo)")
plt.xlabel("Week")
plt.ylabel("Promo Rate")
savefig("03_weekly_promo_rate.png")

# 3) Discount depth vs mean units (promo only)
promo = df[df["Promotion_Flag"] == 1].copy()
bucket_units = promo.groupby("discount_bucket")["Total_Sales_Volume"].mean().reindex(labels)

plt.figure()
plt.bar(bucket_units.index.astype(str), bucket_units.values)
plt.title("Promo Discount Depth vs Mean Units (Promo Weeks Only)")
plt.xlabel("Discount Bucket")
plt.ylabel("Mean Units")
plt.xticks(rotation=30, ha="right")
savefig("04_discount_bucket_vs_mean_units.png")

# 4) Category lift plot (top to bottom)
plt.figure()
plt.bar(lift_by_cat.index.astype(str), lift_by_cat["lift_pct"].values)
plt.title("Promo Lift (%) by Product Category (Mean Units)")
plt.xlabel("Category")
plt.ylabel("Lift (%)")
plt.xticks(rotation=30, ha="right")
savefig("05_promo_lift_by_category.png")

# 5) Complement / Substitute / Unrelated response during promo vs non-promo (means)
effects = pd.DataFrame({
    "Complementary": [
        df.loc[df["Promotion_Flag"] == 0, "Non_Promoted_Complementary_Sales"].mean(),
        df.loc[df["Promotion_Flag"] == 1, "Promoted_Complementary_Sales"].mean(),
    ],
    "Substitute": [
        df.loc[df["Promotion_Flag"] == 0, "Non_Promoted_Substitute_Sales"].mean(),
        df.loc[df["Promotion_Flag"] == 1, "Promoted_Substitute_Sales"].mean(),
    ],
    "Unrelated": [
        df.loc[df["Promotion_Flag"] == 0, "Non_Promoted_Unrelated_Sales"].mean(),
        df.loc[df["Promotion_Flag"] == 1, "Promoted_Unrelated_Sales"].mean(),
    ],
}, index=["Non-Promo Weeks", "Promo Weeks"])

plt.figure()
# simple grouped bar without specifying colors
x = np.arange(len(effects.index))
w = 0.25
plt.bar(x - w, effects["Complementary"].values, width=w, label="Complementary")
plt.bar(x,     effects["Substitute"].values,     width=w, label="Substitute")
plt.bar(x + w, effects["Unrelated"].values,      width=w, label="Unrelated")
plt.xticks(x, effects.index, rotation=0)
plt.title("Cross-Effects: Mean Related-Product Sales During Promo vs Non-Promo")
plt.ylabel("Mean Sales Volume (Related Products)")
plt.legend()
savefig("06_cross_effects_promo_vs_nonpromo.png")

# 6) Traffic vs units (scatter)
plt.figure()
plt.scatter(df["Store_Traffic"], df["Total_Sales_Volume"], alpha=0.25)
plt.title("Store Traffic vs Weekly Units Sold (All Rows)")
plt.xlabel("Store Traffic (weekly)")
plt.ylabel("Units Sold")
savefig("07_traffic_vs_units_scatter.png")

print_header("DONE")
print("Next step: use these EDA outputs in README, then build the forecasting model with lag features.")
print(f"Figures saved in: {FIG_DIR}")


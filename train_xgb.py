

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# XGBoost
from xgboost import XGBRegressor


DATA_PATH = "Grocery_data.csv"
TARGET = "Total_Sales_Volume"
TRAIN_WEEKS_MAX = 42
TEST_WEEKS_MIN = 43


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }


df = pd.read_csv(DATA_PATH).sort_values(["Product_ID", "Week"]).reset_index(drop=True)

# features
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

train_mask = df["Week"] <= TRAIN_WEEKS_MAX
test_mask = df["Week"] >= TEST_WEEKS_MIN

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

X_cols = [
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

X_train = train_df[X_cols].copy()
X_test = test_df[X_cols].copy()

# log target
y_train = np.log1p(train_df[TARGET].values)
y_test_units = test_df[TARGET].values

# preprocess
cat_cols = ["Product_Category", "Weekday_Indicator"]
num_cols = [c for c in X_cols if c not in cat_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, cat_cols),
        ("num", numeric_transformer, num_cols),
    ]
)

xgb = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", xgb)
])

pipe.fit(X_train, y_train)

# predict + invert log
pred_log = pipe.predict(X_test)
pred_units = np.expm1(pred_log)

# baselines (units space)
train_mean = train_df[TARGET].mean()
pred_mean = np.full(shape=len(y_test_units), fill_value=train_mean, dtype=float)

pred_last_week = X_test["sales_lag_1"].fillna(train_mean).values
pred_roll4 = X_test["sales_roll4_mean"].fillna(train_mean).values

rows = []
rows.append({"model": "Train mean", **metrics(y_test_units, pred_mean)})
rows.append({"model": "Last-week", **metrics(y_test_units, pred_last_week)})
rows.append({"model": "Roll4 mean", **metrics(y_test_units, pred_roll4)})
rows.append({"model": "XGBoost(log1p)", **metrics(y_test_units, pred_units)})

results_df = pd.DataFrame(rows).sort_values("RMSE")

print("\nRESULTS (Weeks 43–52 test)\n" + "-" * 70)
print(results_df.to_string(index=False))

# save outputs
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

results_path = REPORT_DIR / "model_results_xgb.csv"
results_df.to_csv(results_path, index=False)
print(f"\n[CSV] Wrote: {results_path.resolve()}")

pred_path = REPORT_DIR / "predictions_test_xgb.csv"
pd.DataFrame({
    "Product_ID": test_df["Product_ID"].values,
    "Week": test_df["Week"].values,
    "Actual_Units": y_test_units,
    "Pred_TrainMean": pred_mean,
    "Pred_LastWeek": pred_last_week,
    "Pred_Roll4Mean": pred_roll4,
    "Pred_XGB": pred_units,
}).to_csv(pred_path, index=False)
print(f"[CSV] Wrote: {pred_path.resolve()}")



from joblib import dump
model_path = REPORT_DIR / "xgb_pipeline.joblib"
dump(pipe, model_path)
print(f"[MODEL] Saved: {model_path.resolve()}")


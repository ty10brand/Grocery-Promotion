

# Grocery Promotion Data — Promotion Lift & Demand Modeling

This project analyzes weekly grocery sales for 200 products over 52 weeks and builds a first-pass demand model to understand how **promotions, discount depth, store traffic, and cross-product effects** relate to sales volume.

The dataset includes:
- Product-level weekly sales and revenue
- Promotion flag, regular price, promoted price (discounts)
- Weekly store traffic and store profit (week-level metrics)
- Related-product sales signals: complementary, substitute, unrelated (promo vs non-promo)

## Goals
1. **Promotion impact (lift):** quantify how promotions correlate with changes in unit sales.
2. **Discount response:** analyze how different discount depths relate to mean unit sales.
3. **Cross-effects:** examine how promotions relate to complementary / substitute / unrelated product sales.
4. **Baseline demand modeling:** train an initial ML model to predict weekly sales volume (and set up for improved forecasting).


Dataset Source

Kaggle: Grocery Promotion Data (weekly aggregated product sales + promotion indicators)

Link: https://www.kaggle.com/datasets/yadavbharti/grocery-promotion-data

License

This repo is for learning and portfolio purposes. Dataset license/terms are governed by Kaggle.


Key EDA Findings (Figures)
Promotions shift the sales distribution upward

Promoted weeks show a right-shift in unit sales, consistent with positive promotion lift.

Lift varies by category

Promotion responsiveness differs by product category, suggesting category-aware promo strategies.

Cross-effects: halo (complements) is strong

Complementary-related sales increase during promotions. Substitute-related sales also increase in this dataset, suggesting broad basket expansion effects or non-classical substitute definitions.

Discount depth is not monotonic

Larger discounts do not necessarily produce higher mean unit sales.

Modeling: Baselines vs XGBoost

We evaluated forecasting using a time split:

Train: Weeks 1–42

Test: Weeks 43–52

Results (test set):

Train mean baseline RMSE ≈ 1275 (best)

XGBoost(log1p) RMSE ≈ 1323

Roll4 mean baseline RMSE ≈ 1415

Last-week baseline RMSE ≈ 1767

Interpretation:

The dataset shows weak temporal continuity at the product-week level in the final 10 weeks.

A constant mean predictor is hard to beat, indicating high noise and/or synthetic generation patterns.

Despite limited forecasting accuracy, the trained model is still useful for what-if promotion simulation.

What-If Promotion Simulator (Decision Support)

The what-if tool answers:

“For a chosen product-week, what happens to predicted sales if we apply different discount depths?”

Example output (one product-week):

0% discount: ~2783 units

5% discount: ~3734 units (+34% lift)

10% discount: ~3530 units (+27% lift)

15% discount: ~3705 units (+33% lift)

20% discount: ~3495 units (+26% lift)

Diagnostics (How wrong are predictions?)

Diagnostics quantify typical error on Weeks 43–52:

MAE ≈ 1119 (average absolute miss ~1.1k units)

RMSE ≈ 1323 (penalizes big misses; indicates occasional large errors)

Mean Error ≈ -332 (model tends to underpredict on average)

90th percentile absolute error ≈ 2105 (10% of rows miss by >2.1k units)

Key charts:

What We Built (Portfolio Value)

This repo demonstrates a complete, real-world workflow:

Clean EDA with multiple business-facing charts

Time-based evaluation with baselines

Trained model pipeline saved to disk

A scenario simulation tool (what-if promotion planner)

Diagnostics and error breakdowns (week/category)


License / Notes

This repo is for learning and portfolio purposes. Dataset rights and terms are governed by Kaggle.



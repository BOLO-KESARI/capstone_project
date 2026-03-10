"""
Generate ML Model Result Visualizations for Capstone Paper
Produces publication-quality charts for all 12 models.
Output: generated_reports/ folder with PNG images.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, mean_absolute_percentage_error,
    mean_squared_error, r2_score, precision_recall_curve, auc
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Config ──
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "generated_reports")
os.makedirs(OUT, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
          "#00BCD4", "#FF5722", "#795548", "#607D8B", "#CDDC39"]

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved: {name}")


# ═══════════════════════════════════════════════════════
# 1. DEMAND FORECASTING (XGBoost / RF Regressor)
# ═══════════════════════════════════════════════════════
print("\n[1/11] Demand Forecasting Model...")
df_demand = pd.read_csv(os.path.join(BASE, "new_models/demand_forecasting_dataset.csv"))
features_demand = ["Month", "Year", "Historical_Passenger_Count", "Total_Flights_Operated",
                   "Cancellation_Rate", "Delay_Rate", "Weather_Disruption_Rate", "Seasonal_Index"]
X = df_demand[features_demand]
y = df_demand["Passenger_Demand"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

if HAS_XGB:
    model_demand = XGBRegressor(objective="reg:squarederror", n_estimators=100, verbosity=0)
else:
    model_demand = RandomForestRegressor(n_estimators=100, random_state=42)
model_demand.fit(X_tr, y_tr)
y_pred_demand = model_demand.predict(X_te)

r2_demand = r2_score(y_te, y_pred_demand)
mape_demand = mean_absolute_percentage_error(y_te, y_pred_demand) * 100
rmse_demand = np.sqrt(mean_squared_error(y_te, y_pred_demand))

# Fig 1: Actual vs Predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.scatter(y_te, y_pred_demand, alpha=0.6, c=COLORS[0], edgecolors="white", s=60)
lims = [min(y_te.min(), y_pred_demand.min()) - 50, max(y_te.max(), y_pred_demand.max()) + 50]
ax.plot(lims, lims, "--", color="red", lw=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Passenger Demand")
ax.set_ylabel("Predicted Passenger Demand")
ax.set_title(f"Demand Forecasting: Actual vs Predicted\nR² = {r2_demand:.4f}  |  MAPE = {mape_demand:.2f}%  |  RMSE = {rmse_demand:.1f}")
ax.legend()

# Fig 1b: Feature importance
if hasattr(model_demand, "feature_importances_"):
    imp = model_demand.feature_importances_
else:
    imp = np.ones(len(features_demand)) / len(features_demand)
idx = np.argsort(imp)
ax2 = axes[1]
ax2.barh([features_demand[i] for i in idx], imp[idx], color=COLORS[0], edgecolor="white")
ax2.set_xlabel("Feature Importance")
ax2.set_title("Demand Forecasting — Feature Importance")
fig.tight_layout()
save(fig, "01_demand_forecasting_results.png")


# ═══════════════════════════════════════════════════════
# 2. DYNAMIC PRICING (XGBoost / RF Regressor)
# ═══════════════════════════════════════════════════════
print("[2/11] Dynamic Pricing Model...")
df_price = pd.read_csv(os.path.join(BASE, "new_models/pricing_optimization_dataset.csv"))
features_price = ["Passenger_Demand", "Load_Factor", "Operating_Cost", "Fuel_Cost",
                  "Delay_Rate", "Cancellation_Rate", "Seasonal_Index"]
X = df_price[features_price]
y = df_price["Optimal_Price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

if HAS_XGB:
    model_price = XGBRegressor(objective="reg:squarederror", n_estimators=100, verbosity=0)
else:
    model_price = RandomForestRegressor(n_estimators=100, random_state=42)
model_price.fit(X_tr, y_tr)
y_pred_price = model_price.predict(X_te)

r2_price = r2_score(y_te, y_pred_price)
mape_price = mean_absolute_percentage_error(y_te, y_pred_price) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# scatter
ax = axes[0]
ax.scatter(y_te, y_pred_price, alpha=0.6, c=COLORS[1], edgecolors="white", s=60)
lims = [min(y_te.min(), y_pred_price.min()) - 200, max(y_te.max(), y_pred_price.max()) + 200]
ax.plot(lims, lims, "--", color="red", lw=1.5, label="Perfect Prediction")
ax.set_xlabel("Actual Optimal Price (₹)")
ax.set_ylabel("Predicted Optimal Price (₹)")
ax.set_title(f"Dynamic Pricing: Actual vs Predicted\nR² = {r2_price:.4f}  |  MAPE = {mape_price:.2f}%")
ax.legend()
# residual distribution
residuals = y_te.values - y_pred_price
ax2 = axes[1]
ax2.hist(residuals, bins=30, color=COLORS[1], edgecolor="white", alpha=0.8)
ax2.axvline(0, color="red", ls="--", lw=1.5)
ax2.set_xlabel("Residual (Actual − Predicted) ₹")
ax2.set_ylabel("Frequency")
ax2.set_title(f"Pricing Model — Residual Distribution\nMean = {residuals.mean():.1f}  |  Std = {residuals.std():.1f}")
fig.tight_layout()
save(fig, "02_dynamic_pricing_results.png")


# ═══════════════════════════════════════════════════════
# 3. OVERBOOKING OPTIMIZATION (Monte Carlo)
# ═══════════════════════════════════════════════════════
print("[3/11] Overbooking Optimization (Monte Carlo)...")
df_ob = pd.read_csv(os.path.join(BASE, "new_models/overbooking_dataset.csv"))
row = df_ob.iloc[0]
cap = int(row["Seat_Capacity"])
noshow_rate = float(row["No_Show_Rate"])
deny_cost = float(row["Compensation_Cost"])
empty_cost = float(row.get("Ticket_Price", 5000))

results_ob = []
for extra in range(25):
    total_booked = cap + extra
    denied_list, empty_list = [], []
    for _ in range(1000):
        shows = total_booked - np.random.binomial(total_booked, noshow_rate)
        denied = max(0, shows - cap)
        empty = max(0, cap - shows)
        denied_list.append(denied)
        empty_list.append(empty)
    avg_denied = np.mean(denied_list)
    avg_empty = np.mean(empty_list)
    total_cost = avg_denied * deny_cost + avg_empty * empty_cost
    results_ob.append({
        "extra_seats": extra, "avg_denied": avg_denied,
        "avg_empty": avg_empty, "denied_cost": avg_denied * deny_cost,
        "empty_cost": avg_empty * empty_cost, "total_cost": total_cost
    })
df_res = pd.DataFrame(results_ob)
optimal_k = df_res.loc[df_res["total_cost"].idxmin(), "extra_seats"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.plot(df_res["extra_seats"], df_res["denied_cost"], "o-", color=COLORS[3], label="Denied Boarding Cost", lw=2)
ax.plot(df_res["extra_seats"], df_res["empty_cost"], "s-", color=COLORS[0], label="Empty Seat Cost", lw=2)
ax.plot(df_res["extra_seats"], df_res["total_cost"], "^-", color=COLORS[4], label="Total Cost", lw=2.5)
ax.axvline(optimal_k, color="green", ls="--", lw=2, label=f"Optimal k* = {int(optimal_k)}")
ax.set_xlabel("Extra Seats Overbooked")
ax.set_ylabel("Expected Cost (₹)")
ax.set_title(f"Monte Carlo Overbooking Optimization (1,000 sims)\nCapacity = {cap}  |  No-Show Rate = {noshow_rate:.2%}")
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.bar(df_res["extra_seats"], df_res["avg_denied"], color=COLORS[3], alpha=0.7, label="Avg Denied Passengers")
ax2.bar(df_res["extra_seats"], -df_res["avg_empty"], color=COLORS[0], alpha=0.7, label="Avg Empty Seats (neg)")
ax2.axhline(0, color="black", lw=0.5)
ax2.axvline(optimal_k, color="green", ls="--", lw=2, label=f"Optimal k* = {int(optimal_k)}")
ax2.set_xlabel("Extra Seats Overbooked")
ax2.set_ylabel("Count")
ax2.set_title("Denied Passengers vs Empty Seats by Overbooking Level")
ax2.legend(fontsize=9)
fig.tight_layout()
save(fig, "03_overbooking_monte_carlo_results.png")


# ═══════════════════════════════════════════════════════
# 4. ROUTE PROFITABILITY (Linear Regression)
# ═══════════════════════════════════════════════════════
print("[4/11] Route Profitability Model...")
df_profit = pd.read_csv(os.path.join(BASE, "new_models/route_profitability_dataset.csv"))
features_profit = ["Passenger_Count", "Load_Factor", "Operating_Cost", "Flights_Per_Route", "Fuel_Cost"]
X = df_profit[features_profit]
y = df_profit["Route_Profit"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_profit = LinearRegression()
model_profit.fit(X_tr, y_tr)
y_pred_profit = model_profit.predict(X_te)
r2_profit = r2_score(y_te, y_pred_profit)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.scatter(y_te, y_pred_profit, alpha=0.6, c=COLORS[2], edgecolors="white", s=60)
lims = [min(y_te.min(), y_pred_profit.min()), max(y_te.max(), y_pred_profit.max())]
ax.plot(lims, lims, "--", color="red", lw=1.5)
ax.set_xlabel("Actual Route Profit (₹)")
ax.set_ylabel("Predicted Route Profit (₹)")
ax.set_title(f"Route Profitability: Actual vs Predicted\nR² = {r2_profit:.4f}")

# coefficient chart
coefs = model_profit.coef_
ax2 = axes[1]
colors_coef = [COLORS[1] if c > 0 else COLORS[3] for c in coefs]
ax2.barh(features_profit, coefs, color=colors_coef, edgecolor="white")
ax2.axvline(0, color="black", lw=0.5)
ax2.set_xlabel("Coefficient Value")
ax2.set_title("Route Profitability — Regression Coefficients")
fig.tight_layout()
save(fig, "04_route_profitability_results.png")


# ═══════════════════════════════════════════════════════
# 5. OPERATIONAL RISK (Random Forest Classifier)
# ═══════════════════════════════════════════════════════
print("[5/11] Operational Risk Model...")
df_risk = pd.read_csv(os.path.join(BASE, "new_models/operational_risk_dataset.csv"))
df_risk["Route_Encoded"] = df_risk["Route"].astype("category").cat.codes
features_risk = ["Delay_Rate", "Cancellation_Rate", "Weather_Disruption_Rate",
                 "Technical_Fault_Rate", "Route_Encoded"]
X = df_risk[features_risk]
y = df_risk["Risk_Category"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_risk = RandomForestClassifier(n_estimators=100, random_state=42)
model_risk.fit(X_tr, y_tr)
y_pred_risk = model_risk.predict(X_te)
acc_risk = accuracy_score(y_te, y_pred_risk)

labels_risk = sorted(y.unique())
cm_risk = confusion_matrix(y_te, y_pred_risk, labels=labels_risk)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.heatmap(cm_risk, annot=True, fmt="d", cmap="Blues", xticklabels=labels_risk,
            yticklabels=labels_risk, ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title(f"Operational Risk — Confusion Matrix\nAccuracy = {acc_risk:.4f}")

imp_risk = model_risk.feature_importances_
idx = np.argsort(imp_risk)
axes[1].barh([features_risk[i] for i in idx], imp_risk[idx], color=COLORS[4], edgecolor="white")
axes[1].set_xlabel("Feature Importance")
axes[1].set_title("Operational Risk — Feature Importance")
fig.tight_layout()
save(fig, "05_operational_risk_results.png")


# ═══════════════════════════════════════════════════════
# 6. CUSTOMER CHURN (Gradient Boosting)
# ═══════════════════════════════════════════════════════
print("[6/11] Customer Churn Model...")
df_pax = pd.read_csv(os.path.join(BASE, "simulated_data/passengers.csv"))
df_pax["churn_label"] = (df_pax["churn_risk_score"] > 0.5).astype(int)
le_loy = LabelEncoder()
df_pax["loyalty_encoded"] = le_loy.fit_transform(df_pax["loyalty_tier"])
le_gen = LabelEncoder()
df_pax["gender_encoded"] = le_gen.fit_transform(df_pax["gender"])

features_churn = ["age", "gender_encoded", "loyalty_encoded", "total_miles_flown",
                  "lifetime_spend", "avg_ticket_price", "cancellation_rate",
                  "upgrade_history_count", "ancillary_spend_avg"]
X = df_pax[features_churn]
y = df_pax["churn_label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_churn = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
model_churn.fit(X_tr, y_tr)
y_pred_churn = model_churn.predict(X_te)
y_proba_churn = model_churn.predict_proba(X_te)[:, 1]
acc_churn = accuracy_score(y_te, y_pred_churn)
f1_churn = f1_score(y_te, y_pred_churn)
auc_churn = roc_auc_score(y_te, y_proba_churn)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# ROC curve
fpr, tpr, _ = roc_curve(y_te, y_proba_churn)
axes[0].plot(fpr, tpr, color=COLORS[3], lw=2.5, label=f"AUC = {auc_churn:.4f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Customer Churn — ROC Curve")
axes[0].legend(fontsize=11)

# Confusion matrix
cm_churn = confusion_matrix(y_te, y_pred_churn)
sns.heatmap(cm_churn, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"Churn — Confusion Matrix\nAcc = {acc_churn:.4f}  |  F1 = {f1_churn:.4f}")

# Feature importance
imp_churn = model_churn.feature_importances_
idx = np.argsort(imp_churn)
axes[2].barh([features_churn[i] for i in idx], imp_churn[idx], color=COLORS[3], edgecolor="white")
axes[2].set_xlabel("Feature Importance")
axes[2].set_title("Churn — Feature Importance")
fig.tight_layout()
save(fig, "06_customer_churn_results.png")


# ═══════════════════════════════════════════════════════
# 7. FLIGHT DELAY PREDICTION (Random Forest)
# ═══════════════════════════════════════════════════════
print("[7/11] Flight Delay Model...")
df_ops = pd.read_csv(os.path.join(BASE, "simulated_data/operations.csv"))
df_fl = pd.read_csv(os.path.join(BASE, "simulated_data/flights.csv"))
merged = df_ops.merge(df_fl[["flight_id", "distance_km", "seat_capacity", "origin_airport"]],
                      on="flight_id", how="left")
merged["is_delayed"] = (merged["departure_delay_minutes"] > 15).astype(int)
merged["departure_hour"] = pd.to_datetime(merged["actual_departure"], errors="coerce").dt.hour
le_air = LabelEncoder()
merged["origin_encoded"] = le_air.fit_transform(merged["origin_airport"].astype(str))
for col in ["crew_delay_flag", "technical_issue_flag", "maintenance_flag"]:
    merged[col] = merged[col].astype(int) if merged[col].dtype == bool else merged[col]

features_delay = ["departure_hour", "origin_encoded", "turnaround_time_minutes",
                  "aircraft_utilization_hours", "crew_delay_flag", "technical_issue_flag",
                  "load_factor", "distance_km", "seat_capacity", "maintenance_flag"]
merged_clean = merged.dropna(subset=features_delay + ["is_delayed"])
X = merged_clean[features_delay]
y = merged_clean["is_delayed"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_delay = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
model_delay.fit(X_tr, y_tr)
y_pred_delay = model_delay.predict(X_te)
y_proba_delay = model_delay.predict_proba(X_te)[:, 1]
acc_delay = accuracy_score(y_te, y_pred_delay)
auc_delay = roc_auc_score(y_te, y_proba_delay)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# ROC
fpr, tpr, _ = roc_curve(y_te, y_proba_delay)
axes[0].plot(fpr, tpr, color=COLORS[5], lw=2.5, label=f"AUC = {auc_delay:.4f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Flight Delay — ROC Curve")
axes[0].legend(fontsize=11)

# Confusion matrix
cm_delay = confusion_matrix(y_te, y_pred_delay)
sns.heatmap(cm_delay, annot=True, fmt="d", cmap="YlOrBr",
            xticklabels=["On-Time", "Delayed"], yticklabels=["On-Time", "Delayed"], ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"Delay — Confusion Matrix\nAccuracy = {acc_delay:.4f}")

# Feature importance
imp_delay = model_delay.feature_importances_
idx = np.argsort(imp_delay)
axes[2].barh([features_delay[i] for i in idx], imp_delay[idx], color=COLORS[5], edgecolor="white")
axes[2].set_xlabel("Feature Importance")
axes[2].set_title("Flight Delay — Feature Importance")
fig.tight_layout()
save(fig, "07_flight_delay_results.png")


# ═══════════════════════════════════════════════════════
# 8. CANCELLATION PREDICTION (Gradient Boosting)
# ═══════════════════════════════════════════════════════
print("[8/11] Cancellation Model...")
df_book = pd.read_csv(os.path.join(BASE, "simulated_data/bookings.csv"))
df_book["is_cancelled"] = (df_book["booking_status"] == "Cancelled").astype(int)
le_fare = LabelEncoder()
df_book["fare_encoded"] = le_fare.fit_transform(df_book["fare_class"])
le_ch = LabelEncoder()
df_book["channel_encoded"] = le_ch.fit_transform(df_book["booking_channel"])
le_pay = LabelEncoder()
df_book["payment_encoded"] = le_pay.fit_transform(df_book["payment_method"])

features_cancel = ["fare_encoded", "base_fare", "taxes", "ancillary_revenue",
                   "total_price_paid", "discount_applied", "channel_encoded",
                   "passenger_count", "payment_encoded", "days_before_departure"]
df_book_clean = df_book.dropna(subset=features_cancel + ["is_cancelled"])
X = df_book_clean[features_cancel]
y = df_book_clean["is_cancelled"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_cancel = GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
model_cancel.fit(X_tr, y_tr)
y_pred_cancel = model_cancel.predict(X_te)
y_proba_cancel = model_cancel.predict_proba(X_te)[:, 1]
acc_cancel = accuracy_score(y_te, y_pred_cancel)
f1_cancel = f1_score(y_te, y_pred_cancel)
auc_cancel = roc_auc_score(y_te, y_proba_cancel)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fpr, tpr, _ = roc_curve(y_te, y_proba_cancel)
axes[0].plot(fpr, tpr, color=COLORS[6], lw=2.5, label=f"AUC = {auc_cancel:.4f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Cancellation — ROC Curve")
axes[0].legend(fontsize=11)

cm_cancel = confusion_matrix(y_te, y_pred_cancel)
sns.heatmap(cm_cancel, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Not Cancelled", "Cancelled"],
            yticklabels=["Not Cancelled", "Cancelled"], ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"Cancellation — Confusion Matrix\nAcc = {acc_cancel:.4f}  |  F1 = {f1_cancel:.4f}")

imp_cancel = model_cancel.feature_importances_
idx = np.argsort(imp_cancel)
axes[2].barh([features_cancel[i] for i in idx], imp_cancel[idx], color=COLORS[6], edgecolor="white")
axes[2].set_xlabel("Feature Importance")
axes[2].set_title("Cancellation — Feature Importance")
fig.tight_layout()
save(fig, "08_cancellation_results.png")


# ═══════════════════════════════════════════════════════
# 9. LOAD FACTOR PREDICTION (RF Regressor)
# ═══════════════════════════════════════════════════════
print("[9/11] Load Factor Model...")
df_af = pd.read_csv(os.path.join(BASE, "simulated_data/advanced_features.csv"))
features_lf = ["seat_capacity", "distance_km", "lead_time_days", "booking_velocity",
               "month", "is_weekend", "peak_season", "comp_avg_price", "base_fare"]
df_lf = df_af.dropna(subset=features_lf + ["hist_load_factor"])
X = df_lf[features_lf]
y = df_lf["hist_load_factor"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_lf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_lf.fit(X_tr, y_tr)
y_pred_lf = model_lf.predict(X_te)
r2_lf = r2_score(y_te, y_pred_lf)
mape_lf = mean_absolute_percentage_error(y_te, y_pred_lf) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.scatter(y_te, y_pred_lf, alpha=0.5, c=COLORS[5], edgecolors="white", s=50)
lims = [min(y_te.min(), y_pred_lf.min()) - 0.05, max(y_te.max(), y_pred_lf.max()) + 0.05]
ax.plot(lims, lims, "--", color="red", lw=1.5)
ax.set_xlabel("Actual Load Factor")
ax.set_ylabel("Predicted Load Factor")
ax.set_title(f"Load Factor: Actual vs Predicted\nR² = {r2_lf:.4f}  |  MAPE = {mape_lf:.2f}%")

imp_lf = model_lf.feature_importances_
idx = np.argsort(imp_lf)
axes[1].barh([features_lf[i] for i in idx], imp_lf[idx], color=COLORS[5], edgecolor="white")
axes[1].set_xlabel("Feature Importance")
axes[1].set_title("Load Factor — Feature Importance")
fig.tight_layout()
save(fig, "09_load_factor_results.png")


# ═══════════════════════════════════════════════════════
# 10. NO-SHOW PREDICTION (Gradient Boosting)
# ═══════════════════════════════════════════════════════
print("[10/11] No-Show Model...")
af_ns = df_af.copy()
af_ns["is_noshow"] = (af_ns["booking_status"] == "No-show").astype(int)
le_fc = LabelEncoder()
af_ns["fare_class_enc"] = le_fc.fit_transform(af_ns["fare_class"].astype(str))
features_ns = ["fare_class_enc", "lead_time_days", "base_fare", "discount_applied",
               "passenger_count", "is_weekend", "peak_season", "distance_km"]
af_ns_clean = af_ns.dropna(subset=features_ns + ["is_noshow"])
X = af_ns_clean[features_ns]
y = af_ns_clean["is_noshow"]

# only stratify if both classes present
if y.nunique() > 1:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model_ns = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model_ns.fit(X_tr, y_tr)
y_pred_ns = model_ns.predict(X_te)
y_proba_ns = model_ns.predict_proba(X_te)[:, 1] if len(model_ns.classes_) > 1 else np.zeros(len(X_te))
acc_ns = accuracy_score(y_te, y_pred_ns)
f1_ns = f1_score(y_te, y_pred_ns, zero_division=0)
auc_ns = roc_auc_score(y_te, y_proba_ns) if y_te.nunique() > 1 else 0.5

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
if y_te.nunique() > 1:
    fpr, tpr, _ = roc_curve(y_te, y_proba_ns)
    axes[0].plot(fpr, tpr, color=COLORS[7], lw=2.5, label=f"AUC = {auc_ns:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("No-Show — ROC Curve")
axes[0].legend(fontsize=11)

cm_ns = confusion_matrix(y_te, y_pred_ns)
sns.heatmap(cm_ns, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Show", "No-Show"], yticklabels=["Show", "No-Show"], ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"No-Show — Confusion Matrix\nAcc = {acc_ns:.4f}  |  F1 = {f1_ns:.4f}")

imp_ns = model_ns.feature_importances_
idx = np.argsort(imp_ns)
axes[2].barh([features_ns[i] for i in idx], imp_ns[idx], color=COLORS[7], edgecolor="white")
axes[2].set_xlabel("Feature Importance")
axes[2].set_title("No-Show — Feature Importance")
fig.tight_layout()
save(fig, "10_noshow_results.png")


# ═══════════════════════════════════════════════════════
# 11. PASSENGER CLUSTERING (K-Means)
# ═══════════════════════════════════════════════════════
print("[11/11] Passenger Clustering...")
df_clust = pd.read_csv(os.path.join(BASE, "simulated_data/passengers.csv"))
features_clust = ["age", "total_miles_flown", "lifetime_spend", "avg_ticket_price",
                  "cancellation_rate", "upgrade_history_count", "ancillary_spend_avg"]
X_clust = df_clust[features_clust].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
cluster_names = ["Premium Frequent", "High-Value Leisure", "At-Risk / Churners", "Budget Traveler"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Scatter: miles vs spend
ax = axes[0]
scatter = ax.scatter(X_clust["total_miles_flown"], X_clust["lifetime_spend"],
                     c=labels, cmap="Set1", alpha=0.6, s=40, edgecolors="white")
ax.set_xlabel("Total Miles Flown")
ax.set_ylabel("Lifetime Spend (₹)")
ax.set_title("Passenger Clusters — Miles vs Spend")
handles = [mpatches.Patch(color=plt.cm.Set1(i / 4), label=cluster_names[i]) for i in range(4)]
ax.legend(handles=handles, fontsize=8, loc="upper left")

# Cluster sizes
sizes = pd.Series(labels).value_counts().sort_index()
axes[1].bar([cluster_names[i] for i in sizes.index], sizes.values,
            color=[plt.cm.Set1(i / 4) for i in sizes.index], edgecolor="white")
axes[1].set_ylabel("Number of Passengers")
axes[1].set_title("Cluster Size Distribution")
axes[1].tick_params(axis="x", rotation=20)

# Radar / heatmap of cluster means
cluster_df = X_clust.copy()
cluster_df["cluster"] = labels
means = cluster_df.groupby("cluster").mean()
means_norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
sns.heatmap(means_norm.T, annot=means.T.round(1).values, fmt="", cmap="YlGnBu",
            xticklabels=[cluster_names[i] for i in means.index],
            yticklabels=features_clust, ax=axes[2])
axes[2].set_title("Cluster Feature Profiles (normalized)")
axes[2].tick_params(axis="x", rotation=20)
fig.tight_layout()
save(fig, "11_passenger_clustering_results.png")


# ═══════════════════════════════════════════════════════
# 12. SUMMARY DASHBOARD — All Models Comparison
# ═══════════════════════════════════════════════════════
print("\n[BONUS] Generating Summary Dashboard...")

# Collect all metrics
summary = {
    "Demand Forecasting": {"R²": r2_demand, "MAPE (%)": mape_demand},
    "Dynamic Pricing": {"R²": r2_price, "MAPE (%)": mape_price},
    "Route Profitability": {"R²": r2_profit},
    "Load Factor": {"R²": r2_lf, "MAPE (%)": mape_lf},
    "Operational Risk": {"Accuracy": acc_risk},
    "Customer Churn": {"Accuracy": acc_churn, "F1": f1_churn, "AUC-ROC": auc_churn},
    "Flight Delay": {"Accuracy": acc_delay, "AUC-ROC": auc_delay},
    "Cancellation": {"Accuracy": acc_cancel, "F1": f1_cancel, "AUC-ROC": auc_cancel},
    "No-Show": {"Accuracy": acc_ns, "F1": f1_ns, "AUC-ROC": auc_ns},
}

# Summary bar chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Regression R²
reg_models = ["Demand Forecasting", "Dynamic Pricing", "Route Profitability", "Load Factor"]
r2_vals = [summary[m].get("R²", 0) for m in reg_models]
bars = axes[0, 0].bar(reg_models, r2_vals, color=[COLORS[0], COLORS[1], COLORS[2], COLORS[5]], edgecolor="white")
for bar, val in zip(bars, r2_vals):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].set_ylabel("R² Score")
axes[0, 0].set_title("Regression Models — R² Score Comparison")
axes[0, 0].tick_params(axis="x", rotation=15)

# Classification Accuracy
cls_models = ["Operational Risk", "Customer Churn", "Flight Delay", "Cancellation", "No-Show"]
acc_vals = [summary[m].get("Accuracy", 0) for m in cls_models]
bars = axes[0, 1].bar(cls_models, acc_vals, color=[COLORS[4], COLORS[3], COLORS[5], COLORS[6], COLORS[7]], edgecolor="white")
for bar, val in zip(bars, acc_vals):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Classification Models — Accuracy Comparison")
axes[0, 1].tick_params(axis="x", rotation=15)

# AUC-ROC comparison
auc_models = ["Customer Churn", "Flight Delay", "Cancellation", "No-Show"]
auc_vals = [summary[m].get("AUC-ROC", 0) for m in auc_models]
bars = axes[1, 0].bar(auc_models, auc_vals, color=[COLORS[3], COLORS[5], COLORS[6], COLORS[7]], edgecolor="white")
for bar, val in zip(bars, auc_vals):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].set_ylabel("AUC-ROC")
axes[1, 0].set_title("Binary Classification — AUC-ROC Comparison")
axes[1, 0].tick_params(axis="x", rotation=15)

# MAPE comparison
mape_models = ["Demand Forecasting", "Dynamic Pricing", "Load Factor"]
mape_vals = [summary[m].get("MAPE (%)", 0) for m in mape_models]
bars = axes[1, 1].bar(mape_models, mape_vals, color=[COLORS[0], COLORS[1], COLORS[5]], edgecolor="white")
for bar, val in zip(bars, mape_vals):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
axes[1, 1].set_ylabel("MAPE (%)")
axes[1, 1].set_title("Regression Models — MAPE Comparison (lower is better)")
axes[1, 1].tick_params(axis="x", rotation=15)

fig.suptitle("Airline Revenue Optimization — ML Model Performance Summary", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "12_model_performance_summary.png")


# ═══════════════════════════════════════════════════════
# PRINT FINAL SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ALL RESULTS GENERATED SUCCESSFULLY!")
print("=" * 65)
print(f"\n  Output folder: {OUT}\n")
print("  Files generated:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        size_kb = os.path.getsize(os.path.join(OUT, f)) / 1024
        print(f"    • {f}  ({size_kb:.0f} KB)")

print("\n  Model Performance Summary:")
print("  " + "-" * 60)
print(f"  {'Model':<25} {'Metric':<12} {'Value'}")
print("  " + "-" * 60)
for model_name, metrics in summary.items():
    for metric, val in metrics.items():
        print(f"  {model_name:<25} {metric:<12} {val:.4f}")
print("  " + "-" * 60)
print(f"\n  Overbooking: Optimal Extra Seats = {int(optimal_k)} (Monte Carlo 1,000 sims)")
print(f"  Clustering: 4 segments identified via K-Means")
print()

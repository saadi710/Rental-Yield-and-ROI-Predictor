import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb

import joblib
import time

# ─────────────────────────────────────────────
# 0. PATHS SETUP
# ─────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "DATA", "Processed_Data")
models_dir    = os.path.join(script_dir, "Models")
os.makedirs(models_dir, exist_ok=True)

print("=" * 65)
print("         PHASE 4 — MODEL SELECTION & EVALUATION")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/5] Loading processed data...")

X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
y_test  = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

with open(os.path.join(processed_dir, "feature_names.json")) as f:
    names = json.load(f)

target_cols = names['targets']

print(f"      X_train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"      X_test:  {X_test.shape[0]:,} rows  × {X_test.shape[1]} features")
print(f"      Targets: {target_cols}")

# ─────────────────────────────────────────────
# 2. DEFINE MODELS
# ─────────────────────────────────────────────
print("\n[2/5] Setting up models to compare...")

models = {
    "Linear Regression": MultiOutputRegressor(
        LinearRegression()
    ),
    "Ridge Regression": MultiOutputRegressor(
        Ridge(alpha=1.0)
    ),
    "Lasso Regression": MultiOutputRegressor(
        Lasso(alpha=0.1)
    ),
    "Decision Tree": MultiOutputRegressor(
        DecisionTreeRegressor(max_depth=10, random_state=42)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    ),
    "XGBoost": MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
    ),
    "LightGBM": MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    ),
}

print(f"      {len(models)} models queued for comparison")

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────
print("\n[3/5] Training and evaluating all models...")
print("-" * 65)

results = {}

for name, model in models.items():
    print(f"\n  → Training: {name}...")
    start = time.time()

    # Train
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols)

    # Evaluate per target
    target_results = {}
    for i, target in enumerate(target_cols):
        mae  = mean_absolute_error(y_test[target], y_pred_df[target])
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred_df[target]))
        r2   = r2_score(y_test[target], y_pred_df[target])
        mape = np.mean(np.abs((y_test[target] - y_pred_df[target]) / (y_test[target] + 1e-9))) * 100
        target_results[target] = {
            "MAE":  round(mae,  4),
            "RMSE": round(rmse, 4),
            "R2":   round(r2,   4),
            "MAPE": round(mape, 2)
        }

    # Average R2 across all targets
    avg_r2 = round(np.mean([target_results[t]["R2"] for t in target_cols]), 4)
    avg_mae = round(np.mean([target_results[t]["MAE"] for t in target_cols]), 4)

    results[name] = {
        "per_target":  target_results,
        "avg_r2":      avg_r2,
        "avg_mae":     avg_mae,
        "train_time":  train_time
    }

    print(f"     Done in {train_time}s | Avg R² = {avg_r2} | Avg MAE = {avg_mae}")

# ─────────────────────────────────────────────
# 4. COMPARE & RANK MODELS
# ─────────────────────────────────────────────
print("\n\n[4/5] Model Comparison Results")
print("=" * 65)
print(f"{'Rank':<5} {'Model':<25} {'Avg R²':>8} {'Avg MAE':>12} {'Time(s)':>9}")
print("-" * 65)

# Sort by avg R2 descending
ranked = sorted(results.items(), key=lambda x: x[1]['avg_r2'], reverse=True)

for rank, (name, res) in enumerate(ranked, 1):
    marker = " ← BEST" if rank == 1 else ""
    print(f"{rank:<5} {name:<25} {res['avg_r2']:>8} {res['avg_mae']:>12} {res['train_time']:>9}{marker}")

print("-" * 65)

# Best model name
best_model_name = ranked[0][0]
best_model      = models[best_model_name]
print(f"\n  Best Model: {best_model_name}")
print(f"  Avg R²:     {ranked[0][1]['avg_r2']}")
print(f"  Avg MAE:    {ranked[0][1]['avg_mae']}")

# Detailed results for best model
print(f"\n  Detailed results for {best_model_name}:")
print(f"  {'Target':<30} {'R²':>8} {'MAE':>12} {'RMSE':>12} {'MAPE%':>8}")
print("  " + "-" * 72)
for target, metrics in ranked[0][1]['per_target'].items():
    print(f"  {target:<30} {metrics['R2']:>8} {metrics['MAE']:>12} {metrics['RMSE']:>12} {metrics['MAPE']:>7}%")

# ─────────────────────────────────────────────
# 5. SAVE EVERYTHING
# ─────────────────────────────────────────────
print(f"\n\n[5/5] Saving all results to Models folder...")

# Save all trained models
for name, model in models.items():
    safe_name = name.replace(" ", "_").lower()
    model_path = os.path.join(models_dir, f"{safe_name}.pkl")
    joblib.dump(model, model_path)

print(f"      Saved {len(models)} model .pkl files")

# Save comparison results as JSON
results_path = os.path.join(models_dir, "model_comparison.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"      Saved: model_comparison.json")

# Save comparison as CSV (easy to read)
rows = []
for name, res in results.items():
    row = {
        "Model":    name,
        "Avg_R2":   res['avg_r2'],
        "Avg_MAE":  res['avg_mae'],
        "Train_Time_s": res['train_time']
    }
    for target, metrics in res['per_target'].items():
        short = target.replace("_", " ").title()[:20]
        row[f"{short}_R2"]   = metrics['R2']
        row[f"{short}_MAE"]  = metrics['MAE']
        row[f"{short}_MAPE"] = metrics['MAPE']
    rows.append(row)

comparison_df = pd.DataFrame(rows).sort_values("Avg_R2", ascending=False)
csv_path = os.path.join(models_dir, "model_comparison.csv")
comparison_df.to_csv(csv_path, index=False)
print(f"      Saved: model_comparison.csv")

# Save best model separately
best_path = os.path.join(models_dir, "best_model.pkl")
joblib.dump(best_model, best_path)

# Save best model info
best_info = {
    "best_model_name": best_model_name,
    "avg_r2":          ranked[0][1]['avg_r2'],
    "avg_mae":         ranked[0][1]['avg_mae'],
    "per_target":      ranked[0][1]['per_target'],
    "feature_cols":    list(X_train.columns),
    "target_cols":     target_cols
}
info_path = os.path.join(models_dir, "best_model_info.json")
with open(info_path, 'w') as f:
    json.dump(best_info, f, indent=2)

print(f"      Saved: best_model.pkl")
print(f"      Saved: best_model_info.json")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("              PHASE 4 COMPLETE!")
print("=" * 65)
print(f"\n  Your Models folder now contains:")
print(f"   best_model.pkl          → best performing model")
print(f"   best_model_info.json    → best model metrics & details")
print(f"   model_comparison.csv    → all models ranked by R²")
print(f"   model_comparison.json   → full detailed results")
print(f"   + {len(models)} individual model .pkl files")
print(f"\n  Winner: {best_model_name} with R² = {ranked[0][1]['avg_r2']}")
print(f"\n  Ready for Phase 5 — Hyperparameter Tuning!")
print("=" * 65)
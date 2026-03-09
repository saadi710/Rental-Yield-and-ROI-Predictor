import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

import joblib
import time

# ─────────────────────────────────────────────
# 0. PATHS SETUP
# ─────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "DATA", "Processed_Data")
tuning_dir    = os.path.join(script_dir, "Model_Tuning")
os.makedirs(tuning_dir, exist_ok=True)

print("=" * 65)
print("   PHASE 5 — HYPERPARAMETER TUNING (XGBoost & LightGBM)")
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

target_cols  = names['targets']
feature_cols = names['features']
tune_targets = ['gross_rental_yield', 'net_rental_yield', 'roi_percent']

print(f"      Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"      Tuning on targets: {tune_targets}")

# ─────────────────────────────────────────────
# 2. HELPER — EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, targets):
    y_pred    = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols)
    results   = {}
    for t in targets:
        mae  = mean_absolute_error(y_test[t], y_pred_df[t])
        rmse = np.sqrt(mean_squared_error(y_test[t], y_pred_df[t]))
        r2   = r2_score(y_test[t], y_pred_df[t])
        results[t] = {'MAE': round(mae,4), 'RMSE': round(rmse,4), 'R2': round(r2,4)}
    avg_r2 = round(np.mean([results[t]['R2'] for t in targets]), 4)
    return results, avg_r2

# ─────────────────────────────────────────────
# 3. TUNE XGBOOST
# ─────────────────────────────────────────────
print("\n[2/5] Tuning XGBoost (50 trials)...")
print("      This will take ~3-5 minutes...")

def xgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0, 2.0),
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1
    }
    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    model.fit(X_train, y_train)
    _, avg_r2 = evaluate(model, X_test, y_test, tune_targets)
    return avg_r2

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

best_xgb_params = xgb_study.best_params
best_xgb_params.update({'random_state': 42, 'verbosity': 0, 'n_jobs': -1})

print(f"\n      XGBoost best R²:  {xgb_study.best_value:.4f}")
print(f"      Best params saved")

# ─────────────────────────────────────────────
# 4. TUNE LIGHTGBM
# ─────────────────────────────────────────────
print("\n[3/5] Tuning LightGBM (50 trials)...")
print("      This will take ~2-4 minutes...")

def lgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0, 2.0),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 150),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model.fit(X_train, y_train)
    _, avg_r2 = evaluate(model, X_test, y_test, tune_targets)
    return avg_r2

lgb_study = optuna.create_study(direction='maximize')
lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)

best_lgb_params = lgb_study.best_params
best_lgb_params.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})

print(f"\n      LightGBM best R²: {lgb_study.best_value:.4f}")
print(f"      Best params saved")

# ─────────────────────────────────────────────
# 5. TRAIN FINAL TUNED MODELS & COMPARE
# ─────────────────────────────────────────────
print("\n[4/5] Training final tuned models...")

tuned_models = {
    "XGBoost_Tuned":  MultiOutputRegressor(xgb.XGBRegressor(**best_xgb_params)),
    "LightGBM_Tuned": MultiOutputRegressor(lgb.LGBMRegressor(**best_lgb_params)),
}

final_results = {}
for name, model in tuned_models.items():
    print(f"\n  → Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)
    per_target, avg_r2 = evaluate(model, X_test, y_test, tune_targets)
    final_results[name] = {
        'per_target': per_target,
        'avg_r2':     avg_r2,
        'train_time': elapsed
    }
    print(f"     Done in {elapsed}s | Avg R² = {avg_r2}")

# ─────────────────────────────────────────────
# 6. PICK WINNER & SAVE EVERYTHING
# ─────────────────────────────────────────────
print("\n[5/5] Saving all results...")

ranked       = sorted(final_results.items(), key=lambda x: x[1]['avg_r2'], reverse=True)
winner_name  = ranked[0][0]
winner_model = tuned_models[winner_name]

# Print comparison table
print("\n" + "=" * 65)
print("            FINAL TUNING RESULTS")
print("=" * 65)
print(f"{'Rank':<5} {'Model':<25} {'Avg R²':>8} {'Time(s)':>9}")
print("-" * 65)
for rank, (name, res) in enumerate(ranked, 1):
    marker = " ← WINNER" if rank == 1 else ""
    print(f"{rank:<5} {name:<25} {res['avg_r2']:>8} {res['train_time']:>9}{marker}")
print("-" * 65)

print(f"\n  Detailed results for {winner_name}:")
print(f"  {'Target':<25} {'R²':>8} {'MAE':>10} {'RMSE':>10}")
print("  " + "-" * 55)
for t, m in ranked[0][1]['per_target'].items():
    print(f"  {t:<25} {m['R2']:>8} {m['MAE']:>10} {m['RMSE']:>10}")

# Save tuned model pkl files
for name, model in tuned_models.items():
    joblib.dump(model, os.path.join(tuning_dir, f"{name}.pkl"))

# Save best tuned model
joblib.dump(winner_model, os.path.join(tuning_dir, "best_tuned_model.pkl"))

# Save best hyperparameters
all_best_params = {
    "XGBoost_Tuned":  best_xgb_params,
    "LightGBM_Tuned": best_lgb_params,
}
with open(os.path.join(tuning_dir, "best_hyperparameters.json"), 'w') as f:
    json.dump(all_best_params, f, indent=2)

# Save tuning results
with open(os.path.join(tuning_dir, "tuning_results.json"), 'w') as f:
    json.dump(final_results, f, indent=2)

# Save comparison CSV
rows = []
for name, res in final_results.items():
    row = {'Model': name, 'Avg_R2': res['avg_r2'], 'Train_Time_s': res['train_time']}
    for t, m in res['per_target'].items():
        row[f"{t}_R2"]   = m['R2']
        row[f"{t}_MAE"]  = m['MAE']
        row[f"{t}_RMSE"] = m['RMSE']
    rows.append(row)
pd.DataFrame(rows).sort_values('Avg_R2', ascending=False).to_csv(
    os.path.join(tuning_dir, "tuning_comparison.csv"), index=False
)

# Save winner info
winner_info = {
    "winner_name":        winner_name,
    "avg_r2":             ranked[0][1]['avg_r2'],
    "per_target_metrics": ranked[0][1]['per_target'],
    "best_params":        all_best_params[winner_name],
    "feature_cols":       feature_cols,
    "target_cols":        target_cols,
    "tune_targets":       tune_targets
}
with open(os.path.join(tuning_dir, "winner_info.json"), 'w') as f:
    json.dump(winner_info, f, indent=2)

print(f"\n  Saved: best_tuned_model.pkl")
print(f"  Saved: XGBoost_Tuned.pkl, LightGBM_Tuned.pkl")
print(f"  Saved: best_hyperparameters.json")
print(f"  Saved: tuning_results.json")
print(f"  Saved: tuning_comparison.csv")
print(f"  Saved: winner_info.json")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("              PHASE 5 COMPLETE!")
print("=" * 65)
print(f"\n  Your Model_Tuning folder now contains:")
print(f"   best_tuned_model.pkl      → final production model")
print(f"   winner_info.json          → winner details & params")
print(f"   best_hyperparameters.json → all tuned params")
print(f"   tuning_results.json       → full tuning results")
print(f"   tuning_comparison.csv     → models ranked by R²")
print(f"   XGBoost_Tuned.pkl         → tuned XGBoost model")
print(f"   LightGBM_Tuned.pkl        → tuned LightGBM model")
print(f"\n  Winner: {winner_name}")
print(f"  Avg R²: {ranked[0][1]['avg_r2']}")
print(f"\n  Ready for Phase 6 — Explainability & SHAP!")
print("=" * 65)
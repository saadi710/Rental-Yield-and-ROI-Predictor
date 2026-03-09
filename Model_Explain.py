import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

import shap
import joblib

# ─────────────────────────────────────────────
# 0. PATHS SETUP
# ─────────────────────────────────────────────
script_dir      = os.path.dirname(os.path.abspath(__file__))
processed_dir   = os.path.join(script_dir, "DATA", "Processed_Data")
tuning_dir      = os.path.join(script_dir, "Model_Tuning")
explainability_dir = os.path.join(script_dir, "Model_Explainability")
os.makedirs(explainability_dir, exist_ok=True)

print("=" * 65)
print("       PHASE 6 — MODEL EXPLAINABILITY & SHAP")
print("=" * 65)

# ─────────────────────────────────────────────
# 1. LOAD DATA & MODEL
# ─────────────────────────────────────────────
print("\n[1/7] Loading model and data...")

model = joblib.load(os.path.join(tuning_dir, "best_tuned_model.pkl"))

X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

with open(os.path.join(processed_dir, "feature_names.json")) as f:
    names = json.load(f)

feature_cols = names['features']
target_cols  = names['targets']
tune_targets = ['gross_rental_yield', 'net_rental_yield', 'roi_percent']

# Use a sample for SHAP (faster)
sample_size = min(500, len(X_test))
X_sample    = X_test.sample(sample_size, random_state=42)

print(f"      Model loaded: XGBoost_Tuned")
print(f"      SHAP sample:  {sample_size} rows")

# ─────────────────────────────────────────────
# 2. COMPUTE SHAP VALUES PER TARGET
# ─────────────────────────────────────────────
print("\n[2/7] Computing SHAP values for each target...")
print("      This may take 1-2 minutes...")

shap_values_dict   = {}
explainer_dict     = {}

for i, target in enumerate(tune_targets):
    print(f"      Computing SHAP for: {target}...")
    # Each estimator in MultiOutputRegressor corresponds to one target
    single_model = model.estimators_[i]
    explainer    = shap.TreeExplainer(single_model)
    shap_vals    = explainer.shap_values(X_sample)
    shap_values_dict[target] = shap_vals
    explainer_dict[target]   = explainer
    print(f"      Done — {target}")

print("      All SHAP values computed!")

# ─────────────────────────────────────────────
# 3. PLOT 1 — SHAP SUMMARY (BAR) PER TARGET
# ─────────────────────────────────────────────
print("\n[3/7] Generating SHAP summary bar charts...")

for target in tune_targets:
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values_dict[target],
        X_sample,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title(f"Feature Importance — {target.replace('_', ' ').title()}",
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(explainability_dir, f"shap_bar_{target}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: shap_bar_{target}.png")

# ─────────────────────────────────────────────
# 4. PLOT 2 — SHAP SUMMARY (DOT/BEE SWARM) PER TARGET
# ─────────────────────────────────────────────
print("\n[4/7] Generating SHAP dot plots (impact direction)...")

for target in tune_targets:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values_dict[target],
        X_sample,
        feature_names=feature_cols,
        plot_type="dot",
        show=False,
        max_display=15
    )
    plt.title(f"Feature Impact Direction — {target.replace('_', ' ').title()}",
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(explainability_dir, f"shap_dot_{target}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: shap_dot_{target}.png")

# ─────────────────────────────────────────────
# 5. PLOT 3 — COMBINED FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n[5/7] Generating combined feature importance chart...")

# Average absolute SHAP values across all 3 targets
importance_dict = {}
for feature in feature_cols:
    idx = feature_cols.index(feature)
    avg_importance = np.mean([
        np.mean(np.abs(shap_values_dict[t][:, idx]))
        for t in tune_targets
    ])
    importance_dict[feature] = round(avg_importance, 6)

# Sort
importance_df = pd.DataFrame({
    'Feature':    list(importance_dict.keys()),
    'Importance': list(importance_dict.values())
}).sort_values('Importance', ascending=True).tail(15)

# Plot
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance_df)))
fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
               color=colors, edgecolor='white', height=0.7)

# Add value labels
for bar, val in zip(bars, importance_df['Importance']):
    ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, color='#333333')

ax.set_xlabel('Mean |SHAP Value| (Average Impact)', fontsize=12)
ax.set_title('Top 15 Features — Combined Importance Across All Targets',
             fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('#ffffff')
plt.tight_layout()
path = os.path.join(explainability_dir, "combined_feature_importance.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"      Saved: combined_feature_importance.png")

# ─────────────────────────────────────────────
# 6. PLOT 4 — SINGLE PREDICTION EXPLANATION
# ─────────────────────────────────────────────
print("\n[6/7] Generating single prediction explanation (waterfall)...")

# Pick one sample property to explain
sample_row = X_sample.iloc[[0]]

for target in tune_targets:
    single_model = model.estimators_[tune_targets.index(target)]
    explainer    = explainer_dict[target]
    shap_vals    = explainer.shap_values(sample_row)
    base_val     = explainer.expected_value

    # Waterfall style manual plot
    feature_shap = list(zip(feature_cols, shap_vals[0]))
    feature_shap_sorted = sorted(feature_shap, key=lambda x: abs(x[1]), reverse=True)[:10]
    features_plot = [f[0] for f in feature_shap_sorted]
    values_plot   = [f[1] for f in feature_shap_sorted]

    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features_plot[::-1], values_plot[::-1],
                   color=colors[::-1], edgecolor='white', height=0.6)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
    ax.set_title(f'Why this prediction? — {target.replace("_", " ").title()}\n'
                 f'Base value: {base_val:.3f}',
                 fontsize=12, fontweight='bold', pad=12)

    red_patch   = mpatches.Patch(color='#e74c3c', label='Increases yield/ROI')
    green_patch = mpatches.Patch(color='#2ecc71', label='Decreases yield/ROI')
    ax.legend(handles=[red_patch, green_patch], fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    path = os.path.join(explainability_dir, f"waterfall_{target}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: waterfall_{target}.png")

# ─────────────────────────────────────────────
# 7. SAVE FEATURE IMPORTANCE JSON
# ─────────────────────────────────────────────
print("\n[7/7] Saving feature importance data...")

# Per target importance
per_target_importance = {}
for target in tune_targets:
    imp = {}
    for j, feature in enumerate(feature_cols):
        imp[feature] = round(float(np.float64(np.mean(np.abs(shap_values_dict[target][:, j])))), 6)
    per_target_importance[target] = dict(
        sorted(imp.items(), key=lambda x: x[1], reverse=True)
    )

# Combined importance — force all values to native Python float
combined_importance = dict(
    sorted(
        {k: float(v) for k, v in importance_dict.items()}.items(),
        key=lambda x: x[1], reverse=True
    )
)

explainability_report = {
    "model":                   "XGBoost_Tuned",
    "shap_sample_size":        sample_size,
    "combined_importance":     combined_importance,
    "per_target_importance":   per_target_importance,
    "top_5_features_overall":  list(combined_importance.keys())[:5],
}

report_path = os.path.join(explainability_dir, "explainability_report.json")
with open(report_path, 'w') as f:
    json.dump(explainability_report, f, indent=2)
print(f"      Saved: explainability_report.json")

# Print top features
print(f"\n  Top 5 Most Important Features Overall:")
for i, (feat, imp) in enumerate(list(combined_importance.items())[:5], 1):
    print(f"   {i}. {feat:<25} importance = {imp:.4f}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("              PHASE 6 COMPLETE!")
print("=" * 65)
print(f"\n  Your Model_Explainability folder contains:")
print(f"   combined_feature_importance.png  → top 15 features overall")
print(f"   shap_bar_gross_rental_yield.png  → feature importance (yield)")
print(f"   shap_bar_net_rental_yield.png    → feature importance (net yield)")
print(f"   shap_bar_roi_percent.png         → feature importance (ROI)")
print(f"   shap_dot_gross_rental_yield.png  → impact direction (yield)")
print(f"   shap_dot_net_rental_yield.png    → impact direction (net yield)")
print(f"   shap_dot_roi_percent.png         → impact direction (ROI)")
print(f"   waterfall_gross_rental_yield.png → single prediction explained")
print(f"   waterfall_net_rental_yield.png   → single prediction explained")
print(f"   waterfall_roi_percent.png        → single prediction explained")
print(f"   explainability_report.json       → all importance scores")
print(f"\n  Ready for Phase 7 — Web Dashboard (Streamlit)!")
print("=" * 65)
from flask import Flask, render_template, request, jsonify
import os, json, joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'roi_predictor_secret_2024'

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR         = os.path.join(BASE_DIR, "Model_Tuning")
PROCESSED_DIR      = os.path.join(BASE_DIR, "DATA", "Processed_Data")
EXPLAINABILITY_DIR = os.path.join(BASE_DIR, "Model_Explainability")

model = joblib.load(os.path.join(TUNING_DIR, "best_tuned_model.pkl"))

with open(os.path.join(PROCESSED_DIR, "label_encodings.json")) as f:
    encodings = json.load(f)
with open(os.path.join(PROCESSED_DIR, "scaler_params.json")) as f:
    scaler_params = json.load(f)
with open(os.path.join(PROCESSED_DIR, "feature_names.json")) as f:
    feature_names = json.load(f)
with open(os.path.join(EXPLAINABILITY_DIR, "explainability_report.json")) as f:
    shap_report = json.load(f)

feature_cols = feature_names['features']
target_cols  = feature_names['targets']

def encode(col, val):
    return encodings.get(col, {}).get(str(val), 0)

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    if month in [3, 4, 5]:  return 'Spring'
    if month in [6, 7, 8]:  return 'Summer'
    return 'Autumn'

def get_size_cat(m):
    if m <= 3: return 0
    if m <= 5: return 1
    if m <= 10: return 2
    if m <= 20: return 3
    if m <= 40: return 4
    return 5

def get_price_tier(p):
    if p < 3000000:  return 0
    if p < 8000000:  return 1
    if p < 18000000: return 2
    if p < 40000000: return 3
    return 4

def get_rating(val, metric='yield'):
    if metric == 'yield':
        if val >= 6:   return 'Excellent', '#00ff99', '🟢'
        if val >= 4.5: return 'Good',      '#00d4ff', '🔵'
        if val >= 3:   return 'Average',   '#ffd700', '🟡'
        return 'Low', '#ff4d4d', '🔴'
    else:
        if val >= 8:   return 'Excellent', '#00ff99', '🟢'
        if val >= 6:   return 'Good',      '#00d4ff', '🔵'
        if val >= 4:   return 'Average',   '#ffd700', '🟡'
        return 'Low', '#ff4d4d', '🔴'

def build_features(data):
    price      = float(data['price'])
    area_marla = float(data['area_marla'])
    area_sqft  = area_marla * 272.25
    bedrooms   = int(data['bedrooms'])
    baths      = int(data['baths'])
    month      = int(data['month'])
    raw = {
        'price': price, 'area_marla': area_marla, 'area_sqft': area_sqft,
        'bedrooms': bedrooms, 'baths': baths,
        'price_per_sqft': price / max(area_sqft, 1),
        'price_per_marla': price / max(area_marla, 1),
        'bath_bed_ratio': baths / max(bedrooms, 1),
        'area_per_bedroom': area_sqft / max(bedrooms, 1),
        'latitude': float(data['latitude']),
        'longitude': float(data['longitude']),
        'location_score': 3.0,
        'year': int(data['year']), 'month': month,
        'property_type_encoded': encode('property_type', data['prop_type']),
        'city_encoded': encode('city', data['city']),
        'province_name_encoded': encode('province_name', data['province']),
        'locality_encoded': 0,
        'season_listed_encoded': encode('season_listed', get_season(month)),
        'size_category_encoded': get_size_cat(area_marla),
        'price_tier_encoded': get_price_tier(price),
    }
    scaled = {f: (raw[f] - scaler_params['mean'][f]) / scaler_params['scale'][f] for f in feature_cols}
    return pd.DataFrame([scaled])[feature_cols]

@app.route('/')
def index():
    return render_template('index.html',
        cities=sorted(encodings['city'].keys()),
        provinces=sorted(encodings['province_name'].keys()),
        prop_types=sorted(encodings['property_type'].keys()),
        top_features=list(shap_report['combined_importance'].items())[:10])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.json
        X     = build_features(data)
        preds = model.predict(X)[0]
        pred  = dict(zip(target_cols, preds))

        gross = round(float(pred['gross_rental_yield']), 2)
        net   = round(float(pred['net_rental_yield']),   2)
        roi   = round(float(pred['roi_percent']),        2)
        mrent = round(float(pred['estimated_monthly_rent']))
        arent = round(float(pred['estimated_annual_rent']))
        price = float(data['price'])

        gr, gc, gi = get_rating(gross, 'yield')
        nr, nc, ni = get_rating(net,   'yield')
        rr, rc, ri = get_rating(roi,   'roi')

        return jsonify({
            'success': True,
            'gross': gross, 'net': net, 'roi': roi,
            'monthly_rent': mrent, 'annual_rent': arent,
            'breakeven': round(price / max(arent, 1), 1),
            'five_year': mrent * 60, 'ten_year': mrent * 120,
            'gr': gr, 'gc': gc, 'gi': gi,
            'nr': nr, 'nc': nc, 'ni': ni,
            'rr': rr, 'rc': rc, 'ri': ri,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  🏠 ROI Predictor — Starting...")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
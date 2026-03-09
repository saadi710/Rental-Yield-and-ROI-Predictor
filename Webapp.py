import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ROI Predictor — Pakistan Property",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f1923 0%, #112233 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Title */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #0077ff, #00ff99);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        color: #7a9bc0;
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0f1e2e, #112840);
        border: 1px solid #1e4060;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,100,255,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }

    .metric-label {
        font-family: 'DM Sans', sans-serif;
        color: #5a8aaa;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        line-height: 1;
    }
    .metric-value.green  { color: #00ff99; }
    .metric-value.yellow { color: #ffd700; }
    .metric-value.orange { color: #ff8c42; }

    .metric-tag {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .tag-excellent { background: #00ff9920; color: #00ff99; }
    .tag-good      { background: #00d4ff20; color: #00d4ff; }
    .tag-average   { background: #ffd70020; color: #ffd700; }
    .tag-low       { background: #ff4d4d20; color: #ff4d4d; }

    /* Section headers */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #cce4ff;
        border-left: 3px solid #0077ff;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Investment verdict */
    .verdict-box {
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-top: 1rem;
        font-family: 'DM Sans', sans-serif;
    }
    .verdict-excellent { background: #00ff9915; border: 1px solid #00ff9940; }
    .verdict-good      { background: #00d4ff15; border: 1px solid #00d4ff40; }
    .verdict-average   { background: #ffd70015; border: 1px solid #ffd70040; }
    .verdict-low       { background: #ff4d4d15; border: 1px solid #ff4d4d40; }

    .verdict-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    /* Comparison table */
    .compare-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #1a2f45;
        color: #a0c0d8;
        font-size: 0.9rem;
    }
    .compare-val { color: #00d4ff; font-weight: 600; }

    /* Stremlit widget overrides */
    .stSlider > div > div > div { background: #0077ff !important; }
    .stSelectbox > div > div { background: #0f1e2e; border-color: #1e4060; color: #cce4ff; }
    .stNumberInput > div > div > input { background: #0f1e2e; border-color: #1e4060; color: #cce4ff; }
    label { color: #7a9bc0 !important; font-size: 0.85rem !important; }

    /* Divider */
    hr { border-color: #1e3a5f; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL & ASSETS
# ─────────────────────────────────────────────
script_dir         = os.path.dirname(os.path.abspath(__file__))
tuning_dir         = os.path.join(script_dir, "Model_Tuning")
processed_dir      = os.path.join(script_dir, "DATA", "Processed_Data")
explainability_dir = os.path.join(script_dir, "Model_Explainability")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(tuning_dir, "best_tuned_model.pkl"))

@st.cache_data
def load_assets():
    with open(os.path.join(processed_dir, "label_encodings.json")) as f:
        encodings = json.load(f)
    with open(os.path.join(processed_dir, "scaler_params.json")) as f:
        scaler_params = json.load(f)
    with open(os.path.join(processed_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(explainability_dir, "explainability_report.json")) as f:
        shap_report = json.load(f)
    return encodings, scaler_params, feature_names, shap_report

model                                     = load_model()
encodings, scaler_params, feature_names, shap_report = load_assets()

feature_cols = feature_names['features']
target_cols  = feature_names['targets']

# ─────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <span style='font-size:2.2rem;'>🏠</span>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700;
                    color:#cce4ff; margin-top:0.3rem;'>ROI Predictor</div>
        <div style='color:#5a8aaa; font-size:0.75rem;'>Pakistan Property Intelligence</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📍 Location</div>", unsafe_allow_html=True)

    cities = list(encodings['city'].keys())
    city   = st.selectbox("City", sorted(cities))

    provinces = list(encodings['province_name'].keys())
    province  = st.selectbox("Province", sorted(provinces))

    latitude  = st.number_input("Latitude",  value=31.5204, format="%.4f")
    longitude = st.number_input("Longitude", value=74.3587, format="%.4f")

    st.markdown("<div class='section-header'>🏗️ Property Details</div>", unsafe_allow_html=True)

    prop_types = list(encodings['property_type'].keys())
    prop_type  = st.selectbox("Property Type", sorted(prop_types))

    area_marla = st.slider("Area (Marla)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    area_sqft  = area_marla * 272.25
    st.caption(f"≈ {area_sqft:,.0f} sq ft")

    bedrooms = st.slider("Bedrooms",  min_value=1, max_value=10, value=3)
    baths    = st.slider("Bathrooms", min_value=1, max_value=10, value=2)

    st.markdown("<div class='section-header'>💰 Financials</div>", unsafe_allow_html=True)
    price = st.number_input("Purchase Price (PKR)", min_value=500000, max_value=500000000,
                             value=15000000, step=500000)
    st.caption(f"≈ PKR {price/1000000:.1f}M")

    st.markdown("<div class='section-header'>📅 Listing Info</div>", unsafe_allow_html=True)
    month = st.selectbox("Month Listed", list(range(1, 13)),
                          format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                  'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    year  = st.selectbox("Year", [2018, 2019, 2020, 2021, 2022, 2023, 2024])

    predict_btn = st.button("🔍  Predict ROI & Yield", use_container_width=True)

# ─────────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────────
def encode(col, val):
    return encodings.get(col, {}).get(str(val), 0)

def get_season(month):
    if month in [12, 1, 2]:  return 'Winter'
    if month in [3, 4, 5]:   return 'Spring'
    if month in [6, 7, 8]:   return 'Summer'
    return 'Autumn'

def build_features():
    price_per_sqft   = price / max(area_sqft, 1)
    price_per_marla  = price / max(area_marla, 1)
    bath_bed_ratio   = baths / max(bedrooms, 1)
    area_per_bedroom = area_sqft / max(bedrooms, 1)
    season           = get_season(month)

    # Size category encoded
    if   area_marla <= 3:   size_cat = 0
    elif area_marla <= 5:   size_cat = 1
    elif area_marla <= 10:  size_cat = 2
    elif area_marla <= 20:  size_cat = 3
    elif area_marla <= 40:  size_cat = 4
    else:                   size_cat = 5

    # Price tier encoded (rough quantile approximation)
    if   price < 3000000:    price_tier = 0
    elif price < 8000000:    price_tier = 1
    elif price < 18000000:   price_tier = 2
    elif price < 40000000:   price_tier = 3
    else:                    price_tier = 4

    location_score = 3.0  # default mid-score for new input

    raw = {
        'price':                    price,
        'area_marla':               area_marla,
        'area_sqft':                area_sqft,
        'bedrooms':                 bedrooms,
        'baths':                    baths,
        'price_per_sqft':           price_per_sqft,
        'price_per_marla':          price_per_marla,
        'bath_bed_ratio':           bath_bed_ratio,
        'area_per_bedroom':         area_per_bedroom,
        'latitude':                 latitude,
        'longitude':                longitude,
        'location_score':           location_score,
        'year':                     year,
        'month':                    month,
        'property_type_encoded':    encode('property_type', prop_type),
        'city_encoded':             encode('city', city),
        'province_name_encoded':    encode('province_name', province),
        'locality_encoded':         0,
        'season_listed_encoded':    encode('season_listed', season),
        'size_category_encoded':    size_cat,
        'price_tier_encoded':       price_tier,
    }

    # Scale using saved scaler params
    scaled = {}
    for feat in feature_cols:
        mean  = scaler_params['mean'][feat]
        scale = scaler_params['scale'][feat]
        scaled[feat] = (raw[feat] - mean) / scale

    return pd.DataFrame([scaled])[feature_cols]

# ─────────────────────────────────────────────
# HELPER — RATING TAG
# ─────────────────────────────────────────────
def yield_rating(val):
    if val >= 6:   return "Excellent", "tag-excellent", "verdict-excellent", "🟢"
    if val >= 4.5: return "Good",      "tag-good",      "verdict-good",      "🔵"
    if val >= 3:   return "Average",   "tag-average",   "verdict-average",   "🟡"
    return                "Low",       "tag-low",       "verdict-low",       "🔴"

def roi_rating(val):
    if val >= 8:   return "Excellent", "tag-excellent", "verdict-excellent", "🟢"
    if val >= 6:   return "Good",      "tag-good",      "verdict-good",      "🔵"
    if val >= 4:   return "Average",   "tag-average",   "verdict-average",   "🟡"
    return                "Low",       "tag-low",       "verdict-low",       "🔴"

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Rental Yield & ROI Predictor</div>
<div class='hero-sub'>AI-powered property investment intelligence for Pakistan real estate</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔍 Explainability", "📈 Compare Properties"])

# ════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════
with tab1:
    if predict_btn:
        with st.spinner("Analyzing property..."):
            X_input = build_features()
            preds   = model.predict(X_input)[0]

            pred_dict = dict(zip(target_cols, preds))
            gross_yield   = round(float(pred_dict['gross_rental_yield']), 2)
            net_yield     = round(float(pred_dict['net_rental_yield']),   2)
            roi           = round(float(pred_dict['roi_percent']),        2)
            monthly_rent  = round(float(pred_dict['estimated_monthly_rent']))
            annual_rent   = round(float(pred_dict['estimated_annual_rent']))

        # ── METRIC CARDS ──
        st.markdown("<div class='section-header'>📊 Prediction Results</div>", unsafe_allow_html=True)

        g_rating, g_tag, g_verdict, g_icon = yield_rating(gross_yield)
        n_rating, n_tag, n_verdict, n_icon = yield_rating(net_yield)
        r_rating, r_tag, r_verdict, r_icon = roi_rating(roi)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Gross Rental Yield</div>
                <div class='metric-value'>{gross_yield}%</div>
                <span class='metric-tag {g_tag}'>{g_icon} {g_rating}</span>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Net Rental Yield</div>
                <div class='metric-value green'>{net_yield}%</div>
                <span class='metric-tag {n_tag}'>{n_icon} {n_rating}</span>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>ROI %</div>
                <div class='metric-value yellow'>{roi}%</div>
                <span class='metric-tag {r_tag}'>{r_icon} {r_rating}</span>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Est. Monthly Rent</div>
                <div class='metric-value orange'>PKR {monthly_rent:,}</div>
                <span class='metric-tag tag-average'>per month</span>
            </div>""", unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Est. Annual Rent</div>
                <div class='metric-value orange'>PKR {annual_rent:,}</div>
                <span class='metric-tag tag-average'>per year</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── VERDICT ──
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown(f"""
            <div class='verdict-box {g_verdict}'>
                <div class='verdict-title' style='color:#cce4ff;'>{g_icon} Investment Verdict</div>
                <div style='color:#8ab0cc; font-size:0.9rem; margin-top:0.3rem;'>
                    This property offers a <strong style='color:#cce4ff;'>{g_rating.lower()}</strong>
                    gross rental yield of <strong style='color:#00d4ff;'>{gross_yield}%</strong>.
                    {'Consider this a strong rental income asset.' if gross_yield >= 4.5
                     else 'Rental income may be modest relative to purchase price.'}
                </div>
            </div>""", unsafe_allow_html=True)

        with col_v2:
            st.markdown(f"""
            <div class='verdict-box {r_verdict}'>
                <div class='verdict-title' style='color:#cce4ff;'>{r_icon} ROI Verdict</div>
                <div style='color:#8ab0cc; font-size:0.9rem; margin-top:0.3rem;'>
                    Total ROI including appreciation is <strong style='color:#ffd700;'>{roi}%</strong>.
                    {'This is a strong overall return on investment.' if roi >= 6
                     else 'Consider negotiating price or looking at higher-yield areas.'}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GAUGE CHARTS ──
        st.markdown("<div class='section-header'>📉 Visual Gauges</div>", unsafe_allow_html=True)
        col_g1, col_g2, col_g3 = st.columns(3)

        def make_gauge(title, value, max_val, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': title, 'font': {'size': 13, 'color': '#7a9bc0', 'family': 'DM Sans'}},
                number={'suffix': '%', 'font': {'size': 28, 'color': color, 'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, max_val], 'tickcolor': '#2a4a6a',
                             'tickfont': {'color': '#2a4a6a', 'size': 10}},
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': '#0f1e2e',
                    'bordercolor': '#1e4060',
                    'steps': [
                        {'range': [0, max_val*0.33],     'color': '#0d1520'},
                        {'range': [max_val*0.33, max_val*0.66], 'color': '#0f1e2e'},
                        {'range': [max_val*0.66, max_val], 'color': '#122030'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 2},
                        'thickness': 0.75,
                        'value': value
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                height=220, margin=dict(t=40, b=10, l=20, r=20)
            )
            return fig

        with col_g1:
            st.plotly_chart(make_gauge("Gross Yield", gross_yield, 12, "#00d4ff"),
                            use_container_width=True)
        with col_g2:
            st.plotly_chart(make_gauge("Net Yield", net_yield, 10, "#00ff99"),
                            use_container_width=True)
        with col_g3:
            st.plotly_chart(make_gauge("ROI %", roi, 15, "#ffd700"),
                            use_container_width=True)

        # ── BREAKEVEN ──
        st.markdown("<div class='section-header'>⏱️ Breakeven Analysis</div>", unsafe_allow_html=True)
        breakeven_years = round(price / max(annual_rent, 1), 1)

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.metric("Breakeven (Rent Only)", f"{breakeven_years} years")
        with col_b2:
            st.metric("5-Year Rental Income", f"PKR {annual_rent * 5:,}")
        with col_b3:
            st.metric("10-Year Rental Income", f"PKR {annual_rent * 10:,}")

        # Save to session for comparison tab
        if 'comparisons' not in st.session_state:
            st.session_state.comparisons = []

        if st.button("➕ Add to Comparison"):
            st.session_state.comparisons.append({
                'Label':        f"{prop_type} | {city} | {area_marla}M",
                'Price (PKR)':  price,
                'Gross Yield':  gross_yield,
                'Net Yield':    net_yield,
                'ROI %':        roi,
                'Monthly Rent': monthly_rent,
            })
            st.success("Added to comparison! Go to Compare Properties tab.")

    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem; color:#2a4a6a;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🏠</div>
            <div style='font-family:Syne,sans-serif; font-size:1.4rem;
                        font-weight:700; color:#1e3a5f;'>
                Fill in property details and click Predict
            </div>
            <div style='font-size:0.9rem; margin-top:0.5rem; color:#1e3050;'>
                Use the sidebar to enter location, size, and price
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 2 — EXPLAINABILITY
# ════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🔍 What Drives Rental Yield & ROI?</div>",
                unsafe_allow_html=True)

    # Top features bar chart from SHAP report
    combined_imp = shap_report['combined_importance']
    imp_df = pd.DataFrame({
        'Feature':    list(combined_imp.keys())[:15],
        'Importance': list(combined_imp.values())[:15]
    }).sort_values('Importance')

    fig_imp = px.bar(
        imp_df, x='Importance', y='Feature', orientation='h',
        color='Importance',
        color_continuous_scale=['#0d2a45', '#0077ff', '#00d4ff', '#00ff99'],
        title='Top 15 Features — Overall Importance (SHAP)',
    )
    fig_imp.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#7a9bc0', family='DM Sans'),
        title_font=dict(color='#cce4ff', size=14, family='Syne'),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor='#1a2f45', title='Mean |SHAP Value|'),
        yaxis=dict(gridcolor='#1a2f45', title=''),
        height=500,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show saved SHAP images
    st.markdown("<div class='section-header'>📸 SHAP Analysis Charts</div>",
                unsafe_allow_html=True)

    target_map = {
        'gross_rental_yield': 'Gross Rental Yield',
        'net_rental_yield':   'Net Rental Yield',
        'roi_percent':        'ROI %'
    }

    for target, label in target_map.items():
        st.markdown(f"**{label}**")
        col_s1, col_s2 = st.columns(2)
        bar_path = os.path.join(explainability_dir, f"shap_bar_{target}.png")
        dot_path = os.path.join(explainability_dir, f"shap_dot_{target}.png")

        with col_s1:
            if os.path.exists(bar_path):
                st.image(bar_path, caption=f"Feature Importance — {label}", use_container_width=True)
        with col_s2:
            if os.path.exists(dot_path):
                st.image(dot_path, caption=f"Impact Direction — {label}", use_container_width=True)

        st.markdown("---")

# ════════════════════════════════════════
# TAB 3 — COMPARISON
# ════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>📈 Property Comparison</div>",
                unsafe_allow_html=True)

    if 'comparisons' not in st.session_state or len(st.session_state.comparisons) == 0:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#1e3a5f;'>
            <div style='font-size:2.5rem;'>📊</div>
            <div style='font-family:Syne,sans-serif; color:#2a4a6a; margin-top:0.5rem;'>
                No properties added yet
            </div>
            <div style='font-size:0.85rem; color:#1e3050; margin-top:0.3rem;'>
                Predict a property and click "Add to Comparison"
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        comp_df = pd.DataFrame(st.session_state.comparisons)
        st.dataframe(comp_df, use_container_width=True)

        # Bar chart comparison
        fig_comp = go.Figure()
        metrics  = ['Gross Yield', 'Net Yield', 'ROI %']
        colors   = ['#00d4ff', '#00ff99', '#ffd700']

        for metric, color in zip(metrics, colors):
            fig_comp.add_trace(go.Bar(
                name=metric,
                x=comp_df['Label'],
                y=comp_df[metric],
                marker_color=color,
                opacity=0.85
            ))

        fig_comp.update_layout(
            barmode='group',
            paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            font=dict(color='#7a9bc0', family='DM Sans'),
            title=dict(text='Property Comparison — Yield & ROI',
                       font=dict(color='#cce4ff', size=14, family='Syne')),
            xaxis=dict(gridcolor='#1a2f45'),
            yaxis=dict(gridcolor='#1a2f45', title='Percentage (%)'),
            legend=dict(bgcolor='#0f1e2e', bordercolor='#1e4060'),
            height=420,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        if st.button("🗑️ Clear Comparison"):
            st.session_state.comparisons = []
            st.rerun()
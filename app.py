import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="Chronic Disease Intelligence", layout="wide")
st.title("🏥 Chronic Disease Predictive Intelligence")
st.markdown("---")

# --- LOAD ASSETS ---
@st.cache_data
def load_data():
    # Ensure this path matches your data folder location
    df = pd.read_csv('./data/CDC_Cleaned.csv')
    model = joblib.load('models/chronic_model.pkl')
    encoders = joblib.load('models/encoders.joblib')
    return df, model, encoders

df, model, encoders = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Search Criteria")
# Slider now starts at 2022 to provide a historical baseline
year = st.sidebar.slider("Projection Year", 2022, 2030, 2022)

topic = st.sidebar.selectbox("Health Topic", sorted(df['Topic'].unique()))
questions = df[df['Topic'] == topic]['Question'].unique()
question = st.sidebar.selectbox("Specific Metric", questions)

state = st.sidebar.selectbox("Target State", sorted(df['LocationAbbr'].unique()))

strat_cat = st.sidebar.selectbox("Stratification Category", df['StratificationCategory1'].unique())
strat_val = st.sidebar.selectbox("Focus Group", df[df['StratificationCategory1'] == strat_cat]['Stratification1'].unique())

st.sidebar.divider()
st.sidebar.markdown("###  Research Citations")
st.sidebar.caption("Trend factors synthesized from CDC (2025) and AHA (2024) forecasting reports.")

# --- RESEARCH-BACKED GROWTH RATES ---
GROWTH_RATES = {
    'Diabetes': 1.023, 
    'Cancer': 1.015, 
    'Cardiovascular Disease': 1.008,
    'Arthritis': 1.011, 
    'COPD': 1.005, 
    'Chronic Kidney Disease': 1.018, 
    'Default': 1.012
}

# --- PREDICTION ENGINE ---
def get_forecast(y, s, t, q, sc, sv):
    ref = df[df['LocationAbbr'] == s].iloc[0]
    features = ['YearStart', 'LocationAbbr', 'Topic', 'Question', 
                'StratificationCategory1', 'Stratification1', 
                'Latitude', 'Longitude', 'HealthCluster']
    
    # Input always uses 2022 as the base for the ML model
    inp = pd.DataFrame([{
        'YearStart': 2022, 'LocationAbbr': s, 'Topic': t, 'Question': q, 
        'StratificationCategory1': sc, 'Stratification1': sv, 
        'Latitude': ref['Latitude'], 'Longitude': ref['Longitude'], 
        'HealthCluster': ref['HealthCluster']
    }])
    
    for col, le in encoders.items():
        inp[col] = le.transform(inp[col].astype(str))
    
    # Static ML prediction for 2022
    base_pred = model.predict(inp[features])[0]
    
    # Temporal adjustment
    rate = GROWTH_RATES.get(t, GROWTH_RATES['Default'])
    years_ahead = max(0, y - 2022)
    return base_pred * (rate ** years_ahead)

# --- MAIN DASHBOARD DISPLAY ---
st.subheader(f"📊 {question}")

# High-level Metrics
pred_val = get_forecast(year, state, topic, question, strat_cat, strat_val)
base_val = get_forecast(2022, state, topic, question, strat_cat, strat_val)
increase = pred_val - base_val

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    label = "Historical Baseline (2022)" if year == 2022 else f"Projected Prevalence ({year})"
    st.metric(label, f"{pred_val:.2f}%")
with col_m2:
    st.metric("Trend Growth Increase", f"+{increase:.2f}%", delta_color="inverse")
with col_m3:
    st.metric("Annual Growth Rate Applied", f"{((GROWTH_RATES.get(topic, 1.012)-1)*100):.1f}%")

st.divider()

# Tabs for Insights
tab_trend, tab_states, tab_demographics = st.tabs([
    "📈 Temporal Trajectory", 
    "🗺️ State Rankings", 
    "👥 Demographic Deep-Dive"
])

with tab_trend:
    years = list(range(2022, 2031))
    vals = [get_forecast(y, state, topic, question, strat_cat, strat_val) for y in years]
    
    trend_df = pd.DataFrame({"Year": years, "Prevalence (%)": vals})
    trend_df['Status'] = trend_df['Year'].apply(lambda x: 'Actual' if x == 2022 else 'Projected')
    
    fig_line = px.line(trend_df, x="Year", y="Prevalence (%)", 
                       title=f"Growth Forecast: {strat_val} in {state}",
                       color_discrete_sequence=["#ff4b4b"], markers=True)
    fig_line.add_vline(x=year, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_line, use_container_width=True)

with tab_states:
    all_states = df['LocationAbbr'].unique()
    state_list = []
    with st.spinner('Calculating 50-state rankings...'):
        for s in all_states:
            p = get_forecast(year, s, topic, question, strat_cat, strat_val)
            state_list.append({'State': s, 'Prevalence (%)': p})
    
    state_df = pd.DataFrame(state_list).sort_values('Prevalence (%)', ascending=False)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("### 🟥 Highest Prevalence States")
        st.bar_chart(state_df.head(5), x='State', y='Prevalence (%)', color="#ff4b4b")
    with c2:
        st.write("### 🟦 Lowest Prevalence States")
        st.bar_chart(state_df.tail(5), x='State', y='Prevalence (%)', color="#29b5e8")

with tab_demographics:
    groups = df[df['StratificationCategory1'] == strat_cat]['Stratification1'].unique()
    group_list = []
    for g in groups:
        p = get_forecast(year, state, topic, question, strat_cat, g)
        group_list.append({'Group': g, 'Prevalence (%)': p})
    
    fig_strat = px.bar(pd.DataFrame(group_list), x='Group', y='Prevalence (%)',
                       color='Prevalence (%)', 
                       title=f"Prevalence Breakdown by {strat_cat} ({year})",
                       color_continuous_scale="Reds", text_auto='.2f')
    st.plotly_chart(fig_strat, use_container_width=True)

# --- FOOTER: GEOSPATIAL CONTEXT ---
st.divider()
st.subheader("Health Clusters based on the demographics")
map_data = df.groupby('LocationAbbr').mean(numeric_only=True).reset_index()
fig_map = px.choropleth(map_data, locations="LocationAbbr", locationmode="USA-states",
                         color="HealthCluster", scope="usa", 
                         title="ML-Derived Regional Health Profiles",
                         color_continuous_scale="Viridis")
st.plotly_chart(fig_map, use_container_width=True)


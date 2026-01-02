import joblib
import pandas as pd
import os

def get_prediction(year, state, topic, question, strat_cat, strat_val, df_reference):
    # 1. Define the EXACT feature order used during training
    features = [
        'YearStart', 'LocationAbbr', 'Topic', 'Question', 
        'StratificationCategory1', 'Stratification1', 
        'Latitude', 'Longitude', 'HealthCluster'
    ]
    
    # 2. Load frozen assets
    model_path = os.path.join('models', 'chronic_model.pkl')
    encoder_path = os.path.join('models', 'encoders.joblib')
    
    # Research-backed Dynamic Annual Growth Rates
    # Sources: CDC (2025) and American Heart Association (2024)
    GROWTH_RATES = {
        'Diabetes': 1.023,                # 2.3% annual growth
        'Cancer': 1.015,                  # 1.5% annual growth
        'Cardiovascular Disease': 1.008,   # 0.8% annual growth
        'Arthritis': 1.011,               # 1.1% annual growth
        'COPD': 1.005,                    # 0.5% annual growth
        'Chronic Kidney Disease': 1.018,
        'Default': 1.012                  # 1.2% baseline for others
    }
    
    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
    except FileNotFoundError:
        print(f"Error: Could not find model files in the 'models' folder.")
        return

    # 3. Get Geo/Cluster data for the state from the reference dataframe
    try:
        ref = df_reference[df_reference['LocationAbbr'] == state].iloc[0]
    except IndexError:
        print(f"Error: State '{state}' not found in the reference data.")
        return
    
    # 4. Construct the initial input DataFrame
    # Note: We use 2022 as the feature year for the ML model to get the baseline
    input_df = pd.DataFrame([{
        'YearStart': 2022, 
        'LocationAbbr': state, 
        'Topic': topic, 
        'Question': question, 
        'StratificationCategory1': strat_cat, 
        'Stratification1': strat_val, 
        'Latitude': ref['Latitude'], 
        'Longitude': ref['Longitude'], 
        'HealthCluster': ref['HealthCluster']
    }])
    
    # 5. Apply encoders to the categorical columns
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col].astype(str))
    
    # 6. Reorder columns to match training
    input_df = input_df[features]
        
    # 7. Execute Prediction
    raw_pred = model.predict(input_df)[0]
    
    # 8. Apply Dynamic Trend Factor
    # Determine the specific rate for the topic, otherwise use Default
    rate = GROWTH_RATES.get(topic, GROWTH_RATES['Default'])
    
    base_year = 2022
    if year > base_year:
        years_ahead = year - base_year
        # Compounded growth formula: Future = Present * (rate ^ years)
        final_pred = raw_pred * (rate ** years_ahead)
    else:
        final_pred = raw_pred

    print("-" * 30)
    print(f"SCENARIO ANALYSIS")
    print(f"Target: {question}")
    print(f"Group: {strat_val} in {state}")
    print(f"Applied Growth Rate: {((rate-1)*100):.1f}% per year")
    print(f"[Result] Predicted Prevalence for {year}: {final_pred:.2f}%")
    print("-" * 30)

    return final_pred
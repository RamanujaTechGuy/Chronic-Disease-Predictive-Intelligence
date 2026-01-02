import pandas as pd
import numpy as np

def clean_chronic_data(file_path):
    print("Cleaning Data...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Keep only Crude Prevalence for metric consistency
    df = df[df['DataValueType'] == 'Crude Prevalence'].copy()
    
    # Remove rows without targets
    df = df.dropna(subset=['DataValue'])
    
    # Extract Geolocation
    def extract_coords(point):
        try:
            parts = point.replace('POINT (', '').replace(')', '').split()
            return float(parts[1]), float(parts[0]) # lat, long
        except: return np.nan, np.nan

    df['Latitude'], df['Longitude'] = zip(*df['Geolocation'].apply(extract_coords))
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Drop redundant columns
    cols_to_drop = ['Response','ResponseID', 'DataValueAlt','DataValueUnit', 'DataValueFootnote','DataValueFootnoteSymbol', 
                    'Geolocation','LocationDesc', 'DataSource', 'TopicID', 'QuestionID','StratificationCategoryID1', 'StratificationID1',
                     'StratificationCategory2', 'Stratification2','StratificationCategory3', 'Stratification3',]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    return df
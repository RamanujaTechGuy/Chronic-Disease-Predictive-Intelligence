from src.data_cleaning import clean_chronic_data
from src.clustering import Clustering
from src.model_pipeline import ChronicDiseaseModel
from predict_tool import get_prediction

# 1. RUN THE FULL PIPELINE
raw_csv = '.\\data\\raw\\U.S._Chronic_Disease_Indicators.csv'
df_cleaned = clean_chronic_data(raw_csv)
df_final = Clustering(df_cleaned)
df_final.to_csv("./data/CDC_Cleaned.csv")

pipeline = ChronicDiseaseModel()
score = pipeline.train_model(df_final)
pipeline.save()
print(f"Pipeline Complete. Model Accuracy (R2): {score:.4f}")

# 2. RUN A TEST " WHAT-IF"
get_prediction(2021, 'TX', 'Alcohol', 'Alcohol use among high school students', 'Grade', 'Grade 10', df_final)
get_prediction(2025, 'TN', 'Sleep', 'Short sleep duration among children aged 4 months to 14 years', 'Sex', 'Male', df_final)
get_prediction(2030, 'AL', 'Diabetes', 'Diabetes among adults', 'Sex', 'Male', df_final)
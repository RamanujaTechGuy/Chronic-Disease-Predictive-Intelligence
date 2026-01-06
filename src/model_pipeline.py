import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class ChronicDiseaseModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100,max_depth=20,min_samples_leaf=2, random_state=42, n_jobs=-1)
        self.encoders = {}
        self.features = ['YearStart', 'LocationAbbr', 'Topic', 'Question', 
                         'StratificationCategory1','Stratification1','Latitude', 'Longitude','HealthCluster']

    def train_model(self, df):
        print("----Training Production Model----")
        df_ml = df.copy()
        cat_cols = ['LocationAbbr','Topic','Question','StratificationCategory1','Stratification1']
        for col in cat_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            self.encoders[col] = le

        X= df_ml[self.features]
        y=df_ml['DataValue']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
        self.model.fit(X_train,y_train)
        print(f"On Train Data Accuracy:",self.model.score(X_train,y_train))
        print(f"On Test Data Accuracy:",self.model.score(X_test,y_test))
        return self.model.score(X_test,y_test)
    
    def save(self):
        joblib.dump(self.model,'models//chronic_model.pkl',compress=3)
        joblib.dump(self.encoders,'models//encoders.joblib')
        print("Assets Saved Successfully")
        

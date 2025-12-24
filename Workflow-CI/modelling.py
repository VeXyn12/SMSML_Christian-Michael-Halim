import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# 1. MEMBACA DATA LOKAL
# File CSV harus diletakkan di folder yang sama dengan modelling.py
dataset_name = "car_details_v4_preprocessing.csv" 
df = pd.read_csv(dataset_name)

# 2. PERSIAPAN DATA
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SET EXPERIMENT (Agar tidak tercampur dengan eksperimen manual)
mlflow.set_experiment("Car_Price_Automated_CI")

with mlflow.start_run(run_name="GitHub_Actions_Retraining"):
    # Gunakan Autolog agar ringkas dan pasti tercatat
    mlflow.sklearn.autolog()
    
    # 4. MODEL TRAINING
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. LOG MODEL SECARA EKSPLISIT (Untuk memastikan artifak tersimpan)
    mlflow.sklearn.log_model(model, artifact_path="model_ci")
    
    print("✅ Training otomatis selesai dan artifak telah dicatat.")

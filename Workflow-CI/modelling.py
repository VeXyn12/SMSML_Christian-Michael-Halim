import pandas as pd
import mlflow
import mlflow.sklearn
import os
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Konfigurasi DagsHub untuk CI
repo_owner = "VeXyn12" 
repo_name = "Eksperimen_SML_Christian-Michael-Halim"

# Mengambil token dari environment variable GitHub Actions
token = os.getenv("DAGSHUB_TOKEN")
if token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

# Load data menggunakan URL agar GitHub bisa mengaksesnya
url = 'https://drive.google.com/uc?id=1t8EFE8ZCQWHV71mNypKl1TQjwgi6Q55M'
df = pd.read_csv(url)

X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
mlflow.set_experiment("CI_Automated_Training")
with mlflow.start_run(run_name="GitHub_Actions_Run"):
    mlflow.sklearn.autolog()
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    print("Model training otomatis selesai!")

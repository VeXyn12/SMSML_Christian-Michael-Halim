import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# PENTING: Inisialisasi ulang di dalam script agar terhubung ke Cloud
repo_owner = "VeXyn12"
repo_name = "Eksperimen_SML_Christian-Michael-Halim"
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

# Load Data dari URL Drive
url = 'https://drive.google.com/uc?id=1t8EFE8ZCQWHV71mNypKl1TQjwgi6Q55M'
df = pd.read_csv(url)
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eksperimen
mlflow.set_experiment("Basic_Experiment")
with mlflow.start_run(run_name="RF_Autolog"):
    mlflow.sklearn.autolog() # Autolog akan otomatis merekam model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Berhasil mengirim Basic Experiment ke DagsHub!")

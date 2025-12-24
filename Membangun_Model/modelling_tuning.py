import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# PENTING: Inisialisasi ulang di dalam script agar terhubung ke Cloud
repo_owner = "VeXyn12"
repo_name = "Eksperimen_SML_Christian-Michael-Halim"
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

# Load Data
url = 'https://drive.google.com/uc?id=1t8EFE8ZCQWHV71mNypKl1TQjwgi6Q55M'
df = pd.read_csv(url)
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Advance_Experiment")
with mlflow.start_run(run_name="RF_Manual_Tuning"):
    # Hyperparameter Tuning
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, {'n_estimators': [50, 100], 'max_depth': [None, 10]}, cv=3)
    grid.fit(X_train, y_train)

    # Manual Logging
    mlflow.log_params(grid.best_params_)
    y_pred = grid.predict(X_test)
    mlflow.log_metric("R2_Score", r2_score(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))

    # PENTING: Rekam Model sebagai Artifak
    mlflow.sklearn.log_model(grid.best_estimator_, name="model_advance")
    
    print("✅ Berhasil mengirim Advance Experiment ke DagsHub!")

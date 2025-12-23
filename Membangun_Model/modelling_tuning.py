import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# Inisialisasi DagsHub di dalam script
dagshub.init(repo_owner="VeXyn12", repo_name="Eksperimen_SML_Christian-Michael-Halim", mlflow=True)

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
    
    # MANUAL LOGGING (Wajib untuk Skilled/Advance)
    mlflow.log_params(grid.best_params_)
    y_pred = grid.predict(X_test)
    mlflow.log_metric("R2_Score", r2_score(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    
    # Log Model
    mlflow.sklearn.log_model(grid.best_estimator_, "model_advance")
    print("✅ Eksperimen Advance berhasil dikirim ke DagsHub!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

mlflow.set_experiment("Car_Price_Lokal_Tuning")

def train_and_tune():
    # Load Data
    try:
        url = 'https://drive.google.com/uc?id=1t8EFE8ZCQWHV71mNypKl1TQjwgi6Q55M'
        df = pd.read_csv(url)
        print("✅ Dataset berhasil dimuat.")
    except Exception as e:
        print(f"❌ Error: Gagal memuat dataset. {e}")
        return

    X = df.drop(columns=['Price'])
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SETUP GRID SEARCH (Hyperparameter Tuning)
    rf = RandomForestRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # EVALUASI MODEL
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n✅ Parameter Terbaik: {best_params}")
    print(f"Metrics -> MAE: {mae:.4f}, R2-Score: {r2:.4f}")

    # MLFLOW MANUAL LOGGING
    with mlflow.start_run(run_name="Best_Tuned_RF_CarPrice"):
        
        # Log Parameter & Metrik
        mlflow.log_params(best_params)
        mlflow.log_metrics({"MAE": mae, "R2_Score": r2})
        
        # Log Model dengan Signature
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(best_model, "model_advance", signature=signature)
        
        # Artefak 1: Plot Actual vs Predicted (Pengganti Confusion Matrix untuk Regresi)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted Car Price')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.tight_layout()
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")
        plt.close()
        
        # Artefak 2: Feature Importance
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Car Price Model)")
        plt.bar(range(X.shape[1]), importances[indices], align="center", color='green')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        print("Logging MLflow selesai. Artefak tersimpan.")

if __name__ == "__main__":
    train_and_tune()
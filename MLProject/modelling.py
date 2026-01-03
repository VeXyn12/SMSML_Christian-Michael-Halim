import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train():
    print("=== Training Car Price Model (Production/CI) ===")

    # 1. Load Data
    try:
        url = 'https://drive.google.com/uc?id=1t8EFE8ZCQWHV71mNypKl1TQjwgi6Q55M'
        df = pd.read_csv(url)
        print("✅ Dataset berhasil dimuat.")
    except Exception as e:
        print(f"❌ Error: Gagal memuat dataset. {e}")
        return

    X = df.drop(columns=['Price'])
    y = df['Price']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Model Definition (Gunakan Parameter Terbaik Hasil Tuning)
    best_params = {
        'n_estimators': 100,      
        'max_depth': 10,          
        'min_samples_split': 2,   
        'random_state': 42
    }
    
    print(f"Melatih model dengan parameter: {best_params}")
    
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    # 4. Evaluasi
    y_pred = model.predict(X_test)
    
    # Menggunakan metrik Regresi
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Metrics -> MAE: {mae:.4f}, R2-Score: {r2:.4f}")

    # 5. Log ke MLflow
    mlflow.log_params(best_params)
    mlflow.log_metrics({"MAE": mae, "R2_Score": r2})
    
    # Log Model dengan Signature
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, "model_car_price", signature=signature)
    
    print("✅ Model production berhasil disimpan.")

if __name__ == "__main__":
    train()

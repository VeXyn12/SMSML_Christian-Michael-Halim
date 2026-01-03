import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

mlflow.set_experiment("Car_Price_Lokal_Baseline")

def train_car_price():
    
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

    # Autologging
    mlflow.autolog()

    with mlflow.start_run(run_name="RF_CarPrice_Default_Lokal"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n--- HASIL EVALUASI LOKAL ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R2-Score                : {r2:.4f}")

if __name__ == "__main__":
    train_car_price()
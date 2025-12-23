import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

url = 'https://drive.google.com/uc?id=1t8EFE8ZCQWHV71mNypKl1TQjwgi6Q55M'
df = pd.read_csv(url)
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set Experiment
mlflow.set_experiment("Basic_Experiment")

with mlflow.start_run(run_name="RF_Autolog"):
    mlflow.sklearn.autolog() # Autolog untuk Basic
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

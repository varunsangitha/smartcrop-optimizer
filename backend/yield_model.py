import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
data = pd.read_parquet(
    "data/train-00000-of-00001.parquet"
)


# Input features
X = data[
    [
        "Year",
        "State",
        "Crop",
        "Season",
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"
    ]
]

# Target
y = data["Yield"]


# Convert categorical columns to numerical values
X = pd.get_dummies(
    X,
    columns=["State", "Crop", "Season"]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Train
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Crop Yield Prediction Model")
print("---------------------------")
print("MAE:", round(mae, 4))
print("R2 Score:", round(r2, 4))


# Save model and feature names
model_data = {
    "model": model,
    "features": X.columns.tolist()
}

joblib.dump(
    model_data,
    "models/yield_model.pkl"
)

print("Yield model saved successfully!")
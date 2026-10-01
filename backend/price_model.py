import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load price dataset
data = pd.read_csv("data/crop_prices.csv")


# Crop-name mapping
crop_mapping = {
    "apple": "Apple",
    "banana": "Banana",
    "coconut": "Coconut",
    "coffee": "Coffee",
    "cotton": "Cotton",
    "grapes": "Grapes",
    "jute": "Jute",
    "maize": "Maize",
    "mango": "Mango",
    "orange": "Orange",
    "papaya": "Papaya",
    "pomegranate": "Pomegranate",
    "rice": "Paddy(Common)"
}


# Keep only compatible crops
data["crop_key"] = data["commodity"].str.lower().str.strip()

data = data[data["crop_key"].isin(crop_mapping.keys())].copy()


# Convert date
data["date"] = pd.to_datetime(data["date"])

# Create numerical date features
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day


# Features
X = data[
    [
        "year",
        "month",
        "day",
        "min_price",
        "max_price"
    ]
]

# Target
y = data["modal_price"]


# Remove missing values
valid = X.notna().all(axis=1) & y.notna()

X = X[valid]
y = y[valid]


# Split chronologically
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# Create model
model = RandomForestRegressor(
    n_estimators=150,
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


print("Crop Price Prediction Model")
print("---------------------------")
print("Rows used:", len(X))
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))


# Save model
model_data = {
    "model": model,
    "features": X.columns.tolist()
}

joblib.dump(
    model_data,
    "models/price_model.pkl"
)

print("Price model saved successfully!")
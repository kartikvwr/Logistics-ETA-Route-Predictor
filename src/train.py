import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from data_loader import DataLoader
from feature_engineering import FeatureEngineer


# -----------------------------
# Load Dataset
# -----------------------------

loader = DataLoader("data/raw/delivery_data.csv")

df = loader.load_data()
df = loader.clean_data(df)

engineer = FeatureEngineer()
df = engineer.create_features(df)

# -----------------------------
# Features & Target
# -----------------------------

X = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]

categorical_features = [
    "Weather",
    "Traffic",
    "Vehicle",
    "Area",
    "Category",
]

numeric_features = [
    col
    for col in X.columns
    if col not in categorical_features
]

# -----------------------------
# Preprocessing
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "num",
            "passthrough",
            numeric_features,
        ),
    ]
)

# -----------------------------
# Model
# -----------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    objective="reg:squarederror",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining Model...\n")

pipeline.fit(X_train, y_train)

print("Training Complete!\n")

# -----------------------------
# Predictions
# -----------------------------

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"MAE  : {mae:.2f} minutes")
print(f"RMSE : {rmse:.2f} minutes")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Save Model
# -----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/eta_model.pkl")

print("\nModel saved to models/eta_model.pkl")
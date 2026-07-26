import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/eta_model.pkl")

sample = {
    "Agent_Age": 28,
    "Agent_Rating": 4.8,
    "Weather": "Sunny",
    "Traffic": "Medium",
    "Vehicle": "motorcycle",
    "Area": "Urban",
    "Category": "Food",
    "Distance_km": 7.5,
    "Day": 26,
    "Month": 7,
    "Weekday": 5,
    "Order_Hour": 18,
    "Pickup_Delay_Min": 10,
}

df = pd.DataFrame([sample])

prediction = model.predict(df)

print("=" * 50)
print(f"Predicted ETA: {prediction[0]:.2f} minutes")
print("=" * 50)
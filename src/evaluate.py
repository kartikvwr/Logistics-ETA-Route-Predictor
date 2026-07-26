import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from feature_engineering import FeatureEngineer

loader = DataLoader("data/raw/delivery_data.csv")

df = loader.clean_data(loader.load_data())

engineer = FeatureEngineer()
df = engineer.create_features(df)

X = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = joblib.load("models/eta_model.pkl")

PredictionErrorDisplay.from_estimator(
    model,
    X_test,
    y_test,
    kind="actual_vs_predicted",
)

plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png")
plt.show()
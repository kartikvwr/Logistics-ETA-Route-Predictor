# 🚚 Logistics ETA Prediction System

A machine learning project that predicts the estimated delivery time of an order using factors such as delivery distance, traffic conditions, weather, vehicle type, and agent details. The project includes an end-to-end ML pipeline along with an interactive Streamlit web application for real-time predictions.

---

## 📌 Overview

The objective of this project is to estimate delivery time as accurately as possible using historical delivery data. The workflow includes data preprocessing, feature engineering, model training using XGBoost, and deployment through a Streamlit interface.

The application allows users to modify delivery parameters and instantly predict the estimated delivery time.

---

## ✨ Features

- End-to-end machine learning pipeline
- Data cleaning and preprocessing
- Feature engineering using geographical and temporal information
- XGBoost regression model for ETA prediction
- Interactive Streamlit web application
- Saved trained model using Joblib
- Simple and user-friendly interface for testing different delivery scenarios

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Joblib

---

## 📂 Project Structure

```text
Logistics-ETA-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── raw/
│       └── delivery_data.csv
│
├── models/
│   └── eta_model.pkl
│
├── results/
│   ├── app.png
│   └── metrics.txt
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
└── notebooks/
```

---

## 📊 Dataset

**Dataset:** Amazon Delivery Dataset

After preprocessing:

- Original records: **43,739**
- Cleaned records used for training: **43,594**

### Features Used

- Agent Age
- Agent Rating
- Weather
- Traffic
- Vehicle Type
- Delivery Area
- Product Category
- Delivery Distance
- Pickup Delay
- Order Hour
- Day
- Month
- Weekday

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| MAE | **17.11 minutes** |
| RMSE | **22.08 minutes** |
| R² Score | **0.8146** |

---

## 📸 Application Preview

![Application Screenshot](results/app.png)


---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/kartikvwr/Logistics-ETA-Predictor.git
```

### Move into the project

```bash
cd Logistics-ETA-Predictor
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 💡 Future Improvements

- Integrate live weather data
- Route optimisation using map APIs
- Hyperparameter tuning
- Cloud deployment
- Real-time traffic integration

---

## 👨‍💻 Author

**Kartik V W R**

GitHub: https://github.com/kartikvwr

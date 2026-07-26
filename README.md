# 🚚 Logistics ETA Prediction System

A machine learning-based web application that predicts delivery time for logistics orders using delivery, traffic, weather, and agent information. The project is built with XGBoost and deployed through a Streamlit interface for real-time predictions.

---

## Features

- Predicts delivery ETA using an XGBoost regression model
- Data cleaning and feature engineering pipeline
- Interactive Streamlit web application
- Trained on over **43,000** delivery records
- Model persistence using Joblib
- Easy-to-use interface for testing different delivery scenarios

---

## Tech Stack

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib
- Matplotlib

---

## Project Structure

```
Logistics-ETA-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│
├── models/
│   └── eta_model.pkl
│
├── results/
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

## Dataset

Amazon Delivery Dataset

After preprocessing:

- **43,594** training samples
- **16** original features
- Engineered features include:
  - Delivery distance
  - Pickup delay
  - Order hour
  - Day of week
  - Month

---

## Model Performance

| Metric | Value |
|--------|-------:|
| MAE | **17.11 min** |
| RMSE | **22.08 min** |
| R² Score | **0.8146** |

---

## Screenshots

### Home Page

> Add `results/app.png`

### Prediction

> Add `results/prediction.png`

---

## Installation

Clone the repository

```bash
git clone https://github.com/kartikvwr/Logistics-ETA-Predictor.git
```

Move into the project directory

```bash
cd Logistics-ETA-Predictor
```

Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

- Route optimisation using map APIs
- Live weather integration
- Traffic prediction using real-time data
- Hyperparameter tuning
- Model deployment on cloud platforms

---

## Author

**Kartik V W R**

GitHub: https://github.com/kartikvwr

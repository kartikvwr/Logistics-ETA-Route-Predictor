# Logistics ETA & Route Predictor

An end-to-end machine learning prototype executing predictive regressions to estimate dynamic last-mile delivery times under volatile real-world constraints.

## Key Features & Architecture
- **Gradient Boosted Regression:** Employs an XGBoost framework engineered to estimate travel durations based on external environmental inputs.
- **Dynamic Feature Ingestion:** Designed to parse spatial-temporal features out of dynamic JSON payloads representing live data streams from third-party map and regional weather systems.
- **Performance Evaluation:** Evaluates pipeline predictions against validation test batches using continuous Root Mean Squared Error (RMSE) scoring metrics.

## Setup & Implementation Details
- Core: XGBoost, Scikit-Learn
- Data Structuring: Pandas, NumPy
- Validation Target: Evaluation via RMSE

## Project Status
- [x] Dataset structure definitions and core feature mapping layouts.
- [x] XGBoost regression pipeline initialization configurations.
- [ ] Latency validation checks.

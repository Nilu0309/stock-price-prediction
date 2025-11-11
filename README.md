# 📈 Deep Learning for Stock Price Prediction — Multivariate LSTM Model

This project tackles **multivariate time series forecasting** using deep learning (LSTM + Optuna + Captum).  
The goal is to predict **stock prices for 442 companies** using historical data.

## Key Features
- Data normalization & windowing for time-series input
- LSTM model with configurable layers and dropout
- Hyperparameter tuning via Optuna
- Interpretability via Captum (Integrated Gradients & Saliency)
- Final predictions in submission-ready CSV format

## Run Instructions
```bash
pip install -r requirements.txt
python src/stock_price_prediction.py
```

## License
MIT License © 2025 Nilufar Ibrahimli

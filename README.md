# Deep Learning for Stock Price Prediction — Multivariate LSTM Model

### Overview
This project tackles **multivariate time series forecasting** using **deep learning**.  
The goal is to train a model that can accurately predict **stock prices for 442 companies on April 1st, 2022**, using historical market data.

Key methods used:
- Long Short-Term Memory (**LSTM**) Recurrent Neural Networks  
- Automated hyperparameter tuning with **Optuna**  
- Model interpretability with **Captum (Integrated Gradients & Saliency)**  

---

### Data Preparation

The dataset originally had **companies as rows** and **dates as columns**.  
To feed it into an LSTM, it’s **transposed** so that:
- Each row = a time step (day)  
- Each column = a company (feature)

Then:
- Each company’s prices are **normalized** individually with `MinMaxScaler`
- **Sliding windows** of 60 days are used as input to predict the next day’s prices  
- Data is split into **80% training** and **20% validation**

```python
WINDOW_SIZE = 60
input_window = normalized_data.iloc[i:i+WINDOW_SIZE].values
target_day = normalized_data.iloc[i+WINDOW_SIZE].values
```
---

### Model Architecture

The model is a stacked LSTM network designed for multivariate forecasting:

```python
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.output_layer(out[:, -1, :])
```
-**Input dimension:** number of companies

-**Output dimension:** number of companies

-**Loss function:** Mean Squared Error (MSE)

---

### Hyperparameter Optimization with Optuna

To achieve optimal model performance, **Optuna** is used for automated hyperparameter tuning.  
Optuna performs multiple trials to minimize the **validation loss (MSE)** by testing various parameter combinations.

Parameters tuned:
- **Hidden layer size** (`hidden_dim`)
- **Number of LSTM layers** (`num_layers`)
- **Dropout rate** (`dropout`)
- **Learning rate** (`learning_rate`)
- **Batch size** (`batch_size`)

Each trial trains the model with a different set of hyperparameters, evaluates it on the validation set,  
and returns the average validation loss.

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
```

---
### Final Training

After Optuna identifies the best hyperparameters, the model is retrained on the **full training dataset** using the optimal configuration.  
This phase focuses on long-term training stability and convergence to achieve the lowest possible error.

The final LSTM model is initialized as follows:

```python
final_model = StockLSTM(
    input_dim=X_all.shape[2],
    hidden_dim=optimal_params["hidden_dim"],
    output_dim=Y_all.shape[1],
    num_layers=optimal_params["num_layers"],
    dropout=optimal_params["dropout"]
)
```

---
### Predictions and Inverse Scaling

After training is complete, the model predicts **next-day stock prices** using the most recent 60 days of normalized data.  
Since the model outputs values between 0 and 1 (due to MinMax scaling), these predictions must be **inverse-transformed**  
to restore them to their original price scale.

Each company’s predictions are scaled back individually using the corresponding `MinMaxScaler` saved earlier:

```python
predicted_original = scaler_dict[company].inverse_transform(predicted_scaled_output)
```

The final predictions are wrapped into a DataFrame and saved in the submission format:
```
ID,value
Company_1,Price_1
Company_2,Price_2
...
```
The output file is exported as:
```python
submission_df.to_csv("predicted_prices.csv")
```
Result:
A clean, submission-ready CSV containing the predicted stock prices for all 442 companies on the next trading day

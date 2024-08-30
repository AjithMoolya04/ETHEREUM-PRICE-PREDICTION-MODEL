
# Ethereum Price Prediction Using MLP Regression

## Overview

A mini project involving an Ethereum price prediction model has been completed. This project utilized historical price data and machine learning algorithms to forecast future Ethereum prices. The model aims to assist investors in making informed decisions by providing accurate price predictions.

## Table of Contents

- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Implementation](#model-implementation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Dataset

- **Source**: Kaggle
- **Time Period**: January 2018 - July 2021
- **Features**:
  - `Date`: Date of the price record.
  - `Open`: Opening price of Ethereum on the given date.
  - `High`: Highest price on the given date.
  - `Low`: Lowest price on the given date.
  - `Close`: Closing price on the given date.
  - `Adj Close`: Adjusted closing price (not used in the model).
  - `Volume`: Trading volume on the given date.

## Feature Engineering

- Extracted `Year`, `Month`, and `Day` from the `Date` to capture seasonal and yearly trends.
- Selected features for the model:
  - `Year`
  - `Month`
  - `Day`
  - `Open`
  - `High`
  - `Low`
- **Excluded**: `Adj Close`, as it does not directly reflect market demand.

```python
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low']]
y = data['Close']
```

## Model Implementation

1. **Train-Test Split**: 
   - 80% for training, 20% for testing. 
   - Sequential split to maintain temporal order.

   ```python
   train_size = int(0.8 * len(data))
   X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
   ```

2. **Feature Scaling**:
   - Standardized features to have zero mean and unit variance.

   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
   y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
   ```

3. **MLP Regressor**:
   - Two hidden layers with 100 and 50 neurons.
   - ReLU activation function.
   - Adam optimizer, max 1000 iterations.

   ```python
   mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
   mlp_regressor.fit(X_train_scaled, y_train_scaled)
   ```

4. **Prediction**:
   - Make predictions on both training and test sets.

   ```python
   y_pred = mlp_regressor.predict(X_test_scaled)
   y_pred_train = mlp_regressor.predict(X_train_scaled)
   ```

## Results

- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - R-squared (R²)

- **Visualization**:
  - Line plots comparing actual vs. predicted prices for both training and testing datasets.

  ```python
  plt.plot(data["Date"][len(y_train):], y_test.values, label='Actual', color='blue')
  plt.plot(data["Date"][len(y_train):], y_pred, label='Predicted', color='red')
  plt.xlabel('Time')
  plt.ylabel('Price (INR)')
  plt.title('ETH Price Prediction')
  plt.legend()
  plt.show()
  ```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/eth-price-prediction.git
   cd eth-price-prediction
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the main script**:

   ```bash
   python main.py
   ```

2. **Explore the results**:
   - Review the evaluation metrics and plots to understand the model's performance.




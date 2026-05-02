from imports import *
import importlib
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data module (prepares nifty, X_test, y_test, y_test_raw, val_end, last_date, X_all_scaled)
data_mod = importlib.import_module('data')

# Ensure model artifacts are available: load saved model and scalers, or train if missing
models_dir = Path('models')
model_path = models_dir / 'nifty_mlp.keras'
x_scaler_path = models_dir / 'x_scaler.pkl'
y_scaler_path = models_dir / 'y_scaler.pkl'

if model_path.exists() and x_scaler_path.exists() and y_scaler_path.exists():
    model = tf.keras.models.load_model(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
else:
    # train (this will save artifacts)
    importlib.import_module('model')
    model = tf.keras.models.load_model(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

# Predict on test set and invert scaling
y_test_pred_scaled = model.predict(data_mod.X_test, verbose=0)
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
y_test_actual = data_mod.y_test_raw.ravel()

# Dates for test split
test_dates = data_mod.nifty.index[data_mod.val_end:]

# Compute metrics
mae = mean_absolute_error(y_test_actual, y_test_pred)
rmse = mean_squared_error(y_test_actual, y_test_pred) ** 0.5
with np.errstate(divide='ignore', invalid='ignore'):
    mape = np.mean(np.abs((y_test_actual - y_test_pred) / y_test_actual)) * 100

plt.figure(figsize=(12, 6))
plt.plot(data_mod.nifty.index, data_mod.nifty['Close'], label='Actual Prices')
plt.plot(test_dates, y_test_pred, color='red', linestyle='--', label='Predicted (Test)')

# Compute and show single next-day prediction point from last row
last_features = data_mod.X_all_scaled[-1].reshape(1, -1)
pred_next_scaled = model.predict(last_features, verbose=0)
predicted_price_next_day = y_scaler.inverse_transform(pred_next_scaled.reshape(-1, 1))[0, 0]
next_date = data_mod.last_date + pd.offsets.BDay(1)
plt.plot([data_mod.nifty.index[-1], next_date],
         [data_mod.nifty['Close'].iloc[-1], predicted_price_next_day],
         color='orange', marker='o', linestyle='--', label='Predicted Price (Next Day)')

plt.title(f'Actual Prices — Test predictions (MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
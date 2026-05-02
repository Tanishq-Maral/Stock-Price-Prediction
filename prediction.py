from imports import *
import joblib
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


# Paths for saved artifacts
MODEL_PATH = Path("models") / "nifty_mlp.keras"
X_SCALER_PATH = Path("models") / "x_scaler.pkl"
Y_SCALER_PATH = Path("models") / "y_scaler.pkl"


def _load_artifacts():
    if MODEL_PATH.exists() and X_SCALER_PATH.exists() and Y_SCALER_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        x_scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        return model, x_scaler, y_scaler
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Make a single next-day prediction using saved model.")
    parser.add_argument('--no-plot', action='store_true', help='Do not display plot')
    parser.add_argument('--n-days', type=int, default=120, help='Number of historical days to show in plot')
    args = parser.parse_args()

    # Try to load saved artifacts
    model, x_scaler, y_scaler = _load_artifacts()

    if model is None:
        # No saved model: train by importing model.py (this will create and save artifacts)
        print("Saved model not found — training model now (this may take a while)...")
        import importlib
        importlib.import_module('model')
        model, x_scaler, y_scaler = _load_artifacts()

    if model is None:
        raise SystemExit("Model artifacts not available after training. Aborting.")

    # Load data (this prepares X_all_scaled, nifty, last_date)
    import importlib
    data_mod = importlib.import_module('data')

    # Predict using the last available feature row
    last_date_features = data_mod.X_all_scaled[-1].reshape(1, -1)
    predicted_scaled = model.predict(last_date_features, verbose=0)
    predicted_price_next_day = y_scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
    next_date = data_mod.last_date + pd.offsets.BDay(1)

    print(f"Predicted price for the next day ({next_date}): {predicted_price_next_day:.4f}")

    # Persist prediction if DB reachable; otherwise skip gracefully
    prediction_row = pd.DataFrame([
        {
            'prediction_date': pd.Timestamp(next_date).date(),
            'predicted_close': float(predicted_price_next_day),
            'model_path': str(MODEL_PATH)
        }
    ])

    try:
        engine = create_engine(build_mysql_url())
        prediction_row.to_sql('predictions', engine, if_exists='append', index=False)
        print("Stored prediction in MySQL table 'predictions'.")
    except SQLAlchemyError as exc:
        print(f"MySQL write skipped for predictions: {exc}")

    # Plot recent history + predicted point unless disabled
    if not args.no_plot:
        window = args.n_days
        window = min(window, len(data_mod.nifty))
        recent = data_mod.nifty['Close'].iloc[-window:]

        plt.figure(figsize=(10, 5))
        plt.plot(recent.index, recent.values, label='Recent Actual Close')
        last_real_date = recent.index[-1]
        last_real_close = recent.values[-1]
        plt.plot([last_real_date, next_date], [last_real_close, predicted_price_next_day],
                 color='red', marker='o', linestyle='--', label='Predicted Next Day')
        plt.title(f'Last {window} days and Predicted Next Day')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()

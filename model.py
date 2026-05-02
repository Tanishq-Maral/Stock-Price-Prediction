from imports import *
from data import *

Path("models").mkdir(parents=True, exist_ok=True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    shuffle=False,
    verbose=1
)

model_save_path = Path("models") / "nifty_mlp.keras"
model.save(model_save_path)
joblib.dump(x_scaler, Path("models") / "x_scaler.pkl")
joblib.dump(y_scaler, Path("models") / "y_scaler.pkl")

print(f"Saved trained model to {model_save_path}")
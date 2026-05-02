from imports import *

nifty = yf.Ticker("^NSEI")
nifty = nifty.history(period='max')
del nifty['Dividends']
del nifty['Stock Splits']
nifty['Tomorrow'] = nifty['Close'].shift(-1)
nifty.dropna(inplace=True)
last_date = nifty.index[-1]

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
target_column = 'Tomorrow'

X_raw = nifty[feature_columns].values
y_raw = nifty[[target_column]].values

total_rows = len(nifty)
train_end = int(total_rows * 0.70)
val_end = int(total_rows * 0.85)

X_train_raw = X_raw[:train_end]
X_val_raw = X_raw[train_end:val_end]
X_test_raw = X_raw[val_end:]

y_train_raw = y_raw[:train_end]
y_val_raw = y_raw[train_end:val_end]
y_test_raw = y_raw[val_end:]

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train_raw)
X_val = x_scaler.transform(X_val_raw)
X_test = x_scaler.transform(X_test_raw)

y_train = y_scaler.fit_transform(y_train_raw).ravel()
y_val = y_scaler.transform(y_val_raw).ravel()
y_test = y_scaler.transform(y_test_raw).ravel()

# Preserve scaled full feature matrix for one-step-ahead prediction.
X_all_scaled = x_scaler.transform(X_raw)

mysql_url = build_mysql_url()

raw_prices_for_db = nifty[feature_columns + [target_column]].copy().reset_index()
if 'Date' in raw_prices_for_db.columns:
	raw_prices_for_db.rename(columns={'Date': 'date'}, inplace=True)

try:
	engine = create_engine(mysql_url)
	raw_prices_for_db.to_sql('raw_prices', engine, if_exists='replace', index=False)
	print("Stored raw prices in MySQL table 'raw_prices'.")
except SQLAlchemyError as exc:
	print(f"MySQL write skipped for raw_prices: {exc}")
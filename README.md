# Stock Price Prediction (MLP + MySQL)

This project predicts the next-day NIFTY close using a feedforward neural network (MLP), saves the trained model, and writes data/predictions to MySQL.

## What was added

- Chronological train/validation/test split (70/15/15)
- Saved trained model and scalers in `models/`
- MySQL integration for:
  - `raw_prices` table
  - `predictions` table

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set MySQL connection string (PowerShell example):

```powershell
$env:MYSQL_URL="mysql+pymysql://root:root@localhost:3306/stock_prediction"
```

3. Ensure the MySQL database exists:

```sql
CREATE DATABASE stock_prediction;
```

## Run

```bash
python main.py
```

## Notes

- If MySQL is not reachable, the script prints a warning and continues training/prediction.
- Model artifacts are saved to:
  - `models/nifty_mlp.keras`
  - `models/x_scaler.pkl`
  - `models/y_scaler.pkl`

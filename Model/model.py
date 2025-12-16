import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

# load
df = pd.read_csv("../Data/AllData_PolyesterForecast.csv")

# convert "period" to Date
df['Date'] = pd.to_datetime(df['period'].astype(str), format='%Y%m')
df = df.sort_values('Date')

log_cols = [
    "Polyester_yarn_price",
    "PTA_price",
    "MEG_price",
    "Global_clothing_export",
    "Global Price of Cotton (US Cents/lb)",
    "usd2vnd_exchange_monthly_avg_rate",
    "brent_oil_price_monthly_avg_rate"
]

for col in log_cols:
    df[f"log_{col}"] = np.log(df[col])

# target values
y = df["log_Polyester_yarn_price"]

df["log_Polyester_yarn_price_lag1"] = df["log_Polyester_yarn_price"].shift(1)
df_lag = df.dropna().reset_index(drop=True)

target_col = "log_Polyester_yarn_price"

# feature values
feature_cols = [
    "log_Polyester_yarn_price_lag1",
    "hs61_Vietnam_ExpPrice",
    "log_PTA_price",
    "log_MEG_price",
    "log_Global_clothing_export",
    "log_Global Price of Cotton (US Cents/lb)",
    "log_usd2vnd_exchange_monthly_avg_rate",
    "log_brent_oil_price_monthly_avg_rate"
]

X = df_lag[feature_cols]
y = df_lag[target_col]

def make_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

# Cross Validation
n_splits = 5 
tscv = TimeSeriesSplit(n_splits=n_splits)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = make_model()
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_val)

    y_val_real = np.exp(y_val.values)
    y_pred_real = np.exp(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
    mape = mean_absolute_percentage_error(y_val_real, y_pred_real) * 100

    fold_results.append({
        "fold": fold,
        "train_start": df_lag["Date"].iloc[train_idx[0]],
        "train_end": df_lag["Date"].iloc[train_idx[-1]],
        "val_start": df_lag["Date"].iloc[val_idx[0]],
        "val_end": df_lag["Date"].iloc[val_idx[-1]],
        "rmse": rmse,
        "mape": mape
    })

results_df = pd.DataFrame(fold_results)
print(results_df[["fold", "train_start", "train_end", "val_start", "val_end", "rmse", "mape"]])

print("\nCV average:")
print("RMSE mean:", results_df["rmse"].mean(), " / std:", results_df["rmse"].std())
print("MAPE mean:", results_df["mape"].mean(), " / std:", results_df["mape"].std())
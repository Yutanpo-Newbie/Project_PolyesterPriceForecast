import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import List, Dict


class ModelEvaluator:
    """モデル評価を担当するクラス"""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """評価メトリクスを計算"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return {"rmse": rmse, "mape": mape}

    def cross_validate(self, model_class, model_params: dict, 
                      X: pd.DataFrame, y: pd.Series, 
                      df_dates: pd.Series) -> pd.DataFrame:
        """時系列クロスバリデーションを実行"""
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # モデルの訓練
            model = model_class(model_params)
            model.build_model()
            model.train(X_train, y_train)
            
            # 予測
            y_pred_log = model.predict(X_val)
            
            # log -> 元スケールへ
            y_val_real = np.exp(y_val.values)
            y_pred_real = np.exp(y_pred_log)
            
            # メトリクス計算
            metrics = self.calculate_metrics(y_val_real, y_pred_real)
            
            fold_results.append({
                "fold": fold,
                "train_start": df_dates.iloc[train_idx[0]],
                "train_end": df_dates.iloc[train_idx[-1]],
                "val_start": df_dates.iloc[val_idx[0]],
                "val_end": df_dates.iloc[val_idx[-1]],
                **metrics
            })

        return pd.DataFrame(fold_results)

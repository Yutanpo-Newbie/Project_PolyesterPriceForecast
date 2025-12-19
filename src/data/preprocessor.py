import pandas as pd
import numpy as np
from typing import List


class TimeSeriesPreprocessor:
    """時系列データの前処理を担当するクラス"""

    def __init__(self, log_columns: List[str], target_column: str):
        self.log_columns = log_columns
        self.target_column = target_column

    def convert_period_to_date(self, df: pd.DataFrame, period_col: str = 'period') -> pd.DataFrame:
        """periodカラムをDate型に変換"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df[period_col].astype(str), format='%Y%m')
        df = df.sort_values('Date')
        return df

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """対数変換を適用"""
        df = df.copy()
        for col in self.log_columns:
            if col in df.columns:
                df[f"log_{col}"] = np.log(df[col])
        return df

    def create_lag_features(self, df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
        """ラグ特徴量を作成"""
        df = df.copy()
        df[f"log_{self.target_column}_lag{lag}"] = df[f"log_{self.target_column}"].shift(lag)
        df = df.dropna().reset_index(drop=True)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """全ての前処理を実行"""
        df = self.convert_period_to_date(df)
        df = self.apply_log_transform(df)
        df = self.create_lag_features(df)
        return df

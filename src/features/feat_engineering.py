import pandas as pd
from typing import List, Tuple


class FeatureEngineering:
    """特徴量エンジニアリングを担当するクラス"""

    def __init__(self, feature_columns: List[str], target_column: str):
        self.feature_columns = feature_columns
        self.target_column = target_column

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """特徴量と目的変数に分割"""
        X = df[self.feature_columns]
        y = df[self.target_column]
        return X, y

    def add_rolling_features(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """移動平均などの特徴量を追加（オプション）"""
        df = df.copy()
        # 将来の拡張用
        return df

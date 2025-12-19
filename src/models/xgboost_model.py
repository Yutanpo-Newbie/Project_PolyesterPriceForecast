from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from .model_base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoostモデルの実装"""

    def __init__(self, model_params: dict):
        super().__init__()
        self.model_params = model_params
        self.model = None

    def build_model(self):
        """XGBoostモデルを構築"""
        self.model = XGBRegressor(**self.model_params)
        return self.model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """モデルを訓練"""
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を実行"""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        return self.model.predict(X)

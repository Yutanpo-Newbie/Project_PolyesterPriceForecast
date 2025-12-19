from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """モデルの基底クラス"""

    def __init__(self):
        self.model = None

    @abstractmethod
    def build_model(self):
        """モデルを構築"""
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """モデルを訓練"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を実行"""
        pass

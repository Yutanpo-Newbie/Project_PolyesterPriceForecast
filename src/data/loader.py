import pandas as pd
from pathlib import Path


class DataLoader:
    """データ読み込みを担当するクラス"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """CSVファイルを読み込む"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pd.read_csv(self.file_path)
        return df

    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """必要なカラムが存在するか確認"""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        return True

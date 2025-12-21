import yaml
from pathlib import Path
import sys

# プロジェクトルートを取得
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.features.feat_engineering import FeatureEngineering
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import ModelEvaluator

def load_config(config_name: str = "config.yaml") -> dict:
    """設定ファイルを読み込む"""
    config_path = PROJECT_ROOT / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def main():
    # 設定読み込み
    config = load_config()

    # 絶対パスに変換
    raw_path_from_config = config['data']['raw_data_path']

    # 絶対パスかチェック
    if Path(raw_path_from_config).is_absolute():
        data_path = Path(raw_path_from_config)
    else:
        # 相対パスなのでPROJECT_ROOTと結合
        data_path = PROJECT_ROOT / raw_path_from_config

    print(f"Resolved data path: {data_path}")
    print(f"File exists: {data_path.exists()}")

    # data/rawディレクトリの内容を確認
    data_raw_dir = PROJECT_ROOT / "data" / "raw"
    print(f"\ndata/raw directory: {data_raw_dir}")
    print(f"Directory exists: {data_raw_dir.exists()}")

    if data_raw_dir.exists():
        print("Files in data/raw:")
        for file in data_raw_dir.iterdir():
            print(f"  - {file.name}")

    print("=" * 80)

    # データ読み込み
    loader = DataLoader(str(data_path))
    df = loader.load_data()
    print(f"Data loaded: {len(df)} rows")

    # 前処理
    preprocessor = TimeSeriesPreprocessor(
        log_columns=config['features']['log_columns'],
        target_column=config['features']['target_column'].replace('log_', '')
    )
    df_processed = preprocessor.preprocess(df)
    print(f"Data preprocessed: {len(df_processed)} rows")

    # 特徴量エンジニアリング
    feature_eng = FeatureEngineering(
        feature_columns=config['features']['feature_columns'],
        target_column=config['features']['target_column']
    )
    X, y = feature_eng.split_features_target(df_processed)
    print(f"Features: {X.shape}, Target: {y.shape}")

    # モデル評価
    evaluator = ModelEvaluator(n_splits=config['evaluation']['n_splits'])
    results = evaluator.cross_validate(
        model_class=XGBoostModel,
        model_params=config['model']['xgboost'],
        X=X,
        y=y,
        df_dates=df_processed['Date']
    )

    # 結果表示
    print("\n" + "="*80)
    print("Cross-Validation Results")
    print("="*80)
    print(results[["fold", "train_start", "train_end", "val_start", "val_end", "rmse", "mape"]])
    print("\nCV平均:")
    print(f"RMSE mean: {results['rmse'].mean():.4f} ± {results['rmse'].std():.4f}")
    print(f"MAPE mean: {results['mape'].mean():.2f}% ± {results['mape'].std():.2f}%")

if __name__ == "__main__":
    main()


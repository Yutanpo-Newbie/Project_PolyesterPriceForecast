import yaml
from pathlib import Path
import sys
import pandas as pd

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

def run_experiment(config_name: str, experiment_name: str) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Config: {config_name}")
    print("=" * 80)
    
    # 設定読み込み
    config = load_config(config_name)
    
    # データパスを絶対パスに変換
    raw_path_from_config = config['data']['raw_data_path']
    if Path(raw_path_from_config).is_absolute():
        data_path = Path(raw_path_from_config)
    else:
        data_path = PROJECT_ROOT / raw_path_from_config
    
    # データ読み込み
    loader = DataLoader(str(data_path))
    df = loader.load_data()
    
    # 前処理
    preprocessor = TimeSeriesPreprocessor(
        log_columns=config['features']['log_columns'],
        target_column=config['features']['target_column'].replace('log_', '')
    )
    df_processed = preprocessor.preprocess(df)
    
    # 特徴量エンジニアリング
    feature_eng = FeatureEngineering(
        feature_columns=config['features']['feature_columns'],
        target_column=config['features']['target_column']
    )
    X, y = feature_eng.split_features_target(df_processed)
    
    print(f"Features used ({len(X.columns)}): {list(X.columns)}")
    
    # モデル評価
    evaluator = ModelEvaluator(n_splits=config['evaluation']['n_splits'])
    results = evaluator.cross_validate(
        model_class=XGBoostModel,
        model_params=config['model']['xgboost'],
        X=X,
        y=y,
        df_dates=df_processed['Date']
    )
    
    # 実験名を追加
    results['experiment'] = experiment_name
    results['n_features'] = len(X.columns)
    
    return results

def compare_experiments():
    """複数の実験を比較"""
    
    experiments = [
        {
            'config': 'config.yaml',
            'name': 'Full Features (7 features)'
        },
        {
            'config': 'config_reduced.yaml',
            'name': 'Reduced Features (5 features)'
        }
    ]
    
    all_results = []
    
    for exp in experiments:
        results = run_experiment(exp['config'], exp['name'])
        all_results.append(results)
    
    # 結果を結合
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # 比較表示
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    summary = combined_results.groupby('experiment').agg({
        'rmse': ['mean', 'std'],
        'mape': ['mean', 'std'],
        'n_features': 'first'
    }).round(4)
    
    print(summary)
    
    # 詳細な比較
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    for exp_name in combined_results['experiment'].unique():
        exp_data = combined_results[combined_results['experiment'] == exp_name]
        print(f"\n{exp_name}:")
        print(f"  RMSE: {exp_data['rmse'].mean():.4f} ± {exp_data['rmse'].std():.4f}")
        print(f"  MAPE: {exp_data['mape'].mean():.2f}% ± {exp_data['mape'].std():.2f}%")
        print(f"  Features: {exp_data['n_features'].iloc[0]}")
    
    # 改善率を計算
    full_rmse = combined_results[combined_results['experiment'] == 'Full Features (7 features)']['rmse'].mean()
    reduced_rmse = combined_results[combined_results['experiment'] == 'Reduced Features (5 features)']['rmse'].mean()
    
    improvement = ((full_rmse - reduced_rmse) / full_rmse) * 100
    
    print("\n" + "=" * 80)
    print("IMPACT ANALYSIS")
    print("=" * 80)
    if improvement > 0:
        print(f"Removing 2 features IMPROVED performance by {improvement:.2f}%")
    else:
        print(f"Removing 2 features DEGRADED performance by {abs(improvement):.2f}%")
    
    # 結果をCSVに保存
    output_path = PROJECT_ROOT / "results" / "feature_comparison.csv"
    output_path.parent.mkdir(exist_ok=True)
    combined_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return combined_results

if __name__ == "__main__":
    compare_experiments()
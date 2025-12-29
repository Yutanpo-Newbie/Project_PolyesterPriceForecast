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
    """
    実験を実行して結果を返す
    
    Args:
        config_name: 設定ファイル名
        experiment_name: 実験名（結果に付与）
    
    Returns:
        結果のDataFrame
    """
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
    
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    
    # モデルパラメータを表示
    model_params = config['model']['xgboost']
    print(f"Model params:")
    print(f"  n_estimators={model_params['n_estimators']}")
    print(f"  learning_rate={model_params['learning_rate']}")
    print(f"  max_depth={model_params['max_depth']}")
    print(f"  subsample={model_params.get('subsample', 1.0)}")
    print(f"  colsample_bytree={model_params.get('colsample_bytree', 1.0)}")
    
    # モデル評価
    evaluator = ModelEvaluator(n_splits=config['evaluation']['n_splits'])
    results = evaluator.cross_validate(
        model_class=XGBoostModel,
        model_params=model_params,
        X=X,
        y=y,
        df_dates=df_processed['Date']
    )
    
    # 実験情報を追加
    results['experiment'] = experiment_name
    results['config_file'] = config_name
    results['n_features'] = len(X.columns)
    results['n_estimators'] = model_params['n_estimators']
    results['learning_rate'] = model_params['learning_rate']
    results['max_depth'] = model_params['max_depth']
    
    return results

def compare_experiments():
    """複数の実験を比較"""
    
    experiments = [
        {
            'config': 'config_base.yaml',
            'name': 'Baseline (default params, 7 features)'
        },
        {
            'config': 'config_noLag1.yaml',
            'name': 'No lag1 feature (default params, 6 features)'
        },
        {
            'config': 'config_noDemand.yaml',
            'name': 'no Demand features (default params, 5 features)'
        },
        {
            'config': 'config_noDemand_Lag1.yaml',
            'name': 'no Demand & Lag1 features (default params, 4 features)'
        }
    ]
    
    all_results = []
    
    for exp in experiments:
        try:
            results = run_experiment(exp['config'], exp['name'])
            all_results.append(results)
        except FileNotFoundError as e:
            print(f"\n  Warning: {e}")
            print(f"   Skipping: {exp['name']}")
            continue
    
    if not all_results:
        print("\n No experiments completed!")
        return None
    
    # 結果を結合
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # 比較表示
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    summary = combined_results.groupby(['experiment', 'config_file']).agg({
        'rmse': ['mean', 'std'],
        'mape': ['mean', 'std'],
        'n_features': 'first',
        'n_estimators': 'first',
        'learning_rate': 'first',
        'max_depth': 'first'
    }).round(4)
    
    print(summary)
    
    # 詳細な比較
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    for exp_name in combined_results['experiment'].unique():
        exp_data = combined_results[combined_results['experiment'] == exp_name]
        print(f"\n {exp_name}:")
        print(f"   Config: {exp_data['config_file'].iloc[0]}")
        print(f"   RMSE: {exp_data['rmse'].mean():.4f} ± {exp_data['rmse'].std():.4f}")
        print(f"   MAPE: {exp_data['mape'].mean():.2f}% ± {exp_data['mape'].std():.2f}%")
        print(f"   Features: {exp_data['n_features'].iloc[0]}")
        print(f"   Params: n_est={exp_data['n_estimators'].iloc[0]}, "
              f"lr={exp_data['learning_rate'].iloc[0]}, "
              f"depth={exp_data['max_depth'].iloc[0]}")
    
    # ベースラインとの比較
    baseline_data = combined_results[combined_results['experiment'].str.contains('Baseline')]
    
    if not baseline_data.empty:
        baseline_rmse = baseline_data['rmse'].mean()
        baseline_mape = baseline_data['mape'].mean()
        
        print("\n" + "=" * 80)
        print("IMPROVEMENT vs BASELINE")
        print("=" * 80)
        
        for exp_name in combined_results['experiment'].unique():
            if 'Baseline' in exp_name:
                continue
            
            exp_data = combined_results[combined_results['experiment'] == exp_name]
            exp_rmse = exp_data['rmse'].mean()
            exp_mape = exp_data['mape'].mean()
            
            rmse_improvement = ((baseline_rmse - exp_rmse) / baseline_rmse) * 100
            mape_improvement = ((baseline_mape - exp_mape) / baseline_mape) * 100
            
            print(f"\n{exp_name}:")
            if rmse_improvement > 0:
                print(f"  RMSE: ✓ IMPROVED by {rmse_improvement:.2f}%")
            else:
                print(f"  RMSE: ✗ DEGRADED by {abs(rmse_improvement):.2f}%")
            
            if mape_improvement > 0:
                print(f"  MAPE: ✓ IMPROVED by {mape_improvement:.2f}%")
            else:
                print(f"  MAPE: ✗ DEGRADED by {abs(mape_improvement):.2f}%")
    
    # 結果をCSVに保存
    output_path = PROJECT_ROOT / "results" / "feature_comparison.csv"
    output_path.parent.mkdir(exist_ok=True)
    combined_results.to_csv(output_path, index=False)
    print(f"\n Results saved to: {output_path}")
    
    return combined_results

if __name__ == "__main__":
    compare_experiments()
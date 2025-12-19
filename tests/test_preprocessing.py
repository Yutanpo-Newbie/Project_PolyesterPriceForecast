import pandas as pd
import numpy as np
from src.data.preprocessor import TimeSeriesPreprocessor


def test_log_transform():
    """対数変換のテスト"""
    df = pd.DataFrame({
        'Polyester_yarn_price': [100, 200, 300],
        'period': [202101, 202102, 202103]
    })

    preprocessor = TimeSeriesPreprocessor(
        log_columns=['Polyester_yarn_price'],
        target_column='Polyester_yarn_price'
    )

    result = preprocessor.apply_log_transform(df)

    assert 'log_Polyester_yarn_price' in result.columns
    assert np.allclose(result['log_Polyester_yarn_price'].values, np.log([100, 200, 300]))

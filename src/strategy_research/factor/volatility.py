"""
波动类：ATR, TR, BollingerWidth...
"""
import numpy as np
from pandas import DataFrame


def calc_atr(df: DataFrame,
             atr_period: int = 14,
             column_name_prefix: str | None = None) -> DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.assign(
        tmp_tr=lambda x: np.maximum(
            x['high'] - x['low'],
            np.maximum(
                (x['high'] - x['close'].shift(1)).abs(),
                (x['low'] - x['close'].shift(1)).abs()
            )
        ),
        tmp_atr=lambda x: x['tmp_tr'].ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean(),
    )

    if column_name_prefix is not None:
        df_copy.rename(columns={
            "tmp_tr": f"{column_name_prefix}_tr_{atr_period}",
            "tmp_atr": f"{column_name_prefix}_atr_{atr_period}"
        }, inplace=True)
    else:
        df_copy.rename(columns={
            "tmp_tr": f"tr_{atr_period}",
            "tmp_atr": f"atr_{atr_period}"
        }, inplace=True)


    return df_copy

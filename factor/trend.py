"""
趋势类：VI, SuperTrend, DualThrust...
"""
import numpy as np
from pandas import DataFrame


def calc_dual_thrust(df: DataFrame,
                     period: int,
                     upper_ratio: float,
                     lower_ratio: float,
                     column_name_prefix: str | None = None) -> DataFrame:
    df_copy = df.copy()
    df_copy['HH'] = df_copy['high'].rolling(period).max()  # N日最高价的最高价
    df_copy['HC'] = df_copy['close'].rolling(period).max()  # N日收盘价的最高价
    df_copy['LC'] = df_copy['close'].rolling(period).min()  # N日收盘价的最低价
    df_copy['LL'] = df_copy['low'].rolling(period).min()  # N日最低价的最低价
    df_copy['dt_range'] = np.maximum(df_copy['HH'] - df_copy['LC'], df_copy['HC'] - df_copy['LL'])
    # 上下轨
    df_copy['upper'] = df_copy['open'] + df_copy['dt_range'] * upper_ratio  # 上轨
    df_copy['lower'] = df_copy['open'] - df_copy['dt_range'] * lower_ratio  # 下轨

    df_copy.drop(columns=['HH', 'HC', 'LC', 'LL', 'dt_range'], inplace=True)

    if column_name_prefix is not None:
        df_copy.rename(columns={
            "upper": f"{column_name_prefix}_upper",
            "lower": f"{column_name_prefix}_lower"
        }, inplace=True)

    return df_copy



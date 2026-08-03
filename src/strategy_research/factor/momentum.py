"""
动量类：CMO, RSI, ROC, TRIX, MACD
"""
import numpy as np
import pandas as pd
from pandas import DataFrame


def calc_cmo(df: DataFrame,
             cmo_period: int = 6,
             cmo_signal_period: int = 4,
             column_name_prefix: str | None = None) -> DataFrame:
    df_copy = df.copy()
    delta = df_copy['close'].diff()
    # 分离上涨和下跌
    up_sum = np.zeros_like(delta)
    down_sum = np.zeros_like(delta)
    # 填充上涨和下跌数组
    up_sum[delta > 0] = delta[delta > 0]
    down_sum[delta < 0] = -delta[delta < 0]  # 注意要取绝对值
    # 计算上涨和下跌的滚动总和
    up_rolling_sum = pd.Series(up_sum).rolling(cmo_period).sum()
    down_rolling_sum = pd.Series(down_sum).rolling(cmo_period).sum()
    # 计算CMO值
    denominator = up_rolling_sum + down_rolling_sum
    cmo_values = np.where(
        denominator == 0,
        0.0,  # 如果分母为0（价格没波动），CMO 记为 0
        100 * ((up_rolling_sum - down_rolling_sum) / denominator)
    )
    cmo = pd.Series(cmo_values, index=df_copy.index)
    df_copy = df_copy.assign(
        cmo=cmo,
        cmo_signal=lambda x: x['cmo'].rolling(cmo_signal_period).mean(),
    )

    if column_name_prefix is not None:
        df_copy.rename(columns={
            "cmo": f"{column_name_prefix}_cmo",
            "cmo_signal": f"{column_name_prefix}_cmo_signal"
        }, inplace=True)

    return df_copy


def calc_vi(df: DataFrame,
            vi_period: int = 14,
            column_name_prefix: str | None = None) -> DataFrame:
    df_copy = df.copy()
    # 正/负向涡旋运动
    df_copy['plus_vm'] = np.abs(df_copy['high'] - df_copy['low'].shift(1))
    df_copy['minus_vm'] = np.abs(df_copy['low'] - df_copy['high'].shift(1))

    # N周期滚动求和
    df_copy['tr_sum'] = df_copy['tr'].rolling(vi_period).sum()
    df_copy['plus_vm_sum'] = df_copy['plus_vm'].rolling(vi_period).sum()
    df_copy['minus_vm_sum'] = df_copy['minus_vm'].rolling(vi_period).sum()

    # 涡旋指标
    df_copy['plus_vi'] = df_copy['plus_vm_sum'] / df_copy['tr_sum']
    df_copy['minus_vi'] = df_copy['minus_vm_sum'] / df_copy['tr_sum']

    df_copy.drop(
        columns=[
            'plus_vm',
            'minus_vm',
            'tr_sum',
            'plus_vm_sum',
            'minus_vm_sum'
        ],
        inplace=True
    )

    if column_name_prefix is not None:
        df_copy.rename(columns={
            "plus_vi": f"{column_name_prefix}_plus_vi",
            "minus_vi": f"{column_name_prefix}_minus_vi"
        }, inplace=True)

    return df_copy


def calc_trix(df: DataFrame,
              trix_period: int = 12,
              trix_signal_period: int = 9,
              column_name_prefix: str | None = None) -> DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.assign(
        ema1=lambda x: x['close'].ewm(span=trix_period, adjust=False, min_periods=trix_period).mean(),
        ema2=lambda x: x['ema1'].ewm(span=trix_period, adjust=False, min_periods=trix_period).mean(),
        ema3=lambda x: x['ema2'].ewm(span=trix_period, adjust=False, min_periods=trix_period).mean(),
        trix=lambda x: 100 * (x['ema3'] / x['ema3'].shift(1) - 1),
        trix_signal=lambda x: x['trix'].rolling(trix_signal_period).mean(),
    )

    df_copy.drop(columns=['ema1', 'ema2', 'ema3'], inplace=True)

    if column_name_prefix is not None:
        df_copy.rename(columns={
            "trix": f"{column_name_prefix}_trix",
            "trix_signal": f"{column_name_prefix}_trix_signal"
        }, inplace=True)

    return df_copy


def calc_macd(df: DataFrame,
              fast_period: int = 12,
              slow_period: int = 26,
              signal_period: int = 9) -> DataFrame:
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

    # 2. 计算 DIF (快线)
    dif = ema_fast - ema_slow

    # 3. 计算 DEA (慢线/信号线)
    dea = dif.ewm(span=signal_period, adjust=False).mean()

    # 4. 计算 MACD 柱
    # 通达信等软件标准是 (DIF - DEA) * 2，为了直观显示红绿柱
    macd_hist = (dif - dea) * 2

    # 将结果写入 DataFrame
    df['macd_diff'] = dif
    df['macd_dea'] = dea
    df['macd_hist'] = macd_hist

    return df

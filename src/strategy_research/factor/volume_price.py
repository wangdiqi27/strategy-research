"""
量价类：OBV, VPT, MFI, A/D Line...
"""
import numpy as np
from pandas import DataFrame


def calc_vpt(df: DataFrame,
             vpt_ma_period: int = 14,
             column_name_prefix: str | None = None) -> DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.assign(
        pct_change=lambda x: x['close'].pct_change(),
        vpt_increment=lambda x: x['volume'] * x['pct_change'],
        vpt=lambda x: x['vpt_increment'].fillna(0).cumsum() + x['volume'].iloc[0],
        vpt_ma=lambda x:x['vpt'].rolling(vpt_ma_period).mean()
    )

    df_copy.drop(columns=['pct_change', 'vpt_increment'], inplace=True)

    if column_name_prefix is not None:
        df_copy.rename(columns={
            "vpt": f"{column_name_prefix}_vpt",
            "vpt_ma": f"{column_name_prefix}_vpt_ma",
        }, inplace=True)

    return df_copy


def calc_ccl(df: DataFrame) -> DataFrame:
    df['ccl'] = df['open_interest'].diff()
    price_change = df['close'] - df['close'].shift(1)
    conditions = [
        (price_change > 0) & (df['ccl'] > 0),  # 价格涨 + 增仓 -> 多头增仓
        (price_change < 0) & (df['ccl'] > 0),  # 价格跌 + 增仓 -> 空头增仓
        (price_change > 0) & (df['ccl'] < 0),  # 价格涨 + 减仓 -> 空头减仓
        (price_change < 0) & (df['ccl'] < 0),  # 价格跌 + 减仓 -> 多头减仓
    ]
    choices = [
        'LONG_OPEN',
        'SHORT_OPEN',
        'SHORT_CLOSE',
        'LONG_CLOSE',
    ]
    # np.select 类似于 Excel 的 IF(Condition1, Value1, IF(Condition2...)) 功能
    df['ccl_type'] = np.select(conditions, choices, default='FLAT')

    return df

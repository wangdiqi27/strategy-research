"""
量价类：OBV, VPT, MFI, A/D Line...
"""
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

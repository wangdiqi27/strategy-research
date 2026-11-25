"""
抛物线转向指标（Parabolic SAR，Wilder 口径）：
逐行迭代计算（SAR 依赖上一根的状态，无法向量化），日线量级性能无压力。

输出列：
- sar:        当日 SAR 点位（上涨趋势中位于价格下方、下跌趋势中位于价格上方）
- sar_trend:  当日趋势方向，+1 = 上涨趋势（SAR 在下方），-1 = 下跌趋势（SAR 在上方）
"""

import numpy as np
from pandas import DataFrame


def calc_sar(df: DataFrame,
             af_start: float = 0.02,
             af_step: float = 0.02,
             af_max: float = 0.2) -> DataFrame:
    """
    标准 Wilder PSAR：
    - candidate = sar_prev + af_prev * (ep_prev - sar_prev)
    - 上涨趋势：candidate 不得高于前两根 low（i>=2 时取 min）；low 击穿 candidate 则反转为下跌，
      sar 跳到前一段的 ep（最高点），ep 重置为当日 low，af 重置为 af_start
    - 下跌趋势：candidate 不得低于前两根 high（i>=2 时取 max）；high 突破 candidate 则反转为上涨，
      sar 跳到前一段的 ep（最低点），ep 重置为当日 high，af 重置为 af_start
    - 触发新极值点时 af 增加 af_step，上限 af_max
    - 初始状态：假设上涨趋势，sar[0] = low[0]，ep[0] = high[0]，af[0] = af_start
    """
    df_copy = df.copy()
    n = len(df_copy)
    sar = np.full(n, np.nan)
    sar_trend = np.zeros(n, dtype=int)

    if n == 0:
        df_copy['sar'] = sar
        df_copy['sar_trend'] = sar_trend
        return df_copy

    high = df_copy['high'].to_numpy(dtype=float)
    low = df_copy['low'].to_numpy(dtype=float)

    # 初始：假设上涨趋势
    trend = 1
    cur_sar = low[0]
    ep = high[0]
    af = af_start

    sar[0] = cur_sar
    sar_trend[0] = trend

    for i in range(1, n):
        candidate = cur_sar + af * (ep - cur_sar)

        if trend == 1:
            # 上涨趋势：SAR 点位不得高于前两根 low
            candidate = min(candidate, low[i - 1], low[i - 2]) if i >= 2 else min(candidate, low[i - 1])

            if low[i] < candidate:
                # 反转为下跌趋势：SAR 跳到前一段极值高点
                trend = -1
                cur_sar = ep
                ep = low[i]
                af = af_start
            else:
                cur_sar = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            # 下跌趋势：SAR 点位不得低于前两根 high
            candidate = max(candidate, high[i - 1], high[i - 2]) if i >= 2 else max(candidate, high[i - 1])

            if high[i] > candidate:
                # 反转为上涨趋势：SAR 跳到前一段极值低点
                trend = 1
                cur_sar = ep
                ep = high[i]
                af = af_start
            else:
                cur_sar = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        sar[i] = cur_sar
        sar_trend[i] = trend

    df_copy['sar'] = sar
    df_copy['sar_trend'] = sar_trend
    return df_copy

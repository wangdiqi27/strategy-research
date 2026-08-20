import numpy as np
import pandas as pd
from pandas import DataFrame


def calc_swing_points(df: DataFrame,
                      left_bars_size: int = 2,
                      right_bars_size: int = 2,
                      exec_mode: bool = False) -> DataFrame:
    df = df.copy()
    ###################################################################################
    # 计算高点、低点

    # ========== Swing High ==========
    # 当前high > 左侧left_bars根K线的high最大值 且 > 右侧right_bars根K线的high最大值
    left_max = df['high'].rolling(
        window=left_bars_size,
        min_periods=left_bars_size).max().shift(1)
    right_max = df['high'].iloc[::-1].rolling(
        window=right_bars_size,
        min_periods=right_bars_size
    ).max().iloc[::-1].shift(-1)

    df['swing_high'] = (df['high'] > left_max) & (df['high'] > right_max)
    df['swing_high_price'] = np.where(df['swing_high'], df['high'], np.nan)

    # ========== Swing Low ==========
    # 当前low < 左侧left_bars根K线的low最小值 且 < 右侧right_bars根K线的low最小值
    left_min = df['low'].rolling(
        window=left_bars_size,
        min_periods=left_bars_size).min().shift(1)
    right_min = df['low'].iloc[::-1].rolling(
        window=right_bars_size,
        min_periods=right_bars_size
    ).min().iloc[::-1].shift(-1)

    df['swing_low'] = (df['low'] < left_min) & (df['low'] < right_min)
    df['swing_low_price'] = np.where(df['swing_low'], df['low'], np.nan)

    # 边界处理：前left_bars行和后right_bars行不可能成为swing point
    df.loc[:left_bars_size - 1, 'swing_high'] = False
    df.loc[:left_bars_size - 1, 'swing_low'] = False
    df.loc[len(df) - right_bars_size:, 'swing_high'] = False
    df.loc[len(df) - right_bars_size:, 'swing_low'] = False

    df.loc[:left_bars_size - 1, 'swing_high_price'] = np.nan
    df.loc[:left_bars_size - 1, 'swing_low_price'] = np.nan
    df.loc[len(df) - right_bars_size:, 'swing_high_price'] = np.nan
    df.loc[len(df) - right_bars_size:, 'swing_low_price'] = np.nan

    ###################################################################################
    # 计算分形结构 HH/HL/LH/LL

    # 获取swing点索引
    valid_high_indices = np.where(df['swing_high'].to_numpy())[0]
    valid_low_indices = np.where(df['swing_low'].to_numpy())[0]

    swing_high_prices = df['swing_high_price'].to_numpy()
    swing_low_prices = df['swing_low_price'].to_numpy()

    # ========== 计算High结构 ==========
    high_structure = np.array([None] * len(df))
    for i in range(1, len(valid_high_indices)):
        curr_idx = valid_high_indices[i]
        prev_idx = valid_high_indices[i - 1]
        if swing_high_prices[curr_idx] > swing_high_prices[prev_idx]:
            high_structure[curr_idx] = 'HH'
        else:
            high_structure[curr_idx] = 'LH'
    df['swing_high_structure'] = high_structure

    # ========== 计算Low结构 ==========
    low_structure = np.array([None] * len(df))
    for i in range(1, len(valid_low_indices)):
        curr_idx = valid_low_indices[i]
        prev_idx = valid_low_indices[i - 1]
        if swing_low_prices[curr_idx] > swing_low_prices[prev_idx]:
            low_structure[curr_idx] = 'HL'
        else:
            low_structure[curr_idx] = 'LL'
    df['swing_low_structure'] = low_structure

    # ========== 前向填充 ==========
    df['swing_high_struct_active'] = df['swing_high_structure'].ffill()
    df['swing_low_struct_active'] = df['swing_low_structure'].ffill()

    # ========== 向量化确定市场结构 ==========
    conditions_high = df['swing_high_struct_active'].to_numpy()
    conditions_low = df['swing_low_struct_active'].to_numpy()

    market_structure = np.full(len(df), 'undefined', dtype=object)
    trend_direction = np.full(len(df), 'neutral', dtype=object)

    # HH-HL: 强上升趋势
    mask_hh_hl = (conditions_high == 'HH') & (conditions_low == 'HL')
    market_structure[mask_hh_hl] = 'HH-HL'
    trend_direction[mask_hh_hl] = 'strong_uptrend'

    # LH-LL: 强下降趋势
    mask_lh_ll = (conditions_high == 'LH') & (conditions_low == 'LL')
    market_structure[mask_lh_ll] = 'LH-LL'
    trend_direction[mask_lh_ll] = 'strong_downtrend'

    # HH-LL: 上升趋势回调
    mask_hh_ll = (conditions_high == 'HH') & (conditions_low == 'LL')
    market_structure[mask_hh_ll] = 'HH-LL'
    trend_direction[mask_hh_ll] = 'uptrend_retracement'

    # LH-HL: 下降趋势反弹
    mask_lh_hl = (conditions_high == 'LH') & (conditions_low == 'HL')
    market_structure[mask_lh_hl] = 'LH-HL'
    trend_direction[mask_lh_hl] = 'downtrend_retracement'

    df['swing_market_structure'] = market_structure
    df['swing_trend_direction'] = trend_direction

    ###################################################################################
    # 计算交易信号
    prev_struct = df['swing_market_structure'].shift(1).to_numpy()
    curr_struct = df['swing_market_structure'].to_numpy()

    n = len(df)
    signals = np.zeros(n, dtype=int)
    signal_types = np.array([''] * n, dtype=object)

    # 做多信号
    mask = (prev_struct == 'LH-LL') & (curr_struct == 'HH-HL')
    signals[mask] = 1
    signal_types[mask] = 'trend_reversal_buy'

    mask = (prev_struct == 'HH-LL') & (curr_struct == 'HH-HL')
    signals[mask] = 1
    signal_types[mask] = 'pullback_buy'

    mask = (prev_struct == 'LH-HL') & (curr_struct == 'HH-HL')
    signals[mask] = 1
    signal_types[mask] = 'breakout_buy'

    # 做空信号
    mask = (prev_struct == 'HH-HL') & (curr_struct == 'LH-LL')
    signals[mask] = -1
    signal_types[mask] = 'trend_reversal_sell'

    mask = (prev_struct == 'LH-HL') & (curr_struct == 'LH-LL')
    signals[mask] = -1
    signal_types[mask] = 'pullback_sell'

    mask = (prev_struct == 'HH-LL') & (curr_struct == 'LH-LL')
    signals[mask] = -1
    signal_types[mask] = 'breakout_sell'

    df['swing_signal'] = signals
    df['swing_signal_type'] = signal_types

    if exec_mode:
        df['swing_high_confirmed'] = df['swing_high'].shift(right_bars_size + 1)
        df['swing_low_confirmed'] = df['swing_low'].shift(right_bars_size + 1)

        df['swing_high_price_confirmed'] = df['swing_high_price'].shift(right_bars_size + 1)
        df['swing_low_price_confirmed'] = df['swing_low_price'].shift(right_bars_size + 1)

        df['swing_latest_swing_high'] = df['swing_high_price_confirmed'].ffill()
        df['swing_latest_swing_low'] = df['swing_low_price_confirmed'].ffill()

        df['swing_market_structure_confirmed'] = df['swing_market_structure'].shift(right_bars_size + 1)
        df['swing_trend_direction_confirmed'] = df['swing_trend_direction'].shift(right_bars_size + 1)

        df['swing_signal_confirmed'] = df['swing_signal'].shift(right_bars_size + 1)
        df['swing_signal_type_confirmed'] = df['swing_signal_type'].shift(right_bars_size + 1)

    return df


def _calc_zigzag_structure(
    confirmed_pivots: list[tuple[int, float, int]]
)-> str:
    if len(confirmed_pivots) < 4:
        return ""

    latest_4_pivots = confirmed_pivots[-4:]

    high_pivots = [pivot for pivot in latest_4_pivots if pivot[2] == 1]
    low_pivots = [pivot for pivot in latest_4_pivots if pivot[2] == -1]

    if high_pivots[1][1] > high_pivots[0][1]:
        high_structure = "HH"
    else:
        high_structure = "LH"

    if low_pivots[1][1] > low_pivots[0][1]:
        low_structure = "HL"
    else:
        low_structure = "LL"


    return f"{high_structure}-{low_structure}"



def _calc_zigzag_core_logic_with_confirmation(
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        atr_ratio: float,
        min_swing_bars: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
    1. pivot_types: 极值点标记 (在极值点位置显示类型)
    2. confirmed_signals: 确认信号 (在确认突破的那根K线显示信号)
       1: 确认低点成立(做多信号), -1: 确认高点成立(做空信号)
    3. confirmed_prices: 被确认的极值点价格 (用于止损/止盈参考)
    4.
    """
    n = len(high)

    # 结果容器
    pivot_types = np.zeros(n, dtype=np.int8)  # 用于绘图
    confirmed_signals = np.zeros(n, dtype=np.int8)  # 用于回测信号
    confirmed_prices = np.full(n, np.nan)  # 用于回测价格参考
    confirmed_structure = np.zeros(n, dtype='U10')

    # 最近高点/低点序列
    latest_highs = np.full(n, np.nan)
    latest_highs_index = np.full(n, np.nan)
    latest_lows = np.full(n, np.nan)
    latest_lows_index = np.full(n, np.nan)

    # 用于保存已确认的极值点信息: (index, price, type) type: 1=高, -1=低
    confirmed_pivots: list[tuple[int, float, int]] = []

    # ... (状态变量初始化同前) ...
    is_warming_up = True
    warmup_high_idx = 0
    warmup_low_idx = 0
    current_extreme_idx = 0
    current_find_type = 0
    bars_since_last_pivot = 0

    for i in range(n):
        if atr[i] <= 0:
            continue

        threshold = atr[i] * atr_ratio

        # === 1. 预热逻辑 (同前) ===
        if is_warming_up:
            if high[i] > high[warmup_high_idx]:
                # 如果最新的最高值大于未确认的最高值，更新最高值索引
                warmup_high_idx = i
            if low[i] < low[warmup_low_idx]:
                # 如果最新的最低值小于未确认的最低值，更新最低值索引
                warmup_low_idx = i

            if (high[warmup_high_idx] - low[warmup_low_idx]) > threshold:
                # 确认第一个点
                if warmup_high_idx < warmup_low_idx:
                    # 如果先出现最高值，则确认为最高值
                    pivot_types[warmup_high_idx] = 1  # 标记位置

                    # 关键：在当前K线 i 产生确认信号
                    # 确认了高点，意味着后续看空，记为 -1
                    confirmed_signals[i] = -1

                    # 记录已确认最高值
                    confirmed_prices[i] = high[warmup_high_idx]

                    latest_highs[i] = high[warmup_high_idx]
                    latest_highs_index[i] = warmup_high_idx

                    # 开始找最低值
                    current_find_type = -1
                    current_extreme_idx = warmup_low_idx

                    confirmed_pivots.append((warmup_high_idx, high[warmup_high_idx], 1))
                else:
                    # 如果先出现最低值，则确认为最低值
                    pivot_types[warmup_low_idx] = -1

                    # 确认了低点，意味着后续看多，记为 1
                    confirmed_signals[i] = 1
                    # 记录已确认最高值
                    confirmed_prices[i] = low[warmup_low_idx]

                    latest_lows[i] = low[warmup_low_idx]
                    latest_lows_index[i] = warmup_low_idx

                    # 开始找最高值
                    current_find_type = 1
                    current_extreme_idx = warmup_high_idx

                    confirmed_pivots.append((warmup_low_idx, low[warmup_low_idx], -1))

                is_warming_up = False
                bars_since_last_pivot = 0
                continue

        # === 2. 常规逻辑 (增加信号输出) ===
        else:
            bars_since_last_pivot += 1

            if bars_since_last_pivot < min_swing_bars:
                if current_find_type == 1 and high[i] > high[current_extreme_idx]:
                    current_extreme_idx = i
                elif current_find_type == -1 and low[i] < low[current_extreme_idx]:
                    current_extreme_idx = i
                continue

            # --- 寻找高点 ---
            if current_find_type == 1:
                if high[i] > high[current_extreme_idx]:
                    current_extreme_idx = i

                # 确认反转：价格跌破 高点 - 阈值
                elif low[i] <= high[current_extreme_idx] - threshold:
                    # 1. 标记物理极值点 (用于绘图)
                    pivot_types[current_extreme_idx] = 1

                    # 2. 记录确认信号 (用于回测)
                    # 在当前 K 线 i 收盘时，我们确认了前面有个高点
                    confirmed_signals[i] = -1  # 高点确认，趋势转空
                    confirmed_prices[i] = high[current_extreme_idx]

                    latest_highs[i] = high[current_extreme_idx]
                    latest_highs_index[i] = current_extreme_idx

                    # 3. 状态切换
                    current_find_type = -1
                    current_extreme_idx = i
                    bars_since_last_pivot = 0

                    confirmed_pivots.append((current_extreme_idx, high[current_extreme_idx], 1))

                    zigzag_structure = _calc_zigzag_structure(confirmed_pivots)
                    confirmed_structure[i] = zigzag_structure

            # --- 寻找低点 ---
            elif current_find_type == -1:
                if low[i] < low[current_extreme_idx]:
                    current_extreme_idx = i

                # 确认反转：价格突破 低点 + 阈值
                elif high[i] >= low[current_extreme_idx] + threshold:
                    # 1. 标记物理极值点
                    pivot_types[current_extreme_idx] = -1

                    # 2. 记录确认信号
                    confirmed_signals[i] = 1  # 低点确认，趋势转多
                    confirmed_prices[i] = low[current_extreme_idx]

                    latest_lows[i] = low[current_extreme_idx]
                    latest_lows_index[i] = current_extreme_idx

                    # 3. 状态切换
                    current_find_type = 1
                    current_extreme_idx = i
                    bars_since_last_pivot = 0

                    confirmed_pivots.append((current_extreme_idx, low[current_extreme_idx], -1))
                    zigzag_structure = _calc_zigzag_structure(confirmed_pivots)
                    confirmed_structure[i] = zigzag_structure

    return (pivot_types,
            confirmed_signals,
            confirmed_prices,
            confirmed_structure,
            latest_highs,
            latest_lows,
            latest_highs_index,
            latest_lows_index)


# ============================================================
# Pandas 包装函数
# ============================================================

def calc_zigzag(
        df: DataFrame,
        atr_ratio: float = 1.5,
        min_swing_bars: int = 3,
        atr_col: str = 'atr_14'
) -> DataFrame:
    """
    生成可直接用于回测的 ZigZag 信号。
    """
    df = df.copy()

    # 提取数组
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    atr = df[atr_col].to_numpy()

    # 核心计算
    (pivot_types,
     confirmed_signals,
     confirmed_prices,
     structure,
     latest_highs,
     latest_lows,
     latest_highs_index,
     latest_lows_index) = \
        _calc_zigzag_core_logic_with_confirmation(high, low, atr, atr_ratio, min_swing_bars)

    # 1. 绘图用数据 (极值点物理位置)
    df['zigzag_pivot_type'] = pivot_types  # 1:High, -1:Low
    df['zigzag_pivot_price'] = np.where(pivot_types != 0,
                                        np.where(pivot_types == 1, high, low),
                                        np.nan)

    # 2. 回测用数据 (确认时刻)
    # signal: 1 (确认低点，做多), -1 (确认高点，做空)
    df['zigzag_signal'] = confirmed_signals

    # 被确认的那个极值点的价格 (可用作止损参考)
    df['zigzag_confirmed_pivot_price'] = confirmed_prices

    # 前向填充，得到最近一次确认的参考价位
    df['zigzag_ref_price'] = df['zigzag_confirmed_pivot_price'].ffill()

    structure_series = pd.Series(structure)
    structure_series = structure_series.replace('', np.nan)
    df['zigzag_structure'] = structure_series.ffill()
    df['zigzag_structure_confirmed'] = df['zigzag_structure'].shift(1)

    df['zigzag_structure'] = df['zigzag_structure'].fillna('')
    df['zigzag_structure_confirmed'] = df['zigzag_structure_confirmed'].fillna('')

    df['zigzag_last_high'] = latest_highs
    df['zigzag_last_high'] = df['zigzag_last_high'].ffill()
    df['zigzag_last_high_confirmed'] = df['zigzag_last_high'].shift(1)

    df['zigzag_last_low'] = latest_lows
    df['zigzag_last_low'] = df['zigzag_last_low'].ffill()
    df['zigzag_last_low_confirmed'] = df['zigzag_last_low'].shift(1)

    df['zigzag_last_high_index'] = latest_highs_index
    df['zigzag_last_high_index'] = df['zigzag_last_high_index'].ffill()
    df['zigzag_last_high_index_confirmed'] = df['zigzag_last_high_index'].shift(1)

    df['zigzag_last_low_index'] = latest_lows_index
    df['zigzag_last_low_index'] = df['zigzag_last_low_index'].ffill()
    df['zigzag_last_low_index_confirmed'] = df['zigzag_last_low_index'].shift(1)

    return df

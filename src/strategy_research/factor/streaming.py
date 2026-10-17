"""
流式（逐根 Bar 喂入）版技术指标，口径逐值对齐研究版 pandas 实现，
供 vnpy 框架策略在 on_bar 中增量更新（live / 回测同一套实现）。
"""
from collections import deque
from datetime import date, datetime, time

import numpy as np
from vnpy.trader.object import BarData


class RangeBreakoutStat:
    """通道压缩类日线指标的流式增量版。

    对齐 trix_vpt_research_v33 `_compute_daily_indicator` 的通道指标（未 shift 口径，
    即当日日线收线后的"信号层"值；执行层照旧由调用方在次日开盘前取用）：

    - channel_high / channel_low = rolling(channel_lookback).max/min
    - channel_width              = channel_high - channel_low
    - atr_channel                = TR 的 ewm(alpha=1/channel_lookback, adjust=False, min_periods=channel_lookback)
    - channel_compression        = channel_width / atr_channel
    - channel_compression_top_n  = compression 的 rolling(compression_lookback).quantile(compression_quantile)
    - is_channel_compression     = channel_compression <= channel_compression_top_n
    - channel_cum_ret            = close 的 pct_change(channel_lookback)

    语义约定（与 pandas 对齐）：
    - update_bar 只喂合成完毕的日线 Bar（trading_date 口径，夜盘归属次日）
    - 任一指标未过预热期时为 None（对应研究版 NaN）
    - is_channel_compression: bool | None；research 版中预热期该列为 False（NaN 比较），
      对策略条件的真假语义一致（falsy）
    - context_ready 等价于研究版 strict 列中关于该指标的 dropna 就绪标志
    """

    def __init__(self,
                 channel_lookback: int = 20,
                 compression_lookback: int = 60,
                 compression_quantile: float = 0.3):
        self.channel_lookback = channel_lookback
        self.compression_lookback = compression_lookback
        self.compression_quantile = compression_quantile

        self.bar_count = 0

        self.tr: float | None = None
        self.atr_channel: float | None = None
        self.channel_high: float | None = None
        self.channel_low: float | None = None
        self.channel_width: float | None = None
        self.channel_compression: float | None = None
        self.channel_compression_top_n: float | None = None
        self.is_channel_compression: bool | None = None
        self.channel_cum_ret: float | None = None

        self._prev_close: float | None = None
        self._tr_ema: float | None = None
        self._tr_count = 0
        self._high_window: deque[float] = deque(maxlen=channel_lookback)
        self._low_window: deque[float] = deque(maxlen=channel_lookback)
        self._close_window: deque[float] = deque(maxlen=channel_lookback + 1)
        self._compression_window: deque[float] = deque(maxlen=compression_lookback)

    @property
    def context_ready(self) -> bool:
        return self.is_channel_compression is not None

    def update_bar(self, bar: BarData):
        high = float(bar.high_price)
        low = float(bar.low_price)
        close = float(bar.close_price)

        # TR：Max(H-L, |H-前收|, |L-前收|)；首根无前收 → NaN(None)
        if self._prev_close is None:
            self.tr = None
        else:
            self.tr = max(high - low,
                          abs(high - self._prev_close),
                          abs(low - self._prev_close))
        self._prev_close = close

        if self.tr is not None:
            if self._tr_ema is None:
                self._tr_ema = self.tr
            else:
                alpha = 1.0 / self.channel_lookback
                self._tr_ema = (1.0 - alpha) * self._tr_ema + alpha * self.tr
            self._tr_count += 1
            if self._tr_count >= self.channel_lookback:
                self.atr_channel = self._tr_ema

        self._high_window.append(high)
        self._low_window.append(low)
        if len(self._high_window) == self.channel_lookback:
            self.channel_high = max(self._high_window)
            self.channel_low = min(self._low_window)
            self.channel_width = self.channel_high - self.channel_low

        if self.channel_high is not None and self.atr_channel is not None:
            self.channel_compression = self.channel_width / self.atr_channel
            self._compression_window.append(self.channel_compression)
            if len(self._compression_window) == self.compression_lookback:
                self.channel_compression_top_n = float(
                    np.quantile(list(self._compression_window),
                                self.compression_quantile,
                                method="linear")
                )

        if self.channel_compression_top_n is not None:
            self.is_channel_compression = bool(
                self.channel_compression <= self.channel_compression_top_n
            )

        self._close_window.append(close)
        if len(self._close_window) == self.channel_lookback + 1:
            self.channel_cum_ret = close / self._close_window[0] - 1

        self.bar_count += 1


class EmaPackCrossStat:
    """均线束三线全穿 EMA40 首次穿越信号机（流式增量版）。

    对齐 trix_vpt_research_v41 的 `EmaPackCrossStateMachine`（默认参数口径：
    min_regime_days=30、max_mixed_gap_days=None，即 v41 实际使用的配置）。

    - update_bar 喂合成完毕的日线 Bar：内部 O(1) 维护 EMA5/10/20/40
      （close 的 ewm(span=n, adjust=False)，与 v41 研究口径逐值等价，首根 Bar 即产值），
      随后用当日 EMA 值推进一次状态机。
    - breakup_today / breakdown_today：该这根日线 Bar 是否触发首次全上/全下 EMA40。
      口径映射：v41 在 T 日盘中读的是"T-1 日收盘后推进"的 flags；流式版在 T-1 日
      Bar 收线推进后 flags 即生效、T 日盘中稳定可读（下次日线收线才变化），语义一致。
    - 状态机纪元从首根喂入的 Bar 开始（计数器从零起算）。与 v41"从执行窗口起点开始
      推进"相比，仅首个片段内（预热历史残留的 run/mixed 计数）可能存在计数差异，
      且首个对侧穿越发生后各计数器清零、两纪元状态永久收敛；FG 全历史实证：
      计数差异 19 天但 breakup/breakdown 信号逐日完全一致。
      稳妥用法：启动时喂入尽可能长的预热历史；实盘首个片段的信号如需与回测严格
      对齐，可在首个对侧穿越发生前人工复核。
    - 状态字段名与 EmaPackCrossStateMachine 一一对应，便于对照。
    """

    EMA_SPANS = (5, 10, 20, 40)

    def __init__(self,
                 min_regime_days: int = 30,
                 max_mixed_gap_days: int | None = None):
        self.min_regime_days = min_regime_days
        self.max_mixed_gap_days = max_mixed_gap_days

        self.bar_count = 0

        self.ema5: float | None = None
        self.ema10: float | None = None
        self.ema20: float | None = None
        self.ema40: float | None = None

        self.pack_above_run_days = 0
        self.pack_below_run_days = 0
        self.pack_max_above_run_days = 0
        self.pack_max_below_run_days = 0
        self.pack_mixed_run_days = 0
        self.pack_prev_all_above = False
        self.pack_prev_all_below = False

        self.pack_breakdown_today = False
        self.pack_breakup_today = False

        self.ema_long_bull_days = 0
        self.ema_long_bear_days = 0

        self._ema_state: dict[int, float | None] = {span: None for span in self.EMA_SPANS}

    @property
    def context_ready(self) -> bool:
        """数据可用性标志（不涉及信号状态）：EMA5/10/20/40 均已可产值。

        与其他 stat 类的惯例一致（关键输出非 None 即 ready）。本类 EMA 以首根
        日线 Bar 的收盘为种子，与研究端 ewm(span=n, adjust=False) 无 min_periods
        的口径一致，故从第一根日线 Bar 起即为 ready。
        注：状态机纪元从喂入起点起算——"纪元与回测历史是否对齐"属于信号可信度
        问题，由策略侧控制喂入起点（补足预热历史）解决，不在此表达。"""
        return self._ema_state[self.EMA_SPANS[-1]] is not None

    @property
    def breakdown_today(self) -> bool:
        return self.pack_breakdown_today

    @property
    def breakup_today(self) -> bool:
        return self.pack_breakup_today

    def _update_ema(self, span: int, value: float) -> float:
        ema = self._ema_state[span]
        if ema is None:
            ema = value
        else:
            alpha = 2.0 / (span + 1.0)
            ema = (1.0 - alpha) * ema + alpha * value
        self._ema_state[span] = ema
        return ema

    def update_bar(self, bar: BarData):
        close = float(bar.close_price)
        ema5 = self.ema5 = self._update_ema(5, close)
        ema10 = self.ema10 = self._update_ema(10, close)
        ema20 = self.ema20 = self._update_ema(20, close)
        ema40 = self.ema40 = self._update_ema(40, close)

        pack_all_above = (ema5 > ema40
                          and ema10 > ema40
                          and ema20 > ema40)
        pack_all_below = (ema5 < ema40
                          and ema10 < ema40
                          and ema20 < ema40)
        pack_mixed = (not pack_all_above) and (not pack_all_below)

        self.pack_breakdown_today = False
        self.pack_breakup_today = False

        pack_mixed_gap_ok = (self.max_mixed_gap_days is None
                             or self.pack_mixed_run_days <= self.max_mixed_gap_days)

        if pack_all_below:
            self.pack_breakdown_today = bool(not self.pack_prev_all_below
                                             and pack_mixed_gap_ok
                                             and self.pack_max_above_run_days >= self.min_regime_days)
            self.pack_above_run_days = 0
            self.pack_max_above_run_days = 0
        else:
            self.pack_above_run_days = self.pack_above_run_days + 1 if pack_all_above else 0
            self.pack_max_above_run_days = max(self.pack_max_above_run_days, self.pack_above_run_days)

        if pack_all_above:
            self.pack_breakup_today = bool(not self.pack_prev_all_above
                                           and pack_mixed_gap_ok
                                           and self.pack_max_below_run_days >= self.min_regime_days)
            self.pack_below_run_days = 0
            self.pack_max_below_run_days = 0
        else:
            self.pack_below_run_days = self.pack_below_run_days + 1 if pack_all_below else 0
            self.pack_max_below_run_days = max(self.pack_max_below_run_days, self.pack_below_run_days)

        if pack_mixed:
            self.pack_mixed_run_days += 1
        else:
            self.pack_mixed_run_days = 0

        self.pack_prev_all_above = pack_all_above
        self.pack_prev_all_below = pack_all_below

        # 通道开仓路径的多头/空头排列天数（v33/v41 日线指标口径，非状态机本体）
        ema_long_bull = bool(ema5 > ema10 and ema10 > ema20)
        ema_long_bear = bool(ema5 < ema10 and ema10 < ema20)
        self.ema_long_bull_days = self.ema_long_bull_days + 1 if ema_long_bull else 0
        self.ema_long_bear_days = self.ema_long_bear_days + 1 if ema_long_bear else 0

        self.bar_count += 1


class AtrStat:
    """TR / ATR 的流式增量版。

    对齐 v33/v41 日线指标：tr = max(H-L, |H-前收|, |L-前收|)（首日无前收为 None）；
    atr = tr 的 ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period)
    （atr_period 同一个数既是 alpha 母数也是 min_periods，研究口径）。
    预热期内 atr 为 None；tr/atr 对应研究版 atr_14（atr 字段）列。
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self.bar_count = 0

        self.tr: float | None = None
        self.atr: float | None = None

        self._prev_close: float | None = None
        self._tr_ema: float | None = None
        self._tr_count = 0

    @property
    def context_ready(self) -> bool:
        return self.atr is not None

    def update_bar(self, bar: BarData):
        high = float(bar.high_price)
        low = float(bar.low_price)
        close = float(bar.close_price)

        if self._prev_close is None:
            self.tr = None
        else:
            self.tr = max(high - low,
                          abs(high - self._prev_close),
                          abs(low - self._prev_close))
        self._prev_close = close

        if self.tr is not None:
            if self._tr_ema is None:
                self._tr_ema = self.tr
            else:
                alpha = 1.0 / self.atr_period
                self._tr_ema = (1.0 - alpha) * self._tr_ema + alpha * self.tr
            self._tr_count += 1
            if self._tr_count >= self.atr_period:
                self.atr = self._tr_ema

        self.bar_count += 1


class TrueGapStat:
    """日线真实缺口（全天未回补）的流式增量版。

    对齐 v33/v41：
    - is_true_gap_up / is_true_gap_down：low > 昨日 high / high < 昨日 low
      （首日无前值，研究版 NaN 比较为 False，这里保持 bool False，语义一致）
    - true_gap_up_pct / true_gap_down_pct = 缺口点数 / 昨日 close
      （首日无前值为 None，无缺口为 0.0）
    平仓链只使用 pct 列（研究版经 shift(1) 后的 daily_true_gap_*_pct）。
    """

    def __init__(self):
        self.bar_count = 0

        self.is_true_gap_up: bool = False
        self.is_true_gap_down: bool = False
        self.true_gap_up_pct: float | None = None
        self.true_gap_down_pct: float | None = None

        self._prev_close: float | None = None
        self._prev_high: float | None = None
        self._prev_low: float | None = None

    @property
    def context_ready(self) -> bool:
        return self.true_gap_up_pct is not None

    def update_bar(self, bar: BarData):
        high = float(bar.high_price)
        low = float(bar.low_price)
        close = float(bar.close_price)

        if self._prev_close is None:
            self.is_true_gap_up = False
            self.is_true_gap_down = False
            self.true_gap_up_pct = None
            self.true_gap_down_pct = None
        else:
            self.is_true_gap_up = bool(low > self._prev_high)
            self.is_true_gap_down = bool(high < self._prev_low)
            self.true_gap_up_pct = (low - self._prev_high) / self._prev_close \
                if self.is_true_gap_up else 0.0
            self.true_gap_down_pct = (self._prev_low - high) / self._prev_close \
                if self.is_true_gap_down else 0.0

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low
        self.bar_count += 1


class KLineStat:
    """日线 K 线形态 + 巨K判定的流式增量版。

    对齐 v33/v41（平仓链用到的子集）：
    - kline_direction = sign(close - open) ∈ {-1, 0, 1}
    - kline_body_ratio = |close - open| / (high - low)（上下影线比未使用，未实现）
    - kline_range_top_n = 振幅的 rolling(range_lookback).quantile(range_quantile)
    - is_kline_long = (振幅 >= top_n) & (振幅 > 0)
    top_n 未过预热期（默认 60 根）时 is_kline_long 为 None
    （研究版该期为 False，falsy 语义一致）。
    """

    def __init__(self,
                 range_lookback: int = 60,
                 range_quantile: float = 0.9):
        self.range_lookback = range_lookback
        self.range_quantile = range_quantile

        self.bar_count = 0

        self.kline_direction: int = 0
        self.kline_body_ratio: float | None = None
        self.kline_range_top_n: float | None = None
        self.is_kline_long: bool | None = None

        self._range_window: deque[float] = deque(maxlen=range_lookback)

    @property
    def context_ready(self) -> bool:
        return self.is_kline_long is not None

    def update_bar(self, bar: BarData):
        open_price = float(bar.open_price)
        high = float(bar.high_price)
        low = float(bar.low_price)
        close = float(bar.close_price)

        body = close - open_price
        abs_body = abs(body)
        kline_range = high - low

        self.kline_direction = 1 if body > 0 else (-1 if body < 0 else 0)

        # 对齐研究版 abs_body / range（0/0 → NaN，x/0 → inf）
        if kline_range > 0:
            self.kline_body_ratio = abs_body / kline_range
        else:
            self.kline_body_ratio = float('nan') if abs_body == 0 else float('inf')

        self._range_window.append(kline_range)
        if len(self._range_window) == self.range_lookback:
            self.kline_range_top_n = float(
                np.quantile(list(self._range_window),
                            self.range_quantile,
                            method="linear")
            )
            self.is_kline_long = bool(kline_range >= self.kline_range_top_n
                                      and kline_range > 0)
        else:
            self.kline_range_top_n = None
            self.is_kline_long = None

        self.bar_count += 1


class TrixStat:
    """TRIX 及其平仓条件的流式增量版。

    对齐 factor/momentum.py calc_trix 在 v33/v41 的默认口径：
    - 三级 EMA：close 的 ewm(span=trix_period, adjust=False, min_periods=trix_period)，
      alpha = 2/(period+1)；前级未过 min_periods 时对后级按 NaN 跳过（不推进计数）
    - trix = 100 * (ema3 / ema3.shift(1) - 1)
    - trix_signal = trix 的 rolling(trix_signal_period).mean()
    - trix_hist = trix - trix_signal（prev_hist 即研究版 prev_trix_hist，内部维护）
    - trix_allow_close_long = (hist < prev_hist) & (hist < 0)，short 反之
    - trix_pos_days / neg_days = hist 连续为正/负天数（研究版 groupby+cumsum 等价；hist 未就绪计 0）
    hist 未就绪时 allow_close_* 为 None（研究版 False，falsy 语义一致）。
    """

    def __init__(self,
                 trix_period: int = 12,
                 trix_signal_period: int = 9):
        self.trix_period = trix_period
        self.trix_signal_period = trix_signal_period

        self.bar_count = 0

        self.trix: float | None = None
        self.trix_signal: float | None = None
        self.trix_hist: float | None = None
        self.trix_allow_close_long: bool | None = None
        self.trix_allow_close_short: bool | None = None
        self.trix_pos_days = 0
        self.trix_neg_days = 0

        self._ema_values: list[float | None] = [None, None, None]
        self._ema_counts = [0, 0, 0]
        self._signal_window: deque[float] = deque(maxlen=trix_signal_period)
        self._prev_hist: float | None = None
        self._prev_ema3: float | None = None

    @property
    def context_ready(self) -> bool:
        return self.trix_hist is not None

    def update_bar(self, bar: BarData):
        close = float(bar.close_price)
        alpha = 2.0 / (self.trix_period + 1.0)

        cur = close
        for idx in range(3):
            ema = self._ema_values[idx]
            if ema is None:
                self._ema_values[idx] = cur
            else:
                self._ema_values[idx] = (1.0 - alpha) * ema + alpha * cur
            self._ema_counts[idx] += 1
            if self._ema_counts[idx] < self.trix_period:
                break
            cur = self._ema_values[idx]

        if self._ema_counts[2] > self.trix_period:
            self.trix = 100.0 * (self._ema_values[2] / self._prev_ema3 - 1)

            self._signal_window.append(self.trix)
            if len(self._signal_window) == self.trix_signal_period:
                self.trix_signal = sum(self._signal_window) / self.trix_signal_period
                cur_hist = self.trix - self.trix_signal
                first_hist_day = self._prev_hist is None
                if first_hist_day:
                    # 对齐研究版：hist 首个有效日的 prev_hist 为 NaN，比较结果为 False
                    self.trix_allow_close_long = False
                    self.trix_allow_close_short = False
                else:
                    self.trix_allow_close_long = bool(cur_hist < self._prev_hist
                                                      and cur_hist < 0)
                    self.trix_allow_close_short = bool(cur_hist > self._prev_hist
                                                       and cur_hist > 0)
                self.trix_pos_days = self.trix_pos_days + 1 if cur_hist > 0 else 0
                self.trix_neg_days = self.trix_neg_days + 1 if cur_hist < 0 else 0
                self._prev_hist = cur_hist
                self.trix_hist = cur_hist

        self._prev_ema3 = self._ema_values[2]
        self.bar_count += 1


class ExtremeWindowStat:
    """日线极值窗口（60 日最高/最低）的流式增量版。

    对齐 v33/v41/v52 日线指标：
    - extreme_window_high = high 的 rolling(extreme_lookback).max()（含当日）
    - extreme_window_low  = low  的 rolling(extreme_lookback).min()（含当日）
    预热期内为 None（研究版 NaN → strict dropna）。

    执行层映射（与 RangeBreakoutStat 同一约定，类输出未 shift 的信号层值）：
    v41/v52 盘中消费的是 shift(1) 后的"前一交易日口径"（在 T 日盘中判断
    cur_can_open_long:  close < 极值低 * (1 + max_extreme_distance_pct)
    cur_can_open_short: close > 极值高 * (1 - max_extreme_distance_pct)）。
    流式时序：T 日日线收线喂入本类 → T+1 盘中读到的是 T 日收线后的窗口值，
    与研究版 shift(1) 语义一致，调用方无需再 shift。
    """

    def __init__(self, extreme_lookback: int = 60):
        self.extreme_lookback = extreme_lookback
        self.bar_count = 0

        self.extreme_window_high: float | None = None
        self.extreme_window_low: float | None = None

        self._high_window: deque[float] = deque(maxlen=extreme_lookback)
        self._low_window: deque[float] = deque(maxlen=extreme_lookback)

    @property
    def context_ready(self) -> bool:
        return self.extreme_window_high is not None

    def update_bar(self, bar: BarData):
        self._high_window.append(float(bar.high_price))
        self._low_window.append(float(bar.low_price))

        if len(self._high_window) == self.extreme_lookback:
            self.extreme_window_high = max(self._high_window)
            self.extreme_window_low = min(self._low_window)

        self.bar_count += 1


class IntradayStat:
    """执行层日内统计：一个逻辑交易日的分钟 K 线增量统计版。

    生命周期 = 一个逻辑交易日，由策略管理：每个新交易日 new 一个实例，构造时
    注入上一交易日的日 K BarData（重采样口径的合成日 Bar），随后用 update_bar
    逐根喂入当日的分钟 K，直到该交易日结束。实例不持有跨日状态、不知道未来的
    交易日，也不关心时段配置——它只负责"本交易日内"的增量统计。

    统计指标（v41/v52 盘中实际消费的最小集）：
    1. intraday_bar_index           本日内第几根分钟 K（自实例创建起的喂入计数，1 起）
    2. intraday_return_daily_open   当根收盘 / 当日开盘 - 1
                                    （当日开盘 = 首根喂入 Bar 的 open，即日 K
                                    重采样口径的 daily_open）
    3. intraday_open_gap_up_pct /
       intraday_open_gap_down_pct   开盘跳空幅度，相对上一交易日日 K：
                                    up   = (day_open - prev_daily_high) / prev_daily_close
                                    down = (prev_daily_low - day_open) / prev_daily_close
                                    未跳空时为 0.0；首根 Bar 确立开盘后全天恒定。
                                    （研究版另算的 is/point/covered 系列 v41/v52
                                    未消费，不实现）

    首个交易日没有上一交易日：prev_daily_bar=None 时 gap 特征恒 None（falsy，
    与研究版首日行由 strict dropna 丢弃的行为一致）。

    context_ready = 首根 Bar 已喂入（当日开盘确立后三组特征全部可读）——
    只表达数据可用性，不涉及信号状态。
    """

    def __init__(self, prev_daily_bar: BarData | None = None):
        # 上一交易日日 K 基价（重采样口径），跳空特征的比较基准
        self.prev_daily_high = float(prev_daily_bar.high_price) if prev_daily_bar is not None else None
        self.prev_daily_low = float(prev_daily_bar.low_price) if prev_daily_bar is not None else None
        self.prev_daily_close = float(prev_daily_bar.close_price) if prev_daily_bar is not None else None

        self.intraday_bar_index: int = 0

        self.intraday_return_daily_open: float | None = None
        self.intraday_open_gap_up_pct: float | None = None
        self.intraday_open_gap_down_pct: float | None = None

        self._day_open: float | None = None

    @property
    def context_ready(self) -> bool:
        return self._day_open is not None

    def update_bar(self, bar: BarData) -> None:
        self.intraday_bar_index += 1

        if self._day_open is None:
            self._day_open = float(bar.open_price)
            if self.prev_daily_close is not None:
                up_point = self._day_open - self.prev_daily_high
                down_point = self.prev_daily_low - self._day_open
                self.intraday_open_gap_up_pct = float(up_point / self.prev_daily_close) if up_point > 0 else 0.0
                self.intraday_open_gap_down_pct = float(down_point / self.prev_daily_close) if down_point > 0 else 0.0

        self.intraday_return_daily_open = float(bar.close_price) / self._day_open - 1.0

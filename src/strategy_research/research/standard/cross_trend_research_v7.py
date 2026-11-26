"""cross_trend_research_v7 = v4 + 持仓期反转搏击反手（部署候选）。

规则：通道开仓与反转搏击开仓同优先级；持仓期间任何平仓条件即平仓；
持仓期间遇通道反向开仓条件或反转搏击开仓条件，均平仓后反向开仓。
反转搏击口径 = v4（贴近极值 + 破 7 日通道 + 量仓门；反手止损 = 结构位小止损）。
血缘：复制自 cross_trend_research_v4（= 原 v90，v85 流式复刻正典）。
"""
import re
from time import perf_counter
import traceback
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from strategy_research.config.backtesting import G_BACKTEST_PRODUCT_LIST
from strategy_research.config.figure import COLOR_BAR_UP, COLOR_BAR_DOWN, COLOR_WHITE, COLOR_YELLOW
from strategy_research.backtesting.object import BacktestingDirection, BacktestingOverallStatistics, \
    BacktestingAllowTradeDirection
from strategy_research.backtesting.pandas_backtesting_base_v2 import PandasBacktestingBaseV2
from strategy_research.tool.data import BarDataManager


from strategy_research.factor.cross_trend_streaming import (
    DailyStatManager, RangeBreakoutStat, EmaPackCrossStat, AtrStat, TrueGapStat,
    KLineStat, TrixStat, ExtremeWindowStat, IntradayStat, SarStat, CclStat,
    ShortChannelStat, VolumeOITrendGateStat)
from vnpy.trader.constant import Exchange
from vnpy.trader.object import BarData


class TrixVptResearch(PandasBacktestingBaseV2):
    def __init__(self,
                 factor_name: str,
                 version: str,
                 symbol: str,
                 bar_period: str,
                 report_dir: str,
                 initial_capital: float,
                 bar_df: DataFrame,
                 enable_fig_daily_mode: bool,
                 enable_debug_mode: bool,
                 enable_jq_index_mode: bool,
                 allow_trade_direction: BacktestingAllowTradeDirection,
                 atr_period: int = 14,
                 volume_period: int = 14,
                 volume_threshold: float = 1.5,
                 stop_loss_multiplier: float = 2.0,
                 take_profit_multiplier: float = 3.0,
                 min_required_holding_days: int = 7,
                 channel_lookback: int = 20,
                 trix_period: int = 12,
                 trix_ma_period: int = 9,
                 trix_z_score_threshold: float = 0.5,
                 vpt_ma_period: int = 14,
                 vpt_z_score_threshold: float = 1.5,
                 ema_pack_min_regime_days: int = 30,
                 ema_pack_max_mixed_gap_days: int | None = None,
                 extreme_lookback: int = 60,
                 max_extreme_distance_pct: float = 0.5,
                 stop_loss_ticks: int = 1,
                 reversal_retrace_atr_mult: float = 3.0,
                 reversal_prev_vol_gate: bool = True,
                 reversal_prev_oi_gate: bool = True,
                 reversal_prev_vol_ratio_max: float = 1.0,
                 reversal_prev_oi_ratio_max: float = 1.0,
                 reversal_prev_window: int = 3,
                 reversal_prev_base_window: int = 14,
                 sar_af_start: float = 0.02,
                 sar_af_step: float = 0.02,
                 sar_af_max: float = 0.2, ):
        # v90 = v85 的流式指标复刻：交易语义目标与 v85 信号级等价，指标计算
        # 由向量化切换为流式（供体 = trowel 认证实现）。保留
        # use_t1_major=True + filter_auction_bars=True（正确口径）。
        super().__init__(factor_name,
                         version,
                         symbol,
                         bar_period,
                         report_dir,
                         initial_capital,
                         bar_df,
                         enable_fig_daily_mode,
                         enable_debug_mode,
                         enable_jq_index_mode,
                         allow_trade_direction)
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.min_required_holding_days = min_required_holding_days
        self.channel_lookback = channel_lookback

        self.trix_period = trix_period
        self.trix_ma_period = trix_ma_period
        self.trix_z_score_threshold = trix_z_score_threshold

        self.vpt_ma_period = vpt_ma_period
        self.vpt_z_score_threshold = vpt_z_score_threshold

        self.ema_pack_min_regime_days = ema_pack_min_regime_days
        self.ema_pack_max_mixed_gap_days = ema_pack_max_mixed_gap_days

        self.extreme_lookback = extreme_lookback
        self.max_extreme_distance_pct = max_extreme_distance_pct

        # 本文件为 v76-b「缩量+减仓门」的正典化版本（v81）+ T-1 主力口径（v83）
        # + 集合竞价脏 bar 过滤（v84）+ 量仓门喂入起点解耦（v85）——
        # v85 修复：量仓门 deque 原随 exec_df（strict dropna 后）才开始喂入，
        # 导致门在 exec_df 起步后的前 ~15 个交易日处于 fail-open，
        # 与 trowel（门自 cache 起点即开始喂入）产生不对称 → 信号数量不一致。
        # v85 将门的逐日喂入数据改从完整日线序列（dropna 之前）提取并预填 deque，
        # 门的可见历史起点与 trowel 对齐。v81/v83/v84 保留为历史口径档案。
        # 自 v81 起约定：不再使用 a/b 等参数后缀，默认参数不同的实验一律新开版本文件。
        #
        # 反转搏击单的确认与止损（v73 起源，v81 定稿口径）：
        # - 触发：贴近极值的首次结构破位（SAR 趋势 + 距 60 日极值回撤 ≤ 10% +
        #   close 击穿/突破 7 日高低点），不等三线穿越 EMA40 的滞后确认；
        # - 实证：v56 均线束单等待穿越后入场，亏损单入场时已从极值走了 20.7%（中位），
        #   前 5 日内有天量的反转单 5 败 1 胜（V 型顶巨量在信号之前发生，穿越时已错过）；
        # - 止损：近期 7 日高低点 ± N tick（结构位小止损）
        # - 方向一致性门：信号日 ccl_type 极性与注单方向须一致（收盘价方向confirm）；
        #   盈亏分解：盈利单 8/9 方向一致，亏损单 12 笔中 5 笔方向相反（信号日反向 = '
        #   回踩噪声首穿'）；放量门经验证为负贡献（放量反转单更易亏损），弃用
        # - 反转搏击单用结构位小止损：近期 7 日高低点 ± N tick（打结构位 = 反转失败即离场），
        #   通道压缩路径仍保持 v56 的入场价 ± 2ATR 止损
        self.stop_loss_ticks = stop_loss_ticks
        # 贴近极值锚的 ATR 归一化阈值：距 60 日极值 ≤ K × ATR
        # （固定 10% 在低波动品种上等价距离过远，导致 ag/sc 反复触发；
        #   ATR 归一化后自动按品种波动收紧/放宽，防御性统一规则）
        self.reversal_retrace_atr_mult = reversal_retrace_atr_mult

        # 前 N 交易日量仓综合门（v81 正典：62 个反转搏击事件的描述性分析 + 全品种回测双确认，
        #   默认启用 vol_gate=True + oi_gate=True，即「缩量+减仓」最优象限门；T-1 口径无未来数据）：
        #   近 window 日均量 / 近 base 日均量 = vol_ratio；持仓同构 oi_ratio。
        #   组合胜率：缩量+减仓 43.5% > 基准 33.9% > 放量系 25~28.6%。
        # - vol_gate：要求 vol_ratio ≤ max（排除放量破位）
        # - oi_gate：要求 oi_ratio ≤ max（排除“缩量+增仓”逆势投机破位，
        #   与 vol_gate 组合得到最优组“缩量+减仓”）
        self.reversal_prev_vol_gate = reversal_prev_vol_gate
        self.reversal_prev_oi_gate = reversal_prev_oi_gate
        self.reversal_prev_vol_ratio_max = reversal_prev_vol_ratio_max
        self.reversal_prev_oi_ratio_max = reversal_prev_oi_ratio_max
        self.reversal_prev_window = reversal_prev_window
        self.reversal_prev_base_window = reversal_prev_base_window

        self.sar_af_start = sar_af_start
        self.sar_af_step = sar_af_step
        self.sar_af_max = sar_af_max

        self._init_streaming()

        # 交易所枚举（BarData 构造用；配置可能给字符串或枚举）
        self._exchange_enum = (self.contract_config.exchange
                               if isinstance(self.contract_config.exchange, Exchange)
                               else Exchange(self.contract_config.exchange))

    def _init_streaming(self) -> None:
        """注册流式统计集：日级更新表 / 就绪判定表 / 绘图列映射。

        v85 向量化管线的 60/0.3（压缩分位）与 60/0.9（K 线幅度分位）
        在 v85 内为字面量，此处按同值传入。
        """
        manager = DailyStatManager()
        atr = AtrStat(self.atr_period)
        channel = RangeBreakoutStat(self.channel_lookback, 60, 0.3)
        ema = EmaPackCrossStat(self.ema_pack_min_regime_days, self.ema_pack_max_mixed_gap_days)
        trix = TrixStat(self.trix_period, self.trix_ma_period)
        ccl = CclStat()
        true_gap = TrueGapStat()
        kline = KLineStat(60, 0.9)
        extreme = ExtremeWindowStat(self.extreme_lookback)
        sar = SarStat(self.sar_af_start, self.sar_af_step, self.sar_af_max)
        short_ch = ShortChannelStat()
        gate = VolumeOITrendGateStat(self.reversal_prev_window, self.reversal_prev_base_window,
                                     self.reversal_prev_vol_ratio_max, self.reversal_prev_oi_ratio_max)

        # 保留与 trowel 策略同名的引用（交易逻辑可读）
        self.daily_stat_manager = manager
        self.daily_atr_stat = atr
        self.daily_channel_stat = channel
        self.daily_ema_pack_stat = ema
        self.daily_trix_stat = trix
        self.daily_ccl_stat = ccl
        self.daily_true_gap_stat = true_gap
        self.daily_kline_stat = kline
        self.daily_price_extreme_window_stat = extreme
        self.daily_sar_stat = sar
        self.daily_short_channel_stat = short_ch
        self.daily_volume_oi_gate_stat = gate
        self.intraday_stat: IntradayStat | None = None

        self.daily_stats = [atr, channel, ema, trix, ccl, true_gap, kline,
                            extreme, sar, short_ch, gate]
        # 就绪判定 = v85 strict 绑定列（60 日滚动族；通道压缩不参与绑定，
        # 其预热期 None 的 falsy 对应研究版 NaN 比较的 False 伪影）
        self.readiness_checks = [
            (atr, "atr"),
            (extreme, "extreme_window_high"),
            (kline, "kline_range_top_n"),
            (channel, "channel_high"),
            (short_ch, "short_channel_high"),
            (sar, "sar_trend"),
            (trix, "trix_hist"),
            (ema, "ema40"),
        ]
        # 绘图列映射：{列名: (统计对象, 属性)}——绘图 df = 交易同源状态
        self.plot_field_map = {
            "ema_5": (ema, "ema5"), "ema_10": (ema, "ema10"), "ema_20": (ema, "ema20"),
            "ema_40": (ema, "ema40"), "ema_60": (ema, "ema60"),
            "sar": (sar, "sar"), "sar_trend": (sar, "sar_trend"),
            "trix": (trix, "trix"), "trix_signal": (trix, "trix_signal"),
            "trix_hist": (trix, "trix_hist"),
            "ccl": (ccl, "ccl"), "ccl_type": (ccl, "ccl_type"),
        }

    @property

    def parameters(self):
        skip_field = [""]

        item_dict = {}

        return item_dict


    # region ====================== v90 流式管道 ======================

    def _compute_daily_indicator(self, bar_1d_df: DataFrame) -> tuple[DataFrame, DataFrame, list]:
        """v90 起已移除：指标全部流式化（交易与绘图同源）。

        原向量化实现见 v85（factor-report 口径的历史档案）。绘图列由
        _update_daily_indicator 的逐日记录（_plot_records）直接给出。
        """
        raise NotImplementedError("v90 已移除向量化日级指标计算；绘图用 _plot_records")
    def compute_indicators(self,
                           bar_1m_df: DataFrame,
                           session_time_type: list[tuple[time, time]]) -> tuple[DataFrame, DataFrame]:
        """v90 起不再使用：指标改为流式统计（run() 内逐 bar 喂入）。

        保留抽象实现以满足基类契约；绘图用的日级列由
        _compute_daily_indicator 在回测后基于 DailyStatManager 的日线重建。
        """
        raise NotImplementedError("v90 使用流式指标管线，不调用 compute_indicators")


    def _trading_logic(self, bar: BarData, cur_trading_date) -> None:
        """单根 bar 的交易逻辑（与 v85 循环体逐字一致，数据源 = 流式统计状态）。

        状态生命周期由 V2 基类承载：open_memory（单日一次建仓，组序号键）、
        after_holding_high/low（持仓后极值，开仓时由本逻辑重置为当根 bar）、
        trading_date_index（持仓天数计算）、_is_last_bar（末根强平标记）。
        """
        cur_position = self.account.get_position_v2()
        cur_datetime = bar.datetime
        cur_high = bar.high_price
        cur_low = bar.low_price

        cur_return_today_open = self.intraday_stat.intraday_return_daily_open
        cur_intraday_bar_index = self.intraday_stat.intraday_bar_index

        cur_is_channel_compression = self.daily_channel_stat.is_channel_compression
        cur_channel_high = self.daily_channel_stat.channel_high
        cur_channel_low = self.daily_channel_stat.channel_low
        cur_daily_atr_14 = self.daily_atr_stat.atr

        # 最新价更新
        cur_close = bar.close_price

        # 距极值入场过滤（前一日口径的 60 日高低点）：
        # 做多：距 60 日低点涨幅达到阈值 → 放弃追多；做空：距 60 日高点跌幅达到阈值 → 放弃追空
        cur_can_open_long = cur_close < (self.daily_price_extreme_window_stat.extreme_window_low * (1 + self.max_extreme_distance_pct))
        cur_can_open_short = cur_close > (self.daily_price_extreme_window_stat.extreme_window_high * (1 - self.max_extreme_distance_pct))
        if cur_position:
            position_bar_data = self._get_contract_bar(cur_position.symbol, cur_datetime)
            self.account.update_latest_price(cur_position.symbol,
                                             position_bar_data["close"])

        cur_major_contract_symbol, cur_major_contract_bar = self._get_major_contract_symbol_and_bar(cur_datetime)
        if (cur_major_contract_symbol is None
                or cur_major_contract_bar is None
        ):
            # print(f"cur_datetime: {cur_datetime} , "
            #       f"cur_major_contract_symbol: {cur_major_contract_symbol}, "
            #       f"cur_major_contract_bar: {cur_major_contract_bar}")
            return  # （v85 循环语境为 continue：跳过本 bar 剩余逻辑）

        channel_can_open_long_condition = (
                cur_is_channel_compression
                and cur_close > (cur_channel_high + 0.1 * cur_daily_atr_14)
                and cur_can_open_long
                and cur_intraday_bar_index >= 30
                and cur_return_today_open > 0
                and (self.daily_ema_pack_stat.ema5 > self.daily_ema_pack_stat.ema10 > self.daily_ema_pack_stat.ema20)
                and self.daily_ema_pack_stat.ema_long_bull_days >= 3
                and self.daily_sar_stat.sar_trend == 1
        )
        # 贴近极值的首次结构破位反转（不等三线穿越 EMA40 的滞后确认）：
        # - 距 60 日极值 ≤ K×ATR（ATR 归一化：低波动品种自动收紧，高波动品种自动放宽；
        #   实证：亏损单入场时已从极值走了 20.7%，盈利单 13.7%——晚入场即追单）
        # - close 击穿 7 日低点（做空）/ 突破 7 日高点（做多）
        # - SAR 趋势同向（快反转确认；实证盈利单 SAR 平均已翻 11.1 天）
        channel_can_open_short_condition = (
                cur_is_channel_compression
                and cur_close < (cur_channel_low - 0.1 * cur_daily_atr_14)
                and cur_can_open_short
                and cur_intraday_bar_index >= 30
                and cur_return_today_open < 0
                and (self.daily_ema_pack_stat.ema5 < self.daily_ema_pack_stat.ema10 < self.daily_ema_pack_stat.ema20)
                and self.daily_ema_pack_stat.ema_long_bear_days >= 3
                and self.daily_sar_stat.sar_trend == -1
        )
        retrace_distance_high = self.daily_price_extreme_window_stat.extreme_window_high - cur_close
        ema_pack_can_open_short_condition = (
                self.daily_sar_stat.sar_trend == -1
                and retrace_distance_high <= self.reversal_retrace_atr_mult * cur_daily_atr_14
                and cur_close < self.daily_short_channel_stat.short_channel_low
                and cur_can_open_short
                and cur_intraday_bar_index >= 30
                and self.daily_volume_oi_gate_stat.gates_pass()
        )

        retrace_distance_low = cur_close - self.daily_price_extreme_window_stat.extreme_window_low
        ema_pack_can_open_long_condition = (
                self.daily_sar_stat.sar_trend == 1
                and retrace_distance_low <= self.reversal_retrace_atr_mult * cur_daily_atr_14
                and cur_close > self.daily_short_channel_stat.short_channel_high
                and cur_can_open_long
                and cur_intraday_bar_index >= 30
                and self.daily_volume_oi_gate_stat.gates_pass()
        )

        if cur_position is None or cur_position.volume == 0:
            """开仓检查"""
            has_memory = self.open_memory.get(cur_trading_date, False)
            if has_memory:
                return  # （v85 循环语境为 continue：跳过本 bar 剩余逻辑）

            open_reason = ""
            if channel_can_open_long_condition or ema_pack_can_open_long_condition:
                """多头开仓"""
                if self.allow_trade_direction == BacktestingAllowTradeDirection.SHORT:
                    return  # （v85 循环语境为 continue：跳过本 bar 剩余逻辑）

                # 通道压缩路径：止损 max(通道低点, 入场价 - 2*ATR)（v56 原口径）；
                # 反转搏击路径：结构位小止损 = 近期低点 - N*tick（打穿结构位 = 反转失败即离场）
                if channel_can_open_long_condition:
                    stop_loss = max(cur_channel_low, cur_close - cur_daily_atr_14 * self.stop_loss_multiplier)
                    open_reason = "channel_condition"
                else:
                    stop_loss = self.daily_short_channel_stat.short_channel_low - self.stop_loss_ticks * self.price_tick
                    open_reason = "ema_condition"

                take_profit = cur_close + cur_daily_atr_14 * self.take_profit_multiplier

                open_contract_symbol, open_contract_close = self._get_next_major_contract_data(
                    cur_datetime
                )
                self.open_position(
                    open_contract_symbol,
                    open_contract_close,
                    BacktestingDirection.LONG,
                    open_reason,
                    cur_datetime,
                    cur_trading_date,
                    stop_loss,
                    take_profit,
                    0,
                    cur_close,
                )
                self.after_holding_high = cur_high
                self.after_holding_low = cur_low
                self.open_memory[cur_trading_date] = True

            elif channel_can_open_short_condition or ema_pack_can_open_short_condition:
                """空头开仓"""
                if self.allow_trade_direction == BacktestingAllowTradeDirection.LONG:
                    return  # （v85 循环语境为 continue：跳过本 bar 剩余逻辑）

                # 通道压缩路径：止损 min(通道高点, 入场价 + 2*ATR)（v56 原口径）；
                # 反转搏击路径：结构位小止损 = 近期高点 + N*tick
                if channel_can_open_short_condition:
                    stop_loss = min(cur_channel_high, cur_close + cur_daily_atr_14 * self.stop_loss_multiplier)
                    open_reason = "channel_condition"
                else:
                    stop_loss = self.daily_short_channel_stat.short_channel_high + self.stop_loss_ticks * self.price_tick
                    open_reason = "ema_condition"

                take_profit = cur_close - cur_daily_atr_14 * self.take_profit_multiplier

                open_contract_symbol, open_contract_close = self._get_next_major_contract_data(
                    cur_datetime
                )
                self.open_position(
                    open_contract_symbol,
                    open_contract_close,
                    BacktestingDirection.SHORT,
                    open_reason,
                    cur_datetime,
                    cur_trading_date,
                    stop_loss,
                    take_profit,
                    0,
                    cur_close,
                )
                self.after_holding_high = cur_high
                self.after_holding_low = cur_low
                self.open_memory[cur_trading_date] = True
        else:
            """平仓检查"""
            position_symbol = cur_position.symbol
            position_bar_data = self._get_contract_bar(position_symbol, cur_datetime)
            if self.after_holding_high is None or cur_high >= self.after_holding_high:
                self.after_holding_high = cur_high

            if self.after_holding_low is None or cur_low <= self.after_holding_low:
                self.after_holding_low = cur_low

            close_reason = None
            is_switch_contract = False
            is_reverse_open = False
            open_trading_date = cur_position.open_trading_date
            holding_trading_days = self.trading_date_index.get(cur_trading_date) - self.trading_date_index.get(
                open_trading_date)
            if cur_position.direction == BacktestingDirection.LONG:
                """多头平仓检查"""
                adjust_stop_loss = -1
                if cur_trading_date != cur_position.open_trading_date:
                    # self.after_holding_high 已由逐 bar 数值追踪维护（v90）
                    adjust_stop_loss = max(cur_position.stop_loss, self.after_holding_high * 0.8)

                if channel_can_open_short_condition or ema_pack_can_open_short_condition:
                    if channel_can_open_short_condition:
                        close_reason = '达到开空条件，平仓开空'
                    else:
                        close_reason = '达到反转搏击开空条件，平仓开空'
                    is_reverse_open = True
                elif cur_close <= cur_position.stop_loss:
                    close_reason = '止损'
                elif (adjust_stop_loss > 0
                      and cur_close < adjust_stop_loss):
                    close_reason = "动态止盈止损"
                elif self.intraday_stat.intraday_open_gap_down_pct > 0.01:
                    close_reason = "开盘跳空反转"
                elif (self.daily_true_gap_stat.true_gap_down_pct > 0.005
                      and cur_intraday_bar_index >= 30
                      and cur_return_today_open < 0
                ):
                    close_reason = "跳空反转"
                elif (self.daily_kline_stat.is_kline_long
                      and self.daily_kline_stat.kline_body_ratio >= 0.8
                      and self.daily_kline_stat.kline_direction == -1
                ):
                    close_reason = "前一根巨型阴线"
                elif (
                        self.daily_trix_stat.trix_allow_close_long
                        and holding_trading_days >= self.min_required_holding_days
                        and not cur_is_channel_compression
                        and self.daily_trix_stat.trix_neg_days >= 5
                ):
                    close_reason = 'trix 信号反转平仓'
                elif self._is_last_bar:
                    close_reason = '最后可交易日'
                elif self._need_switch_contract(cur_datetime, cur_position):
                    close_reason = '移仓换月'
                    is_switch_contract = True

                if close_reason:
                    self.close_position(
                        position_symbol,
                        position_bar_data['close'],
                        cur_datetime,
                        cur_trading_date,
                        close_reason,
                        0,
                        cur_close,
                    )
                    if is_switch_contract:
                        switch_contract_symbol, switch_contract_close = self._get_next_major_contract_data(
                            cur_datetime
                        )
                        self.open_position(
                            switch_contract_symbol,
                            switch_contract_close,
                            BacktestingDirection.LONG,
                            "移仓换月",
                            cur_datetime,
                            cur_trading_date,
                            cur_position.stop_loss,
                            cur_position.take_profit,
                            0,
                            cur_close
                        )

                    if is_reverse_open:
                        if self.allow_trade_direction == BacktestingAllowTradeDirection.LONG:
                            return  # （v85 循环语境为 continue：跳过本 bar 剩余逻辑）

                        if channel_can_open_short_condition:
                            stop_loss = min(cur_channel_high, cur_close + cur_daily_atr_14 * self.stop_loss_multiplier)
                            reverse_open_reason = "channel_condition"
                        else:
                            stop_loss = self.daily_short_channel_stat.short_channel_high + self.stop_loss_ticks * self.price_tick
                            reverse_open_reason = "ema_condition"
                        take_profit = cur_close - cur_daily_atr_14 * self.take_profit_multiplier
                        reverse_contract_symbol, reverse_contract_close = self._get_next_major_contract_data(
                            cur_datetime
                        )
                        self.open_position(
                            reverse_contract_symbol,
                            reverse_contract_close,
                            BacktestingDirection.SHORT,
                            reverse_open_reason,
                            cur_datetime,
                            cur_trading_date,
                            stop_loss,
                            take_profit,
                            0,
                            cur_close,
                        )
                        self.after_holding_high = cur_high
                        self.after_holding_low = cur_low
                        self.open_memory[cur_trading_date] = True

            elif cur_position.direction == BacktestingDirection.SHORT:
                """空仓平仓检查"""
                adjust_stop_loss = -1
                if cur_trading_date != cur_position.open_trading_date:
                    # self.after_holding_low 已由逐 bar 数值追踪维护（v90）
                    adjust_stop_loss = min(cur_position.stop_loss, self.after_holding_low * 1.2)

                if channel_can_open_long_condition or ema_pack_can_open_long_condition:
                    if channel_can_open_long_condition:
                        close_reason = '达到开多条件，平仓开多'
                    else:
                        close_reason = '达到反转搏击开多条件，平仓开多'
                    is_reverse_open = True
                elif cur_close >= cur_position.stop_loss:
                    close_reason = '止损'
                elif (adjust_stop_loss > 0
                      and cur_close > adjust_stop_loss):
                    close_reason = '动态止盈止损'
                elif self.intraday_stat.intraday_open_gap_up_pct > 0.01:
                    close_reason = "开盘跳空反转"
                elif (self.daily_true_gap_stat.true_gap_up_pct > 0.005
                      and cur_intraday_bar_index >= 30
                      and cur_return_today_open > 0
                ):
                    close_reason = "跳空反转"
                elif (self.daily_kline_stat.is_kline_long
                      and self.daily_kline_stat.kline_body_ratio >= 0.8
                      and self.daily_kline_stat.kline_direction == 1
                        # and cur_intraday_bar_index >= 5
                        # and cur_return_today_open > 0
                ):
                    close_reason = "前一根巨型阳线"
                # elif cur_close > cur_channel_high:
                #     close_reason = '价格高于通道高点'
                elif (
                        self.daily_trix_stat.trix_allow_close_short
                        and holding_trading_days >= self.min_required_holding_days
                        and not cur_is_channel_compression
                        and self.daily_trix_stat.trix_pos_days >= 5
                ):
                    close_reason = 'trix 信号反转平仓'
                # elif (daily_zigzag_structure_confirmed_arr[i] == "HH-HL"
                #       and cur_close > daily_zigzag_last_low_confirmed_arr[i]):
                #     close_reason = "HH-HL 上升结构"
                elif self._is_last_bar:
                    close_reason = '最后可交易日'
                elif self._need_switch_contract(cur_datetime, cur_position):
                    close_reason = '移仓换月'
                    is_switch_contract = True

                if close_reason:
                    self.close_position(
                        position_symbol,
                        position_bar_data['close'],
                        cur_datetime,
                        cur_trading_date,
                        close_reason,
                        0,
                        cur_close,
                    )

                    if is_switch_contract:
                        switch_contract_symbol, switch_contract_close = self._get_next_major_contract_data(
                            cur_datetime
                        )
                        self.open_position(
                            switch_contract_symbol,
                            switch_contract_close,
                            BacktestingDirection.SHORT,
                            '移仓换月',
                            cur_datetime,
                            cur_trading_date,
                            cur_position.stop_loss,
                            cur_position.take_profit,
                            0,
                            cur_close
                        )

                    if is_reverse_open:
                        if self.allow_trade_direction == BacktestingAllowTradeDirection.SHORT:
                            return  # （v85 循环语境为 continue：跳过本 bar 剩余逻辑）

                        if channel_can_open_long_condition:
                            stop_loss = max(cur_channel_low, cur_close - cur_daily_atr_14 * self.stop_loss_multiplier)
                            reverse_open_reason = "channel_condition"
                        else:
                            stop_loss = self.daily_short_channel_stat.short_channel_low - self.stop_loss_ticks * self.price_tick
                            reverse_open_reason = "ema_condition"
                        take_profit = cur_close + cur_daily_atr_14 * self.take_profit_multiplier
                        reverse_contract_symbol, reverse_contract_close = self._get_next_major_contract_data(
                            cur_datetime
                        )
                        self.open_position(
                            reverse_contract_symbol,
                            reverse_contract_close,
                            BacktestingDirection.LONG,
                            reverse_open_reason,
                            cur_datetime,
                            cur_trading_date,
                            stop_loss,
                            take_profit,
                            0,
                            cur_close,
                        )
                        self.after_holding_high = cur_high
                        self.after_holding_low = cur_low
                        self.open_memory[cur_trading_date] = True

    def create_kline_fig(self) -> go.Figure:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.15, 0.15, 0.15],
            vertical_spacing=0.05,
            subplot_titles=["K线", "VOL", "TRIX", "CCL"],
            specs=[
                [{}],
                [{"secondary_y": True}],
                [{}],
                [{}],
            ]
        )

        return fig

    def _plot_indicator_ema(self,
                            fig: go.Figure,
                            x_axis: pd.Series,
                            bar_df: DataFrame):
        # EMA5
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['ema_5'],
                mode='lines',
                name='EMA5',
                line=dict(color='white', width=1),
                hovertemplate='<b>%{x}</b><br>EMA5: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

        # EMA10
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['ema_10'],
                mode='lines',
                name='EMA10',
                line=dict(color='yellow', width=1),
                hovertemplate='<b>%{x}</b><br>EMA10: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

        # EMA20
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['ema_20'],
                mode='lines',
                name='EMA20',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x}</b><br>EMA20: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

        # EMA40
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['ema_40'],
                mode='lines',
                name='EMA40',
                line=dict(color='#FF69B4', width=1.8, dash='dot'),
                hovertemplate='<b>%{x}</b><br>EMA40: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

        # # EMA60
        # fig.add_trace(
        #     go.Scatter(
        #         x=x_axis,
        #         y=bar_df['ema_60'],
        #         mode='lines',
        #         name='EMA60',
        #         line=dict(color='#32CD32', width=2.0, dash='dash'),
        #         hovertemplate='<b>%{x}</b><br>EMA60: %{y:.4f}<extra></extra>'
        #     ),
        #     row=1,
        #     col=1)

    def _plot_indicator_sar(self,
                            fig: go.Figure,
                            x_axis: pd.Series,
                            bar_df: DataFrame):
        # SAR 点位叠加在 K 线子图：上涨趋势（SAR 在价格下方）红点，下跌趋势（SAR 在价格上方）绿点
        sar_up_mask = (bar_df['sar_trend'] == 1)
        sar_down_mask = (bar_df['sar_trend'] == -1)

        fig.add_trace(
            go.Scatter(
                x=x_axis[sar_up_mask],
                y=bar_df['sar'][sar_up_mask],
                mode='markers',
                name='SAR(涨)',
                marker=dict(color=COLOR_BAR_UP, size=4),
                hovertemplate='<b>%{x}</b><br>SAR: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=x_axis[sar_down_mask],
                y=bar_df['sar'][sar_down_mask],
                mode='markers',
                name='SAR(跌)',
                marker=dict(color=COLOR_BAR_DOWN, size=4),
                hovertemplate='<b>%{x}</b><br>SAR: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

    def _plot_indicator_trix(self,
                             fig: go.Figure,
                             x_axis: pd.Series,
                             bar_df: DataFrame):
        colors = np.where(bar_df['trix_hist'] > 0, COLOR_BAR_UP, COLOR_BAR_DOWN)
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=bar_df['trix_hist'],
                marker_color=colors,
                name='TRIX-HIST',
                hovertemplate='<b>%{x}</b><br>TRIX-HIST: %{y:.4f}<extra></extra>'
            ),
            row=3,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['trix'],
                name='TRIX',
                line=dict(color=COLOR_WHITE, width=1.2),
            ),
            row=3,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['trix_signal'],
                name='TRIX_SIGNAL',
                line=dict(color=COLOR_YELLOW, width=1.2),
            ),
            row=3,
            col=1)
        fig.update_yaxes(zeroline=True,
                         zerolinecolor='#666666',
                         zerolinewidth=0.5,
                         row=3,
                         col=1)

    def _plot_indicator_macd(self,
                             fig: go.Figure,
                             x_axis: pd.Series,
                             bar_df: DataFrame):
        colors = np.where(bar_df['macd_hist'] > 0, COLOR_BAR_UP, COLOR_BAR_DOWN)
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=bar_df['macd_hist'],
                marker_color=colors,
                name='MACD-HIST',
                hovertemplate='<b>%{x}</b><br>MACD-HIST: %{y:.4f}<extra></extra>'
            ),
            row=3,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['macd_diff'],
                name='DIF',
                line=dict(color=COLOR_WHITE, width=1.2),
            ),
            row=3,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=bar_df['macd_dea'],
                name='DEA',
                line=dict(color=COLOR_YELLOW, width=1.2),
            ),
            row=3,
            col=1)

        fig.update_yaxes(zeroline=True,
                         zerolinecolor='#666666',
                         zerolinewidth=0.5,
                         row=3,
                         col=1)

    def _plot_indicator_vol_oi(self,
                               fig: go.Figure,
                               x_axis: pd.Series,
                               bar_df: DataFrame):
        up = bar_df['close'] > bar_df['close'].shift(1)
        up.iloc[0] = True  # 第一根无前值，默认按红处理
        vol_colors = np.where(up, COLOR_BAR_UP, COLOR_BAR_DOWN)

        fig.add_trace(go.Bar(
            x=x_axis,
            y=bar_df['volume'],
            marker_color=vol_colors,
            name='成交量',
            opacity=0.6,  # 半透明，避免压住OI线
        ), row=2, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=bar_df['open_interest'],
            name='持仓量',
            line=dict(color=COLOR_YELLOW, width=1.8),
        ), row=2, col=1, secondary_y=True)

        fig.update_yaxes(
            # title_text='成交量（手）',
            tickformat=',.0f',
            zeroline=True, zerolinecolor='#666666', zerolinewidth=0.5,
            rangemode='tozero',
            row=2, col=1, secondary_y=False,
        )

        # 右轴（持仓量）：同样处理
        fig.update_yaxes(
            # title_text='持仓量（手）',
            tickformat=',.0f',
            tickfont=dict(color='#888888'),
            showgrid=False,
            row=2, col=1, secondary_y=True,
        )

    def _plot_indicator_ccl(self,
                            fig: go.Figure,
                            x_axis: pd.Series,
                            bar_df: DataFrame):
        ccl_style = {
            'LONG_OPEN': ('多头增仓', COLOR_BAR_UP),  # 红：主动买进
            'SHORT_CLOSE': ('空头平仓', COLOR_BAR_UP),  # 淡红：空头回补
            'SHORT_OPEN': ('空头增仓', COLOR_BAR_DOWN),  # 绿：主动卖出
            'LONG_CLOSE': ('多头平仓', COLOR_BAR_DOWN),  # 淡绿：多头离场
        }

        for ctype, (cname, color) in ccl_style.items():
            mask = (bar_df['ccl_type'] == ctype)
            fig.add_trace(go.Bar(
                x=x_axis[mask],
                y=bar_df['ccl'][mask],
                name=cname,
                marker_color=color,
                hovertemplate='%{x}<br>' + cname + ': %{y:+,.0f} 手<extra></extra>',
            ), row=4, col=1)

        y_max = bar_df['ccl'].abs().max() * 1.1
        fig.update_yaxes(
            # title_text='仓差（手）',
            tickformat=',.0f',  # 沿用你的千分位方案，避免出现 M 后缀
            zeroline=True, zerolinecolor='#666666', zerolinewidth=1,
            range=[-y_max, y_max],  # 对称范围，零轴始终居中
            row=4, col=1,
        )

    def plot_indicator(self,
                       fig: go.Figure,
                       bar_df: DataFrame, ) -> go.Figure:
        datetime_str = bar_df['trading_date'].dt.strftime(self.datetime_formater)
        # EMA
        self._plot_indicator_ema(fig,
                                 datetime_str,
                                 bar_df)

        # SAR
        self._plot_indicator_sar(fig,
                                 datetime_str,
                                 bar_df)

        # VOL
        self._plot_indicator_vol_oi(fig,
                                    datetime_str,
                                    bar_df)

        # TRIX
        self._plot_indicator_trix(fig,
                                  datetime_str,
                                  bar_df)

        # # MACD
        # self._plot_indicator_macd(fig,
        #                           datetime_str,
        #                           bar_df)

        # CCL
        self._plot_indicator_ccl(fig,
                                 datetime_str,
                                 bar_df)

        return fig




def test_jq(backtesting_allow_trade_direction: BacktestingAllowTradeDirection = BacktestingAllowTradeDirection.ALL):
    pd.set_option('display.max_columns', None)
    current_file = Path(__file__)
    filename = current_file.stem
    pattern = r'_(v\d+(?:\.\d+)*)'
    match = re.search(pattern, filename)
    if match:
        version = match.group(1)
    else:
        raise Exception(f"filename does not standardize {filename}")

    report_root_dir = "factor-report"
    factor_name = "cross_trend"
    bar_period = "1m"
    initial_capital = 100_0000
    backtest_product_list = G_BACKTEST_PRODUCT_LIST
    report_factor_dir = Path(f"{report_root_dir}/{factor_name}/{version}/{backtesting_allow_trade_direction.value}")
    report_factor_dir.mkdir(parents=True, exist_ok=True)
    backtesting_overall_stat = BacktestingOverallStatistics(
        factor_name,
        report_factor_dir,
        filename,
        version
    )
    test_start = perf_counter()
    product_timings: list[tuple[str, float]] = []
    for product in backtest_product_list:
        product_start = perf_counter()
        report_factor_product_dir = report_factor_dir / product
        report_factor_product_dir.mkdir(parents=True, exist_ok=True)
        symbol = f"{product}JQ00"

        print(f"----------{symbol}----------")
        try:
            jq_bar_1m_df = BarDataManager.load_symbol_from_cache(symbol, "1m")
            if jq_bar_1m_df.empty:
                print(f"{symbol}-1m dataframe is empty")
                continue

            trix_vpt_research = TrixVptResearch(factor_name,
                                                version,
                                                symbol,
                                                bar_period,
                                                str(report_factor_product_dir),
                                                initial_capital,
                                                jq_bar_1m_df,
                                                True,
                                                False,
                                                True,
                                                backtesting_allow_trade_direction, )
            trix_vpt_research.run()
            trix_vpt_research.show_backtesting_summary()
            backtesting_overall_stat.add_backtesting_summary(product,
                                                             trix_vpt_research.get_backtesting_summary())
            backtesting_overall_stat.add_trade_df(product,
                                                  trix_vpt_research.get_trades_df())
        except Exception as e:
            print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")
            exit(0)
        product_timings.append((product, perf_counter() - product_start))

    backtesting_overall_stat.analyze_summary_to_txt_file()
    backtesting_overall_stat.analyze_summary_to_excel_file()
    backtesting_overall_stat.analyze_trade_to_excel_file()

    print(f"[TIMING] {backtesting_allow_trade_direction.value} 各品种耗时汇总:")
    for pname, secs in product_timings:
        print(f"[TIMING]     {pname:6}: {secs:.1f}s")
    print(f"[TIMING] test_jq 全程耗时: {perf_counter() - test_start:.1f}s")


if __name__ == '__main__':
    test_jq(BacktestingAllowTradeDirection.ALL)
    test_jq(BacktestingAllowTradeDirection.LONG)
    test_jq(BacktestingAllowTradeDirection.SHORT)

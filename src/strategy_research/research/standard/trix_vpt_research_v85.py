import re
import traceback
from collections import deque
from datetime import time, date
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from strategy_research.config.backtesting import G_BACKTEST_PRODUCT_LIST
from strategy_research.config.figure import COLOR_BAR_UP, COLOR_BAR_DOWN, COLOR_WHITE, COLOR_YELLOW
from strategy_research.factor.momentum import calc_trix, calc_macd
from strategy_research.factor.ema_pack import EmaPackCrossStateMachine
from strategy_research.factor.parabolic_sar import calc_sar
from strategy_research.factor.volume_price import calc_ccl
from strategy_research.backtesting.object import BacktestingDirection, BacktestingOverallStatistics, \
    BacktestingAllowTradeDirection
from strategy_research.backtesting.pandas_backtesting_base import PandasBacktestingBase
from strategy_research.tool.data import KLineTool, BarDataManager


class TrixVptResearch(PandasBacktestingBase):

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
        # v83 = v81 + 主力判定 T-1 昨持仓口径（use_t1_major=True）——
        # 对齐 trowel 实盘 _arrange_contract（FuturesDailyData.pre_open_interest），
        # 消除研究版"当日收盘 OI 判主力"的盘中未来数据；其余口径与 v81 完全一致。
        # trowel 侧移植锚自此由 v81 更新为 v83。
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
                         allow_trade_direction,
                         use_t1_major=True,
                         filter_auction_bars=True)
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

        self.ema_pack_state = EmaPackCrossStateMachine(self.ema_pack_min_regime_days,
                                                       self.ema_pack_max_mixed_gap_days)

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

    @property
    def parameters(self):
        skip_field = [""]

        item_dict = {}

        return item_dict

    def _compute_daily_indicator(self,
                                 bar_1d_df: DataFrame) -> tuple[DataFrame, DataFrame, list]:
        daily_signal_bar_df = bar_1d_df.copy()
        daily_signal_bar_df = calc_trix(daily_signal_bar_df,
                                        self.trix_period,
                                        self.trix_ma_period, )
        daily_signal_bar_df = calc_macd(daily_signal_bar_df)
        daily_signal_bar_df = calc_ccl(daily_signal_bar_df)
        daily_signal_bar_df = calc_sar(daily_signal_bar_df,
                                       self.sar_af_start,
                                       self.sar_af_step,
                                       self.sar_af_max)
        daily_signal_bar_df = daily_signal_bar_df.assign(
            # tr
            tr=lambda x: np.maximum(
                x['high'] - x['low'],
                np.maximum(
                    (x['high'] - x['close'].shift(1)).abs(),
                    (x['low'] - x['close'].shift(1)).abs()
                )
            ),

            # atr
            atr_14=lambda x: x['tr'].ewm(alpha=1 / self.atr_period,
                                         adjust=False,
                                         min_periods=self.atr_period).mean(),
            atr_channel=lambda x: x['tr'].ewm(alpha=1 / self.channel_lookback,
                                              adjust=False,
                                              min_periods=self.channel_lookback).mean(),

            # price
            prev_high=lambda x: x['high'].shift(1),
            prev_low=lambda x: x['low'].shift(1),
            prev_close=lambda x: x['close'].shift(1),
            prev_open=lambda x: x['open'].shift(1),
            close_pct=lambda x: x['close'].pct_change(),
            close_up=lambda x: x['close'] > x['prev_close'],
            close_down=lambda x: x['close'] < x['prev_close'],

            # true gap，全天无回补缺口
            is_true_gap_up=lambda x: x['low'] > x['prev_high'],
            is_true_gap_down=lambda x: x['high'] < x['prev_low'],
            true_gap_up_point=lambda x: np.where(x['is_true_gap_up'], x['low'] - x['prev_high'], 0),
            true_gap_down_point=lambda x: np.where(x['is_true_gap_down'], x['prev_low'] - x['high'], 0),
            true_gap_up_pct=lambda x: x['true_gap_up_point'] / x['prev_close'],
            true_gap_down_pct=lambda x: x['true_gap_down_point'] / x['prev_close'],

            # K Line
            kline_range=lambda x: x["high"] - x["low"],
            kline_body=lambda x: x["close"] - x['open'],
            kline_abs_body=lambda x: x["kline_body"].abs(),
            kline_upper_shadow=lambda x: x["high"] - x[["open", "close"]].max(axis=1),
            kline_lower_shadow=lambda x: x[['open', 'close']].min(axis=1) - x['low'],
            kline_direction=lambda x: np.where(x['kline_body'] > 0, 1, np.where(x['kline_body'] < 0, -1, 0)),
            kline_range_top_n=lambda x: x["kline_range"].rolling(60).quantile(0.9),
            is_kline_long=lambda x: (x['kline_range'] >= x['kline_range_top_n']) & (x['kline_range'] > 0),
            kline_body_ratio=lambda x: x['kline_abs_body'] / x['kline_range'],
            kline_upper_shadow_ratio=lambda x: (x['kline_upper_shadow'] / x['kline_range'].replace(0, np.nan)).fillna(
                0),
            kline_lower_shadow_ratio=lambda x: (x['kline_lower_shadow'] / x['kline_range'].replace(0, np.nan)).fillna(
                0),

            # volume
            avg_volume=lambda x: x['volume'].rolling(self.volume_period).mean().shift(1),
            vol_increased=lambda x: x['volume'] > (x['avg_volume'] * self.volume_threshold),

            # ema
            ema_5=lambda x: x['close'].ewm(span=5, adjust=False).mean(),
            ema_10=lambda x: x['close'].ewm(span=10, adjust=False).mean(),
            ema_20=lambda x: x['close'].ewm(span=20, adjust=False).mean(),
            ema_40=lambda x: x['close'].ewm(span=40, adjust=False).mean(),
            ema_60=lambda x: x['close'].ewm(span=60, adjust=False).mean(),

            # channel
            channel_high=lambda x: x["high"].rolling(self.channel_lookback).max(),
            channel_low=lambda x: x["low"].rolling(self.channel_lookback).min(),
            channel_width=lambda x: x['channel_high'] - x['channel_low'],
            channel_compression=lambda x: x["channel_width"] / x["atr_channel"],
            channel_compression_top_n=lambda x: x["channel_compression"].rolling(60).quantile(0.3),
            is_channel_compression=lambda x: x["channel_compression"] <= x["channel_compression_top_n"],
            channel_cum_ret=lambda x: x["close"].pct_change(self.channel_lookback),

            short_channel_high=lambda x: x["high"].rolling(7).max(),
            short_channel_low=lambda x: x["low"].rolling(7).min(),

            # 长窗口极值（用于"距极值入场过滤"：追高/追跌过滤）
            extreme_window_high=lambda x: x["high"].rolling(self.extreme_lookback).max(),
            extreme_window_low=lambda x: x["low"].rolling(self.extreme_lookback).min(),

            # trix
            trix_hist=lambda x: x['trix'] - x['trix_signal'],
            prev_trix_hist=lambda x: x['trix_hist'].shift(1),
            trix_allow_close_long=lambda x: (
                    (x["trix_hist"] < x["prev_trix_hist"])
                    & (x["trix_hist"] < 0)
            ),
            trix_allow_close_short=lambda x: (
                    (x["trix_hist"] > x["prev_trix_hist"])
                    & (x["trix_hist"] > 0)
            ),

            # macd
            prev_macd_hist=lambda x: x["macd_hist"].shift(1),
            macd_allow_close_long=lambda x: (
                    (x["macd_hist"] < x["prev_macd_hist"])
                    & (x["macd_hist"] < 0)
            ),
            macd_allow_close_short=lambda x: (
                    (x["macd_hist"] > x["prev_macd_hist"])
                    & (x["macd_hist"] > 0)
            ),
        )

        # 计算 TRIX 正负持续天数 #
        # is_trix_hist_pos = daily_signal_bar_df['trix'] > 0
        # is_trix_hist_neg = daily_signal_bar_df['trix'] < 0
        is_trix_hist_pos = daily_signal_bar_df['trix_hist'] > 0
        is_trix_hist_neg = daily_signal_bar_df['trix_hist'] < 0
        pos_group = (is_trix_hist_pos != is_trix_hist_pos.shift(1)).cumsum()
        daily_signal_bar_df['trix_pos_days'] = is_trix_hist_pos.groupby(pos_group).cumsum()
        neg_group = (is_trix_hist_neg != is_trix_hist_neg.shift(1)).cumsum()
        daily_signal_bar_df['trix_neg_days'] = is_trix_hist_neg.groupby(neg_group).cumsum()
        ####################################################################################

        # 计算 MACD 正负持续天数 #
        # is_macd_hist_pos = daily_signal_bar_df['macd_diff'] > 0
        # is_macd_hist_neg = daily_signal_bar_df['macd_diff'] < 0
        is_macd_hist_pos = daily_signal_bar_df['macd_hist'] > 0
        is_macd_hist_neg = daily_signal_bar_df['macd_hist'] < 0
        pos_group = (is_macd_hist_pos != is_macd_hist_pos.shift(1)).cumsum()
        daily_signal_bar_df['macd_pos_days'] = is_macd_hist_pos.groupby(pos_group).cumsum()
        neg_group = (is_macd_hist_neg != is_macd_hist_neg.shift(1)).cumsum()
        daily_signal_bar_df['macd_neg_days'] = is_macd_hist_neg.groupby(neg_group).cumsum()
        ####################################################################################

        # 计算 EMA 多头排列、空头排列天数 #
        ema_long_bull = (daily_signal_bar_df['ema_5'] > daily_signal_bar_df['ema_10']) & (
                daily_signal_bar_df['ema_10'] > daily_signal_bar_df['ema_20'])
        ema_long_bear = (daily_signal_bar_df['ema_5'] < daily_signal_bar_df['ema_10']) & (
                daily_signal_bar_df['ema_10'] < daily_signal_bar_df['ema_20'])
        group_id = (ema_long_bull != ema_long_bull.shift(1)).cumsum()
        daily_signal_bar_df['ema_long_bull_days'] = ema_long_bull.groupby(group_id).cumsum()
        group_id = (ema_long_bear != ema_long_bear.shift(1)).cumsum()
        daily_signal_bar_df['ema_long_bear_days'] = ema_long_bear.groupby(group_id).cumsum()
        ####################################################################################

        daily_exec_bar_df = daily_signal_bar_df.assign(
            # atr
            atr_14=daily_signal_bar_df["atr_14"].shift(1),

            # vol
            volume=daily_signal_bar_df["volume"].shift(1),
            open_interest=daily_signal_bar_df["open_interest"].shift(1),

            # ccl
            ccl=daily_signal_bar_df["ccl"].shift(1),
            ccl_type=daily_signal_bar_df["ccl_type"].shift(1),

            # price
            close_pct=daily_signal_bar_df["close_pct"].shift(1),

            # gap
            is_true_gap_up=daily_signal_bar_df["is_true_gap_up"].shift(1),
            is_true_gap_down=daily_signal_bar_df["is_true_gap_down"].shift(1),
            true_gap_up_point=daily_signal_bar_df["true_gap_up_point"].shift(1),
            true_gap_down_point=daily_signal_bar_df["true_gap_down_point"].shift(1),
            true_gap_up_pct=daily_signal_bar_df["true_gap_up_pct"].shift(1),
            true_gap_down_pct=daily_signal_bar_df["true_gap_down_pct"].shift(1),

            # kline
            kline_direction=daily_signal_bar_df["kline_direction"].shift(1),
            is_kline_long=daily_signal_bar_df["is_kline_long"].shift(1),
            kline_body_ratio=daily_signal_bar_df["kline_body_ratio"].shift(1),
            kline_upper_shadow_ratio=daily_signal_bar_df["kline_upper_shadow_ratio"].shift(1),
            kline_lower_shadow_ratio=daily_signal_bar_df["kline_lower_shadow_ratio"].shift(1),
            kline_range_top_n=daily_signal_bar_df["kline_range_top_n"].shift(1),

            # ema
            ema_5=daily_signal_bar_df['ema_5'].shift(1),
            ema_10=daily_signal_bar_df['ema_10'].shift(1),
            ema_20=daily_signal_bar_df['ema_20'].shift(1),
            ema_40=daily_signal_bar_df['ema_40'].shift(1),
            ema_60=daily_signal_bar_df['ema_60'].shift(1),
            ema_long_bull_days=daily_signal_bar_df['ema_long_bull_days'].shift(1),
            ema_long_bear_days=daily_signal_bar_df['ema_long_bear_days'].shift(1),

            # channel
            channel_high=daily_signal_bar_df["channel_high"].shift(1),
            channel_low=daily_signal_bar_df["channel_low"].shift(1),
            is_channel_compression=daily_signal_bar_df["is_channel_compression"].shift(1),

            # 长窗口极值（前一日口径）
            extreme_window_high=daily_signal_bar_df["extreme_window_high"].shift(1),
            extreme_window_low=daily_signal_bar_df["extreme_window_low"].shift(1),

            short_channel_high=daily_signal_bar_df["short_channel_high"].shift(1),
            short_channel_low=daily_signal_bar_df["short_channel_low"].shift(1),

            # sar（前一日口径：T-1 收盘后的 SAR 趋势状态）
            sar=daily_signal_bar_df["sar"].shift(1),
            sar_trend=daily_signal_bar_df["sar_trend"].shift(1),

            # trix
            trix=daily_signal_bar_df["trix"].shift(1),
            trix_signal=daily_signal_bar_df["trix_signal"].shift(1),
            trix_hist=daily_signal_bar_df["trix_hist"].shift(1),
            trix_pos_days=daily_signal_bar_df["trix_pos_days"].shift(1),
            trix_neg_days=daily_signal_bar_df["trix_neg_days"].shift(1),
            trix_allow_close_long=daily_signal_bar_df["trix_allow_close_long"].shift(1),
            trix_allow_close_short=daily_signal_bar_df["trix_allow_close_short"].shift(1),

            # macd
            macd_diff=daily_signal_bar_df["macd_diff"].shift(1),
            macd_dea=daily_signal_bar_df["macd_dea"].shift(1),
            macd_hist=daily_signal_bar_df["macd_hist"].shift(1),
            macd_pos_days=daily_signal_bar_df["macd_pos_days"].shift(1),
            macd_neg_days=daily_signal_bar_df["macd_neg_days"].shift(1),
            macd_allow_close_long=daily_signal_bar_df["macd_allow_close_long"].shift(1),
            macd_allow_close_short=daily_signal_bar_df["macd_allow_close_short"].shift(1),
        )
        require_columns = [
            # atr
            "atr_14",

            # vol
            "volume",
            "open_interest",

            # ccl
            "ccl",
            "ccl_type",

            # price
            "close_pct",
            "open",
            "close",
            "high",
            "low",

            "prev_close",
            "prev_high",
            "prev_low",
            "prev_open",

            # gap
            "is_true_gap_up",
            "is_true_gap_down",
            "true_gap_up_point",
            "true_gap_down_point",
            "true_gap_up_pct",
            "true_gap_down_pct",

            # kline
            "kline_direction",
            "is_kline_long",
            "kline_body_ratio",
            "kline_upper_shadow_ratio",
            "kline_lower_shadow_ratio",
            "kline_range_top_n",

            # ema
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_40",
            "ema_60",
            "ema_long_bull_days",
            "ema_long_bear_days",

            # channel
            "channel_high",
            "channel_low",
            "is_channel_compression",
            "extreme_window_high",
            "extreme_window_low",

            "short_channel_high",
            "short_channel_low",

            # sar
            "sar",
            "sar_trend",

            # trix
            "trix",
            "trix_signal",
            "trix_hist",
            "trix_pos_days",
            "trix_neg_days",
            "trix_allow_close_long",
            "trix_allow_close_short",

            # macd
            "macd_diff",
            "macd_dea",
            "macd_hist",
            "macd_pos_days",
            "macd_neg_days",
            "macd_allow_close_long",
            "macd_allow_close_short",
        ]

        strict_columns = [
            # atr
            "atr_14",

            # macd
            "macd_diff",
            "macd_dea",
            "macd_hist",

            # ccl
            "ccl",
            "ccl_type",

            # kline
            "kline_range_top_n",

            # ema
            "ema_5",
            "ema_10",
            "ema_20",
            "ema_40",
            "ema_60",
            "ema_long_bull_days",
            "ema_long_bear_days",

            # channel
            "channel_high",
            "channel_low",
            "is_channel_compression",
            "extreme_window_high",
            "extreme_window_low",

            "short_channel_high",
            "short_channel_low",

            # sar
            "sar",
            "sar_trend",

            # trix
            "trix",
            "trix_hist",
            "trix_pos_days",
            "trix_neg_days",
            "trix_allow_close_long",
            "trix_allow_close_short",

            # macd
            "macd_diff",
            "macd_dea",
            "macd_hist",
            "macd_pos_days",
            "macd_neg_days",
            "macd_allow_close_long",
            "macd_allow_close_short",
        ]

        daily_exec_bar_df = daily_exec_bar_df[require_columns]
        daily_exec_bar_df = daily_exec_bar_df.dropna(subset=strict_columns)
        daily_exec_bar_df = daily_exec_bar_df.add_prefix("daily_")
        daily_exec_bar_df["trading_date"] = daily_signal_bar_df["trading_date"]
        daily_exec_bar_df = daily_exec_bar_df.reset_index(drop=True)

        daily_strict_columns = [f"daily_{item}" for item in strict_columns]

        signal_require_columns = require_columns.copy()
        signal_require_columns.append("trading_date")
        daily_signal_bar_df = daily_signal_bar_df[signal_require_columns]

        return daily_exec_bar_df, daily_signal_bar_df, daily_strict_columns

    def compute_indicators(self,
                           bar_1m_df: DataFrame,
                           session_time_type: list[tuple[time, time]]) -> tuple[DataFrame, DataFrame]:
        bar_1m_df = bar_1m_df.copy()
        bar_1m_df = KLineTool.add_column_session_to_bar_df(bar_1m_df,
                                                           session_time_type)
        bar_1d_df = KLineTool.resample_bar_df(bar_1m_df,
                                              "1d")

        daily_exec_df, daily_signal_df, daily_strict_columns = self._compute_daily_indicator(bar_1d_df)

        # 执行层
        minute_exec_bar_df = bar_1m_df.merge(daily_exec_df,
                                             on="trading_date",
                                             how="left")
        # 分钟级数据与前一日daily数据
        minute_exec_bar_df = minute_exec_bar_df.assign(
            # return
            intraday_return_prev_daily_close=lambda x: (x["close"] / x["daily_prev_close"] - 1),
            intraday_return_prev_daily_high=lambda x: (x["close"] / x["daily_prev_high"] - 1),
            intraday_return_prev_daily_low=lambda x: (x["close"] / x["daily_prev_low"] - 1),
            intraday_return_daily_open=lambda x: (x["close"] / x["daily_open"] - 1),

            # break
            intraday_above_prev_daily_high=lambda x: (x["close"] >= x["daily_prev_high"]),
            intraday_below_prev_daily_low=lambda x: (x["close"] <= x["daily_prev_low"]),
            intraday_break_prev_daily_high=lambda x: (x["high"] > x["daily_prev_high"]),
            intraday_break_prev_daily_low=lambda x: (x["low"] < x["daily_prev_low"]),

            # open gap
            intraday_is_open_gap_up=lambda x: x['daily_open'] > x['daily_prev_high'],
            intraday_is_open_gap_down=lambda x: x['daily_open'] < x['daily_prev_low'],
            intraday_open_gap_up_point=lambda x: np.where(x['intraday_is_open_gap_up'],
                                                          x['daily_open'] - x['daily_prev_high'],
                                                          0),
            intraday_open_gap_down_point=lambda x: np.where(x['intraday_is_open_gap_down'],
                                                            x['daily_prev_low'] - x['daily_open'], 0),
            intraday_open_gap_up_pct=lambda x: x["intraday_open_gap_up_point"] / x["daily_prev_close"],
            intraday_open_gap_down_pct=lambda x: x["intraday_open_gap_down_point"] / x["daily_prev_close"],
            intraday_is_open_gap_up_covered=lambda x: x['intraday_is_open_gap_up'] & (x['low'] <= x['daily_prev_high']),
            intraday_is_open_gap_down_covered=lambda x: x['intraday_is_open_gap_down'] & (
                    x['high'] >= x['daily_prev_low']),
        )

        # 分钟级数据日内情况
        group_df = minute_exec_bar_df.groupby("trading_date")
        minute_exec_bar_df = minute_exec_bar_df.assign(
            # bar index
            intraday_bar_index=group_df.cumcount() + 1,
            # high low
            intraday_prev_high=group_df["high"].transform(lambda x: x.shift(1).cummax()),
            intraday_prev_low=group_df["low"].transform(lambda x: x.shift(1).cummin()),
            intraday_prev_rolling_high_30=group_df["high"].transform(
                lambda x: x.shift(1).rolling(30, min_periods=30).max()),
            intraday_prev_rolling_low_30=group_df["low"].transform(
                lambda x: x.shift(1).rolling(30, min_periods=30).min()),

            intraday_open=group_df["open"].transform("first"),
            intraday_high=group_df["high"].cummax(),
            intraday_low=group_df["low"].cummin(),
        )

        minute_exec_bar_df = minute_exec_bar_df.assign(
            # K Line
            intraday_kline_range=lambda x: x["intraday_high"] - x["intraday_low"],
            intraday_kline_body=lambda x: x["close"] - x['intraday_open'],
            intraday_kline_abs_body=lambda x: x["intraday_kline_body"].abs(),
            intraday_kline_upper_shadow=lambda x: x["intraday_high"] - x[["intraday_open", "close"]].max(axis=1),
            intraday_kline_lower_shadow=lambda x: x[['intraday_open', 'close']].min(axis=1) - x['intraday_low'],
            intraday_kline_direction=lambda x: np.where(x['intraday_kline_body'] > 0, 1,
                                                        np.where(x['intraday_kline_body'] < 0, -1, 0)),
            intraday_kline_body_ratio=lambda x: x['intraday_kline_abs_body'] / x['intraday_kline_range'],
            intraday_kline_upper_shadow_ratio=lambda x: (
                    x['intraday_kline_upper_shadow'] / x['intraday_kline_range'].replace(0, np.nan)).fillna(
                0),
            intraday_kline_lower_shadow_ratio=lambda x: (
                    x['intraday_kline_lower_shadow'] / x['intraday_kline_range'].replace(0, np.nan)).fillna(
                0),
            intraday_is_kline_long=lambda x: (x['intraday_kline_range'] >= x['daily_kline_range_top_n']) & (
                    x['intraday_kline_range'] > 0),
        )

        minute_exec_bar_df = minute_exec_bar_df.dropna(subset=daily_strict_columns)
        minute_exec_bar_df = minute_exec_bar_df.reset_index(drop=True)

        # v85：量仓门喂入数据与 strict dropna 解耦——
        # 用完整日线序列（dropna 之前的 daily_signal_df，逐日 own 值）提取
        # (trading_date, 当日总量, 日末持仓)，供 hybrid 循环前预填门的 deque。
        self._gate_daily_feed = [
            (row["trading_date"], float(row["volume"]), float(row["open_interest"]))
            for _, row in daily_signal_df[["trading_date", "volume", "open_interest"]].iterrows()
            if pd.notna(row["trading_date"]) and pd.notna(row["volume"]) and pd.notna(row["open_interest"])
        ]

        return minute_exec_bar_df, daily_signal_df

    def _prev_days_gates_pass(self) -> bool:
        """前 N 交易日量仓综合门（T-1 口径，触发时不使用当日量仓，无未来数据）。"""
        vol_dq = list(self._gate_volume_deque)
        oi_dq = list(self._gate_oi_deque)
        need = self.reversal_prev_base_window
        if len(vol_dq) < need + 1 or len(oi_dq) < need + 1:
            return True  # 基线样本不足时不启用（fail-open）
        win = self.reversal_prev_window
        if vol_dq[-win] == vol_dq[-1] == 0:
            return True
        vol_recent = sum(vol_dq[-win:]) / win
        vol_base = sum(vol_dq[-need:]) / need
        oi_recent = sum(oi_dq[-win:]) / win
        oi_base = sum(oi_dq[-need:]) / need
        if self.reversal_prev_vol_gate and vol_base > 0:
            if vol_recent / vol_base > self.reversal_prev_vol_ratio_max:
                return False
        if self.reversal_prev_oi_gate and oi_base > 0:
            if oi_recent / oi_base > self.reversal_prev_oi_ratio_max:
                return False
        return True

    def hybrid_backtest(self,
                        bar_1m_exec_df: DataFrame,
                        verbose: bool = False):
        bar_1m_exec_df = bar_1m_exec_df.copy()

        # 日线级别指标
        ## atr
        daily_atr_14_arr = bar_1m_exec_df['daily_atr_14'].to_numpy()

        ## channel
        daily_channel_high_arr = bar_1m_exec_df['daily_channel_high'].to_numpy()
        daily_channel_low_arr = bar_1m_exec_df['daily_channel_low'].to_numpy()
        daily_is_channel_compression_arr = bar_1m_exec_df['daily_is_channel_compression'].to_numpy()

        ## 长窗口极值（前一日口径）
        daily_extreme_window_high_arr = bar_1m_exec_df["daily_extreme_window_high"].to_numpy()
        daily_extreme_window_low_arr = bar_1m_exec_df["daily_extreme_window_low"].to_numpy()

        ## sar（前一日口径：T-1 收盘后的 SAR 趋势，+1 上涨趋势 / -1 下跌趋势）
        daily_sar_trend_arr = bar_1m_exec_df["daily_sar_trend"].to_numpy()

        ## 均线束反转搏击单的确认：T-1 ccl_type（价格方向 × 增减仓四象限）
        daily_ccl_type_arr = bar_1m_exec_df["daily_ccl_type"].to_numpy()

        ## 反转搏击单的结构位止损参考：T-1 近期 7 日高低点
        daily_short_channel_high_arr = bar_1m_exec_df["daily_short_channel_high"].to_numpy()
        daily_short_channel_low_arr = bar_1m_exec_df["daily_short_channel_low"].to_numpy()

        ## 贴近极值锚：T-1 的 60 日高/低点（extreme window）与收盘
        daily_close_arr = bar_1m_exec_df["close"].to_numpy()

        ## true gap
        daily_prev_true_gap_up_pct_arr = bar_1m_exec_df["daily_true_gap_up_pct"].to_numpy()
        daily_prev_true_gap_down_pct_arr = bar_1m_exec_df["daily_true_gap_down_pct"].to_numpy()

        ## trix
        daily_trix_allow_close_long_arr = bar_1m_exec_df['daily_trix_allow_close_long'].to_numpy()
        daily_trix_allow_close_short_arr = bar_1m_exec_df['daily_trix_allow_close_short'].to_numpy()

        daily_trix_pos_days_arr = bar_1m_exec_df["daily_trix_pos_days"].to_numpy()
        daily_trix_neg_days_arr = bar_1m_exec_df["daily_trix_neg_days"].to_numpy()

        ## macd
        daily_macd_allow_close_long_arr = bar_1m_exec_df['daily_macd_allow_close_long'].to_numpy()
        daily_macd_allow_close_short_arr = bar_1m_exec_df['daily_macd_allow_close_short'].to_numpy()

        daily_macd_pos_days_arr = bar_1m_exec_df["daily_macd_pos_days"].to_numpy()
        daily_macd_neg_days_arr = bar_1m_exec_df["daily_macd_neg_days"].to_numpy()

        ## kline
        daily_prev_kline_direction_arr = bar_1m_exec_df["daily_kline_direction"].to_numpy()
        daily_prev_is_kline_long_arr = bar_1m_exec_df["daily_is_kline_long"].to_numpy()
        daily_prev_kline_body_ratio_arr = bar_1m_exec_df["daily_kline_body_ratio"].to_numpy()

        ## ema
        daily_ema5_arr = bar_1m_exec_df["daily_ema_5"].to_numpy()
        daily_ema10_arr = bar_1m_exec_df["daily_ema_10"].to_numpy()
        daily_ema20_arr = bar_1m_exec_df["daily_ema_20"].to_numpy()
        daily_ema40_arr = bar_1m_exec_df["daily_ema_40"].to_numpy()
        daily_ema_long_bull_days_arr = bar_1m_exec_df["daily_ema_long_bull_days"].to_numpy()
        daily_ema_long_bear_days_arr = bar_1m_exec_df["daily_ema_long_bear_days"].to_numpy()

        # 分钟级指标

        ## 价格涨跌情况
        intraday_return_prev_daily_close_arr = bar_1m_exec_df['intraday_return_prev_daily_close'].to_numpy()
        intraday_return_daily_open_arr = bar_1m_exec_df['intraday_return_daily_open'].to_numpy()

        ## 日内索引
        intraday_bar_index_arr = bar_1m_exec_df['intraday_bar_index'].to_numpy()

        ## 开盘跳空
        intraday_open_gap_up_pct_arr = bar_1m_exec_df["intraday_open_gap_up_pct"].to_numpy()
        intraday_open_gap_down_pct_arr = bar_1m_exec_df["intraday_open_gap_down_pct"].to_numpy()
        intraday_is_open_gap_up_covered_arr = bar_1m_exec_df["intraday_is_open_gap_up_covered"].to_numpy()
        intraday_is_open_gap_down_covered_arr = bar_1m_exec_df["intraday_is_open_gap_down_covered"].to_numpy()

        ## 当日K线形态
        intraday_kline_direction_arr = bar_1m_exec_df["intraday_kline_direction"].to_numpy()
        intraday_is_kline_long_arr = bar_1m_exec_df["intraday_is_kline_long"].to_numpy()
        intraday_kline_body_ratio_arr = bar_1m_exec_df["intraday_kline_body_ratio"].to_numpy()

        # 回测数据
        minute_close_arr = bar_1m_exec_df['close'].to_numpy()
        minute_high_arr = bar_1m_exec_df['high'].to_numpy()
        minute_low_arr = bar_1m_exec_df['low'].to_numpy()
        datetime_arr = bar_1m_exec_df['datetime'].astype(object).to_numpy()
        trading_date_arr = bar_1m_exec_df['trading_date'].to_numpy()
        ## 逻辑交易日
        trading_date_unique_list = sorted(bar_1m_exec_df['trading_date'].unique())
        trading_date_to_index = {
            trading_date: i for i, trading_date in enumerate(trading_date_unique_list)
        }

        ## 上一根Bar的交易日期
        open_memory: dict[date, bool] = {}
        last_trading_date = None

        ## 前 N 日量仓综合门：按交易日的总量/最后持仓（deque 以“日”为单位）
        self._gate_day_volume = 0.0
        self._gate_day_last_oi = None
        self._gate_day_date = None
        self._gate_volume_deque = deque(maxlen=max(self.reversal_prev_base_window + 2, 20))
        self._gate_oi_deque = deque(maxlen=max(self.reversal_prev_base_window + 2, 20))

        # v85：预填——exec_df 首个交易日之前的门历史从完整日线序列补齐，
        # 消除"门的喂入起点被 strict dropna 推迟"导致的 fail-open 不对称。
        first_exec_date = trading_date_arr[0]
        try:
            first_exec_date = pd.Timestamp(first_exec_date).date()
        except Exception:
            first_exec_date = first_exec_date
        if getattr(self, "_gate_daily_feed", None):
            for feed_date, feed_vol, feed_oi in self._gate_daily_feed:
                try:
                    feed_d = pd.Timestamp(feed_date).date()
                except Exception:
                    feed_d = feed_date
                if feed_d >= first_exec_date:
                    continue  # exec_df 起步后的天数仍由循环内收口喂入
                self._gate_volume_deque.append(feed_vol)
                self._gate_oi_deque.append(feed_oi)
        minute_oi_arr = bar_1m_exec_df["open_interest"].to_numpy()
        minute_volume_arr = bar_1m_exec_df["volume"].to_numpy()

        ## 持仓后数据统计
        after_holding_high_bar_index = -1
        after_holding_low_bar_index = -1

        bar_exec_df_copy_length = len(bar_1m_exec_df)

        # 开始逐根 Bar 混合回测
        for i in range(bar_exec_df_copy_length):
            # 交易日和日内索引更新
            cur_trading_date = trading_date_arr[i]
            if last_trading_date != cur_trading_date:
                if last_trading_date is not None:
                    # 如果上一个交易日不是 None，则去更新权益
                    self.account.record_daily_equity(last_trading_date)

                # 上一个交易日设置成当前的交易日
                last_trading_date = cur_trading_date

                # 均线束状态机推进：每个交易日一次（数据显示的是此前一日收盘后的三线状态，无未来数据）
                self.ema_pack_state.on_new_trading_day(daily_ema5_arr[i],
                                                       daily_ema10_arr[i],
                                                       daily_ema20_arr[i],
                                                       daily_ema40_arr[i])

            # cur_position = self.account.get_position(self.symbol)
            cur_position = self.account.get_position_v2()
            cur_datetime = datetime_arr[i]
            cur_high = minute_high_arr[i]
            cur_low = minute_low_arr[i]

            cur_return_today_open = intraday_return_daily_open_arr[i]
            cur_intraday_bar_index = intraday_bar_index_arr[i]

            # 量仓综合门状态：新交易日首根先收口上一交易日总量/最后持仓
            if self._gate_day_date != cur_trading_date:
                if self._gate_day_date is not None:
                    self._gate_volume_deque.append(self._gate_day_volume)
                    self._gate_oi_deque.append(self._gate_day_last_oi)
                self._gate_day_date = cur_trading_date
                self._gate_day_volume = 0.0
            self._gate_day_volume += minute_volume_arr[i]
            self._gate_day_last_oi = minute_oi_arr[i]

            cur_is_channel_compression = daily_is_channel_compression_arr[i]
            cur_channel_high = daily_channel_high_arr[i]
            cur_channel_low = daily_channel_low_arr[i]
            cur_daily_atr_14 = daily_atr_14_arr[i]

            if not self._is_in_trading_time(cur_datetime):
                continue

            # 最新价更新
            cur_close = minute_close_arr[i]

            # 距极值入场过滤（前一日口径的 60 日高低点）：
            # 做多：距 60 日低点涨幅达到阈值 → 放弃追多；做空：距 60 日高点跌幅达到阈值 → 放弃追空
            cur_can_open_long = cur_close < (daily_extreme_window_low_arr[i] * (1 + self.max_extreme_distance_pct))
            cur_can_open_short = cur_close > (daily_extreme_window_high_arr[i] * (1 - self.max_extreme_distance_pct))
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
                continue

            channel_can_open_long_condition = (
                    cur_is_channel_compression
                    and cur_close > (cur_channel_high + 0.1 * cur_daily_atr_14)
                    and cur_can_open_long
                    and cur_intraday_bar_index >= 30
                    and cur_return_today_open > 0
                    and (daily_ema5_arr[i] > daily_ema10_arr[i] > daily_ema20_arr[i])
                    and daily_ema_long_bull_days_arr[i] >= 3
                    and daily_sar_trend_arr[i] == 1
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
                    and (daily_ema5_arr[i] < daily_ema10_arr[i] < daily_ema20_arr[i])
                    and daily_ema_long_bear_days_arr[i] >= 3
                    and daily_sar_trend_arr[i] == -1
            )
            retrace_distance_high = daily_extreme_window_high_arr[i] - daily_close_arr[i]
            ema_pack_can_open_short_condition = (
                    daily_sar_trend_arr[i] == -1
                    and retrace_distance_high <= self.reversal_retrace_atr_mult * cur_daily_atr_14
                    and cur_close < daily_short_channel_low_arr[i]
                    and cur_can_open_short
                    and cur_intraday_bar_index >= 30
                    and self._prev_days_gates_pass()
            )

            retrace_distance_low = daily_close_arr[i] - daily_extreme_window_low_arr[i]
            ema_pack_can_open_long_condition = (
                    daily_sar_trend_arr[i] == 1
                    and retrace_distance_low <= self.reversal_retrace_atr_mult * cur_daily_atr_14
                    and cur_close > daily_short_channel_high_arr[i]
                    and cur_can_open_long
                    and cur_intraday_bar_index >= 30
                    and self._prev_days_gates_pass()
            )

            if cur_position is None or cur_position.volume == 0:
                """开仓检查"""
                has_memory = open_memory.get(cur_trading_date, False)
                if has_memory:
                    continue

                open_reason = ""
                if channel_can_open_long_condition or ema_pack_can_open_long_condition:
                    """多头开仓"""
                    if self.allow_trade_direction == BacktestingAllowTradeDirection.SHORT:
                        continue

                    # 通道压缩路径：止损 max(通道低点, 入场价 - 2*ATR)（v56 原口径）；
                    # 反转搏击路径：结构位小止损 = 近期低点 - N*tick（打穿结构位 = 反转失败即离场）
                    if channel_can_open_long_condition:
                        stop_loss = max(cur_channel_low, cur_close - cur_daily_atr_14 * self.stop_loss_multiplier)
                        open_reason = "channel_condition"
                    else:
                        stop_loss = daily_short_channel_low_arr[i] - self.stop_loss_ticks * self.price_tick
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
                    after_holding_high_bar_index = i
                    after_holding_low_bar_index = i
                    open_memory[cur_trading_date] = True

                elif channel_can_open_short_condition or ema_pack_can_open_short_condition:
                    """空头开仓"""
                    if self.allow_trade_direction == BacktestingAllowTradeDirection.LONG:
                        continue

                    # 通道压缩路径：止损 min(通道高点, 入场价 + 2*ATR)（v56 原口径）；
                    # 反转搏击路径：结构位小止损 = 近期高点 + N*tick
                    if channel_can_open_short_condition:
                        stop_loss = min(cur_channel_high, cur_close + cur_daily_atr_14 * self.stop_loss_multiplier)
                        open_reason = "channel_condition"
                    else:
                        stop_loss = daily_short_channel_high_arr[i] + self.stop_loss_ticks * self.price_tick
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
                    after_holding_high_bar_index = i
                    after_holding_low_bar_index = i
                    open_memory[cur_trading_date] = True
            else:
                """平仓检查"""
                position_symbol = cur_position.symbol
                position_bar_data = self._get_contract_bar(position_symbol, cur_datetime)
                if cur_high >= minute_high_arr[after_holding_high_bar_index]:
                    after_holding_high_bar_index = i

                if cur_low <= minute_low_arr[after_holding_low_bar_index]:
                    after_holding_low_bar_index = i

                close_reason = None
                is_switch_contract = False
                is_reverse_open = False
                open_trading_date = cur_position.open_trading_date
                holding_trading_days = trading_date_to_index.get(cur_trading_date) - trading_date_to_index.get(
                    open_trading_date)
                if cur_position.direction == BacktestingDirection.LONG:
                    """多头平仓检查"""
                    adjust_stop_loss = -1
                    if cur_trading_date != cur_position.open_trading_date:
                        after_holding_high = minute_high_arr[after_holding_high_bar_index]
                        adjust_stop_loss = max(cur_position.stop_loss, after_holding_high * 0.8)

                    if channel_can_open_short_condition:
                        close_reason = '达到开空条件，平仓开空'
                        is_reverse_open = True
                    elif cur_close <= cur_position.stop_loss:
                        close_reason = '止损'
                    elif (adjust_stop_loss > 0
                          and cur_close < adjust_stop_loss):
                        close_reason = "动态止盈止损"
                    elif intraday_open_gap_down_pct_arr[i] > 0.01:
                        close_reason = "开盘跳空反转"
                    elif (daily_prev_true_gap_down_pct_arr[i] > 0.005
                          and cur_intraday_bar_index >= 30
                          and cur_return_today_open < 0
                    ):
                        close_reason = "跳空反转"
                    elif (daily_prev_is_kline_long_arr[i]
                          and daily_prev_kline_body_ratio_arr[i] >= 0.8
                          and daily_prev_kline_direction_arr[i] == -1
                    ):
                        close_reason = "前一根巨型阴线"
                    elif (
                            daily_trix_allow_close_long_arr[i]
                            and holding_trading_days >= self.min_required_holding_days
                            and not cur_is_channel_compression
                            and daily_trix_neg_days_arr[i] >= 5
                    ):
                        close_reason = 'trix 信号反转平仓'
                    elif i == bar_exec_df_copy_length - 1:
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
                                continue

                            stop_loss = min(cur_channel_high, cur_close + cur_daily_atr_14 * self.stop_loss_multiplier)
                            take_profit = cur_close - cur_daily_atr_14 * self.take_profit_multiplier
                            reverse_contract_symbol, reverse_contract_close = self._get_next_major_contract_data(
                                cur_datetime
                            )
                            self.open_position(
                                reverse_contract_symbol,
                                reverse_contract_close,
                                BacktestingDirection.SHORT,
                                "",
                                cur_datetime,
                                cur_trading_date,
                                stop_loss,
                                take_profit,
                                0,
                                cur_close,
                            )
                            after_holding_high_bar_index = i
                            after_holding_low_bar_index = i
                            open_memory[cur_trading_date] = True

                elif cur_position.direction == BacktestingDirection.SHORT:
                    """空仓平仓检查"""
                    adjust_stop_loss = -1
                    if cur_trading_date != cur_position.open_trading_date:
                        after_holding_low = minute_low_arr[after_holding_low_bar_index]
                        adjust_stop_loss = min(cur_position.stop_loss, after_holding_low * 1.2)

                    if channel_can_open_long_condition:
                        close_reason = '达到开多条件，平仓开多'
                        is_reverse_open = True
                    elif cur_close >= cur_position.stop_loss:
                        close_reason = '止损'
                    elif (adjust_stop_loss > 0
                          and cur_close > adjust_stop_loss):
                        close_reason = '动态止盈止损'
                    elif intraday_open_gap_up_pct_arr[i] > 0.01:
                        close_reason = "开盘跳空反转"
                    elif (daily_prev_true_gap_up_pct_arr[i] > 0.005
                          and cur_intraday_bar_index >= 30
                          and cur_return_today_open > 0
                    ):
                        close_reason = "跳空反转"
                    elif (daily_prev_is_kline_long_arr[i]
                          and daily_prev_kline_body_ratio_arr[i] >= 0.8
                          and daily_prev_kline_direction_arr[i] == 1
                            # and cur_intraday_bar_index >= 5
                            # and cur_return_today_open > 0
                    ):
                        close_reason = "前一根巨型阳线"
                    # elif cur_close > cur_channel_high:
                    #     close_reason = '价格高于通道高点'
                    elif (
                            daily_trix_allow_close_short_arr[i]
                            and holding_trading_days >= self.min_required_holding_days
                            and not cur_is_channel_compression
                            and daily_trix_pos_days_arr[i] >= 5
                    ):
                        close_reason = 'trix 信号反转平仓'
                    # elif (daily_zigzag_structure_confirmed_arr[i] == "HH-HL"
                    #       and cur_close > daily_zigzag_last_low_confirmed_arr[i]):
                    #     close_reason = "HH-HL 上升结构"
                    elif i == bar_exec_df_copy_length - 1:
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
                                continue

                            stop_loss = max(cur_channel_low, cur_close - cur_daily_atr_14 * self.stop_loss_multiplier)
                            take_profit = cur_close + cur_daily_atr_14 * self.take_profit_multiplier
                            reverse_contract_symbol, reverse_contract_close = self._get_next_major_contract_data(
                                cur_datetime
                            )
                            self.open_position(
                                reverse_contract_symbol,
                                reverse_contract_close,
                                BacktestingDirection.LONG,
                                "",
                                cur_datetime,
                                cur_trading_date,
                                stop_loss,
                                take_profit,
                                0,
                                cur_close,
                            )
                            after_holding_high_bar_index = i
                            after_holding_low_bar_index = i
                            open_memory[cur_trading_date] = True

        cur_position = self.account.get_position_v2()
        if cur_position:
            raise Exception("unexpected, backtesting is over, but we have position ever")

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
    factor_name = "TRIX-VPT"
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
    for product in backtest_product_list:
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

    backtesting_overall_stat.analyze_summary_to_txt_file()
    backtesting_overall_stat.analyze_summary_to_excel_file()
    backtesting_overall_stat.analyze_trade_to_excel_file()


if __name__ == '__main__':
    test_jq(BacktestingAllowTradeDirection.ALL)  # 本轮 trix 消融分析只运行 ALL 方向
    test_jq(BacktestingAllowTradeDirection.LONG)  # 本轮 trix 消融分析只运行 ALL 方向
    test_jq(BacktestingAllowTradeDirection.SHORT)  # 本轮 trix 消融分析只运行 ALL 方向

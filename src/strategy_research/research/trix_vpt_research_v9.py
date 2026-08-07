import re
import traceback
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from strategy_research.config.backtesting import G_BACKTEST_PRODUCT_LIST
from strategy_research.config.exchange import get_contract_config
from strategy_research.factor.momentum import calc_trix, calc_macd
from strategy_research.factor.volume_price import calc_vpt, calc_ccl
from strategy_research.backtesting.object import BacktestingDirection, BacktestingOverallStatistics
from strategy_research.backtesting.pandas_backtesting_base import PandasBacktestingBase
from strategy_research.tool.contract import ContractTool
from strategy_research.tool.data import KLineTool, BarDataManager


class TrixVptResearch(PandasBacktestingBase):

    def __init__(self,
                 factor_name: str,
                 symbol: str,
                 bar_period: str,
                 report_dir: str,
                 initial_capital: float,
                 enable_fig_daily_mode: bool,
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
                 vpt_z_score_threshold: float = 1.5, ):
        super().__init__(factor_name,
                         symbol,
                         bar_period,
                         report_dir,
                         initial_capital,
                         enable_fig_daily_mode)
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

    @property
    def parameters(self):
        skip_field = [""]

        item_dict = {}

        return item_dict

    def compute_indicators(self,
                           bar_df: DataFrame,
                           session_time_type: list[tuple[time, time]]) -> tuple[DataFrame, DataFrame]:
        bar_df_copy = bar_df.copy()
        bar_df_copy = KLineTool.add_column_session_to_bar_df(bar_df_copy,
                                                             session_time_type)
        daily_signal_bar_df = KLineTool.resample_bar_df(bar_df_copy,
                                                        "1d")

        daily_signal_bar_df = calc_trix(daily_signal_bar_df,
                                        self.trix_period,
                                        self.trix_ma_period, )
        daily_signal_bar_df = calc_vpt(daily_signal_bar_df,
                                       self.vpt_ma_period, )
        daily_signal_bar_df = calc_macd(daily_signal_bar_df)
        daily_signal_bar_df = calc_ccl(daily_signal_bar_df)
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
            close_pct=lambda x: x['close'].pct_change(),
            prev_high=lambda x: x['high'].shift(1),
            prev_low=lambda x: x['low'].shift(1),
            prev_close=lambda x: x['close'].shift(1),
            prev_open=lambda x: x['open'].shift(1),
            close_up=lambda x: x['close'] > x['prev_close'],
            close_down=lambda x: x['close'] < x['prev_close'],

            is_true_gap_up=lambda x: x['low'] > x['prev_high'],
            is_true_gap_down=lambda x: x['high'] < x['prev_low'],
            true_gap_up_point=lambda x: np.where(x['is_true_gap_up'], x['low'] - x['prev_high'], 0),
            true_gap_down_point=lambda x: np.where(x['is_true_gap_down'], x['prev_low'] - x['high'], 0),

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
            kline_upper_shadow_ratio=lambda x: (x['kline_upper_shadow'] / x['kline_range']).replace(0, np.nan).fillna(
                0),
            kline_lower_shadow_ratio=lambda x: (x['kline_lower_shadow'] / x['kline_range']).replace(0, np.nan).fillna(
                0),

            # volume
            avg_volume=lambda x: x['volume'].rolling(self.volume_period).mean().shift(1),
            vol_increased=lambda x: x['volume'] > (x['avg_volume'] * self.volume_threshold),

            # ema
            ema_5=lambda x: x['close'].ewm(span=5, adjust=False).mean(),
            ema_10=lambda x: x['close'].ewm(span=10, adjust=False).mean(),
            ema_20=lambda x: x['close'].ewm(span=20, adjust=False).mean(),
            ema_40=lambda x: x['close'].ewm(span=40, adjust=False).mean(),

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

            # trix
            trix_hist=lambda x: x['trix'] - x['trix_signal'],
            prev_trix_hist=lambda x: x['trix_hist'].shift(1),

            # vpt
            vpt_hist=lambda x: x['vpt'] - x['vpt_ma'],
            prev_vpt_hist=lambda x: x['vpt_hist'].shift(1),
        )

        is_trix_hist_pos = daily_signal_bar_df['trix_hist'] > 0
        is_trix_hist_neg = daily_signal_bar_df['trix_hist'] < 0

        pos_group = (is_trix_hist_pos != is_trix_hist_pos.shift(1)).cumsum()
        daily_signal_bar_df['trix_pos_days'] = is_trix_hist_pos.groupby(pos_group).cumsum()

        neg_group = (is_trix_hist_neg != is_trix_hist_neg.shift(1)).cumsum()
        daily_signal_bar_df['trix_neg_days'] = is_trix_hist_neg.groupby(neg_group).cumsum()

        # 入场信号
        price_allow_open_long = (
                daily_signal_bar_df['close_up']
                & daily_signal_bar_df['vol_increased']
        )
        price_allow_open_short = (
                daily_signal_bar_df['close_down']
                & daily_signal_bar_df['vol_increased']
        )

        trix_allow_open_long = (
                (daily_signal_bar_df["trix_hist"] > daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] > 0)
        )
        trix_allow_open_short = (
                (daily_signal_bar_df["trix_hist"] < daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] < 0)
        )

        # 平仓信号
        trix_allow_close_long = (
                (daily_signal_bar_df["trix_hist"] < daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] < 0)
        )
        trix_allow_close_short = (
                (daily_signal_bar_df["trix_hist"] > daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] > 0)
        )

        daily_signal_bar_df["allow_open_long"] = (
                price_allow_open_long
                & trix_allow_open_long
        )
        daily_signal_bar_df["allow_open_short"] = (
                price_allow_open_short
                & trix_allow_open_short
        )

        daily_signal_bar_df['allow_close_long'] = (
            trix_allow_close_long
        )
        daily_signal_bar_df['allow_close_short'] = (
            trix_allow_close_short
        )

        daily_exec_bar_df = daily_signal_bar_df.assign(
            # atr
            atr_14=daily_signal_bar_df["atr_14"].shift(1),

            # macd
            macd_diff=daily_signal_bar_df["macd_diff"].shift(1),
            macd_dea=daily_signal_bar_df["macd_dea"].shift(1),
            macd_hist=daily_signal_bar_df["macd_hist"].shift(1),

            # ccl
            ccl=daily_signal_bar_df["ccl"].shift(1),
            ccl_type=daily_signal_bar_df["ccl_type"].shift(1),

            # price
            prev_close_pct=daily_signal_bar_df["close_pct"].shift(1),
            today_open=daily_signal_bar_df["open"],
            prev_close=daily_signal_bar_df["prev_close"],
            prev_high=daily_signal_bar_df["prev_high"],
            prev_low=daily_signal_bar_df["prev_low"],

            prev_is_true_gap_up=daily_signal_bar_df["is_true_gap_up"].shift(1),
            prev_is_true_gap_down=daily_signal_bar_df["is_true_gap_down"].shift(1),
            prev_true_gap_up_point=daily_signal_bar_df["true_gap_up_point"].shift(1),
            prev_true_gap_down_point=daily_signal_bar_df["true_gap_down_point"].shift(1),

            # kline
            prev_kline_direction=daily_signal_bar_df["kline_direction"].shift(1),
            prev_is_kline_long=daily_signal_bar_df["is_kline_long"].shift(1),
            prev_kline_body_ratio=daily_signal_bar_df["kline_body_ratio"].shift(1),
            prev_kline_upper_shadow_ratio=daily_signal_bar_df["kline_upper_shadow_ratio"].shift(1),
            prev_kline_lower_shadow_ratio=daily_signal_bar_df["kline_lower_shadow_ratio"].shift(1),

            # ema
            ema_5=daily_signal_bar_df['ema_5'].shift(1),
            ema_10=daily_signal_bar_df['ema_10'].shift(1),
            ema_20=daily_signal_bar_df['ema_20'].shift(1),
            ema_40=daily_signal_bar_df['ema_40'].shift(1),

            # channel
            channel_high=lambda x: x["channel_high"].shift(1),
            channel_low=lambda x: x["channel_low"].shift(1),
            is_channel_compression=lambda x: x["is_channel_compression"].shift(1),

            short_channel_high=lambda x: x["short_channel_high"].shift(1),
            short_channel_low=lambda x: x["short_channel_low"].shift(1),

            # trix
            trix=lambda x: x["trix"].shift(1),
            trix_hist=lambda x: x["trix_hist"].shift(1),
            trix_pos_days=lambda x: x["trix_pos_days"].shift(1),
            trix_neg_days=lambda x: x["trix_neg_days"].shift(1),

            # condition
            allow_open_long=lambda x: x["allow_open_long"].shift(1),
            allow_open_short=lambda x: x["allow_open_short"].shift(1),
            allow_close_long=lambda x: x["allow_close_long"].shift(1),
            allow_close_short=lambda x: x["allow_close_short"].shift(1),

        )
        daily_exec_bar_df = daily_exec_bar_df.add_prefix("daily_")
        daily_exec_bar_df["trading_date"] = daily_signal_bar_df["trading_date"]

        daily_exec_bar_df.to_csv(f"{self.symbol}.csv", index=True)

        # 执行层
        minute_exec_bar_df = bar_df_copy.merge(daily_exec_bar_df,
                                               on="trading_date",
                                               how="left")
        # 分钟级数据与前一日daily数据
        minute_exec_bar_df = minute_exec_bar_df.assign(
            intraday_return_prev_daily_close=lambda x: (x["close"] / x["daily_prev_close"] - 1),
            intraday_return_prev_daily_high=lambda x: (x["close"] / x["daily_prev_high"] - 1),
            intraday_return_prev_daily_low=lambda x: (x["close"] / x["daily_prev_low"] - 1),
            intraday_return_today_open=lambda x: (x["close"] / x["daily_today_open"] - 1),
            intraday_above_prev_daily_high=lambda x: (x["close"] >= x["daily_prev_high"]),
            intraday_below_prev_daily_low=lambda x: (x["close"] <= x["daily_prev_low"]),
            intraday_break_prev_daily_high=lambda x: (x["high"] > x["daily_prev_high"]),
            intraday_break_prev_daily_low=lambda x: (x["low"] < x["daily_prev_low"]),
        )

        # 分钟级数据日内情况
        group_df = minute_exec_bar_df.groupby("trading_date")
        minute_exec_bar_df = minute_exec_bar_df.assign(
            intraday_bar_index=group_df.cumcount() + 1,
            intraday_high=group_df["high"].shift(1).cummax(),
            intraday_low=group_df["low"].shift(1).cummin(),
            intraday_rolling_high_30=group_df["high"].transform(
                lambda x: x.shift(1).rolling(30, min_periods=30).max()),
            intraday_rolling_low_30=group_df["low"].transform(
                lambda x: x.shift(1).rolling(30, min_periods=30).min()),
        )

        critical_columns = [
            "daily_atr_14",
            "daily_ema_40",

            "daily_channel_width",
            "daily_channel_compression_top_n",

            "daily_trix_hist",

            "daily_vpt_hist",

            "daily_allow_close_long",
            "daily_allow_close_short",
        ]
        minute_exec_bar_df.dropna(subset=critical_columns, inplace=True)
        minute_exec_bar_df.reset_index(drop=True, inplace=True)
        return minute_exec_bar_df, daily_signal_bar_df

    def hybrid_backtest(self,
                        bar_exec_df: DataFrame,
                        verbose: bool = False):
        bar_exec_df_copy = bar_exec_df.copy()

        minute_close_arr = bar_exec_df_copy['close'].to_numpy()
        minute_high_arr = bar_exec_df_copy['high'].to_numpy()
        minute_low_arr = bar_exec_df_copy['low'].to_numpy()

        intraday_return_prev_daily_close_arr = bar_exec_df_copy['intraday_return_prev_daily_close'].to_numpy()
        intraday_return_today_open_arr = bar_exec_df_copy['intraday_return_today_open'].to_numpy()
        minute_intraday_bar_index_arr = bar_exec_df_copy['intraday_bar_index'].to_numpy()

        daily_atr_14_arr = bar_exec_df_copy['daily_atr_14'].to_numpy()

        daily_allow_open_long_arr = bar_exec_df_copy['daily_allow_open_long'].to_numpy()
        daily_allow_open_short_arr = bar_exec_df_copy['daily_allow_open_short'].to_numpy()
        daily_allow_close_long_arr = bar_exec_df_copy['daily_allow_close_long'].to_numpy()
        daily_allow_close_short_arr = bar_exec_df_copy['daily_allow_close_short'].to_numpy()

        daily_channel_high_arr = bar_exec_df_copy['daily_channel_high'].to_numpy()
        daily_channel_low_arr = bar_exec_df_copy['daily_channel_low'].to_numpy()
        daily_is_channel_compression_arr = bar_exec_df_copy['daily_is_channel_compression'].to_numpy()

        daily_short_channel_high_arr = bar_exec_df_copy['daily_short_channel_high'].to_numpy()
        daily_short_channel_low_arr = bar_exec_df_copy['daily_short_channel_low'].to_numpy()

        daily_prev_is_true_gap_up_arr = bar_exec_df_copy["daily_prev_is_true_gap_up"].to_numpy()
        daily_prev_is_true_gap_down_arr = bar_exec_df_copy["daily_prev_is_true_gap_down"].to_numpy()
        daily_prev_true_gap_up_point_arr = bar_exec_df_copy["daily_prev_true_gap_up_point"].to_numpy()
        daily_prev_true_gap_down_point_arr = bar_exec_df_copy["daily_prev_true_gap_down_point"].to_numpy()

        daily_trix_pos_days_arr = bar_exec_df_copy["daily_trix_pos_days"].to_numpy()
        daily_trix_neg_days_arr = bar_exec_df_copy["daily_trix_neg_days"].to_numpy()

        prev_kline_direction_arr = bar_exec_df_copy["daily_prev_kline_direction"].to_numpy()
        prev_is_kline_long_arr = bar_exec_df_copy["daily_prev_is_kline_long"].to_numpy()
        prev_kline_body_ratio_arr = bar_exec_df_copy["daily_prev_kline_body_ratio"].to_numpy()
        prev_kline_upper_shadow_ratio_arr = bar_exec_df_copy["daily_prev_kline_upper_shadow_ratio"].to_numpy()
        prev_kline_lower_shadow_ratio_arr = bar_exec_df_copy["daily_prev_kline_lower_shadow_ratio"].to_numpy()

        datetime_arr = bar_exec_df_copy['datetime'].astype(object).to_numpy()
        trading_date_arr = bar_exec_df_copy['trading_date'].to_numpy()

        trading_date_unique_list = sorted(bar_exec_df_copy['trading_date'].unique())
        trading_date_to_index = {
            trading_date: i for i, trading_date in enumerate(trading_date_unique_list)
        }
        bar_exec_df_copy_length = len(bar_exec_df_copy)

        # 上一根Bar的交易日期
        last_trading_date = None

        # 持仓后数据统计
        after_holding_high_bar_index = -1
        after_holding_low_bar_index = -1

        # 开始逐根 Bar 混合回测
        for i in range(bar_exec_df_copy_length):
            cur_position = self.account.get_position(self.symbol)
            cur_trading_date = trading_date_arr[i]
            cur_datetime = datetime_arr[i]
            cur_close = minute_close_arr[i]
            cur_high = minute_high_arr[i]
            cur_low = minute_low_arr[i]

            self.account.update_latest_price(self.symbol,
                                             cur_close)

            cur_return_prev_daily_close = intraday_return_prev_daily_close_arr[i]
            cur_return_today_open = intraday_return_today_open_arr[i]
            cur_intraday_bar_index = minute_intraday_bar_index_arr[i]

            cur_is_channel_compression = daily_is_channel_compression_arr[i]
            cur_channel_high = daily_channel_high_arr[i]
            cur_channel_low = daily_channel_low_arr[i]
            cur_daily_atr_14 = daily_atr_14_arr[i]

            if last_trading_date is None:
                last_trading_date = cur_trading_date

            if last_trading_date != cur_trading_date:
                self.account.record_daily_equity(last_trading_date)

            last_trading_date = cur_trading_date
            if cur_position is None or cur_position.volume == 0:
                """开仓检查"""
                open_reason = ""
                if (
                        cur_is_channel_compression
                        and cur_close > (cur_channel_high + 0.1 * cur_daily_atr_14)
                        and cur_intraday_bar_index >= 30
                        and cur_return_today_open > 0
                ):
                    """多头开仓"""
                    stop_loss = max(cur_channel_low, cur_close - cur_daily_atr_14 * self.stop_loss_multiplier)
                    take_profit = cur_close + cur_daily_atr_14 * self.take_profit_multiplier
                    self.open_position(
                        self.symbol,
                        cur_close,
                        BacktestingDirection.LONG,
                        open_reason,
                        cur_datetime,
                        cur_trading_date,
                        stop_loss,
                        take_profit,
                        0
                    )
                    after_holding_high_bar_index = i
                    after_holding_low_bar_index = i

                elif (
                        cur_is_channel_compression
                        and cur_close < (cur_channel_low - 0.1 * cur_daily_atr_14)
                        and cur_intraday_bar_index >= 30
                        and cur_return_today_open < 0
                ):
                    """空头开仓"""
                    stop_loss = min(cur_channel_high, cur_close + cur_daily_atr_14 * self.stop_loss_multiplier)
                    take_profit = cur_close - cur_daily_atr_14 * self.take_profit_multiplier
                    self.open_position(
                        self.symbol,
                        cur_close,
                        BacktestingDirection.SHORT,
                        open_reason,
                        cur_datetime,
                        cur_trading_date,
                        stop_loss,
                        take_profit,
                        0
                    )
                    after_holding_high_bar_index = i
                    after_holding_low_bar_index = i
            else:
                """平仓检查"""
                if cur_high >= minute_high_arr[after_holding_high_bar_index]:
                    after_holding_high_bar_index = i

                if cur_low <= minute_low_arr[after_holding_low_bar_index]:
                    after_holding_low_bar_index = i

                close_reason = None
                open_trading_date = cur_position.open_trading_date
                holding_trading_days = trading_date_to_index.get(cur_trading_date) - trading_date_to_index.get(
                    open_trading_date)
                if cur_position.direction == BacktestingDirection.LONG:
                    """多头平仓检查"""
                    adjust_stop_loss = -1
                    if cur_trading_date != cur_position.open_trading_date:
                        after_holding_high = minute_high_arr[after_holding_high_bar_index]
                        adjust_stop_loss = max(cur_position.stop_loss, after_holding_high * 0.8)

                    if cur_close <= cur_position.stop_loss:
                        close_reason = '止损'
                    elif (adjust_stop_loss > 0
                          and cur_close < adjust_stop_loss):
                        close_reason = "动态止盈止损"
                    elif (daily_prev_is_true_gap_down_arr[i]
                          and cur_intraday_bar_index >= 30
                          and cur_return_today_open < 0
                    ):
                        close_reason = "跳空反转"
                    elif (prev_is_kline_long_arr[i]
                          and prev_kline_body_ratio_arr[i] >= 0.8
                          and prev_kline_direction_arr[i] == -1
                            # and cur_intraday_bar_index >= 5
                            # and cur_return_today_open < 0
                    ):
                        close_reason = "前一根巨型阴线"
                    # elif cur_close < cur_channel_low:
                    #     close_reason = '价格低于通道低点'
                    elif (
                            daily_allow_close_long_arr[i]
                            and holding_trading_days >= self.min_required_holding_days
                            and not cur_is_channel_compression
                            and daily_trix_neg_days_arr[i] >= 5
                    ):
                        close_reason = 'trix 信号反转平仓'
                    elif i == bar_exec_df_copy_length - 1:
                        close_reason = '最后可交易日'

                    if close_reason:
                        self.close_position(
                            self.symbol,
                            cur_close,
                            cur_datetime,
                            cur_trading_date,
                            close_reason,
                            0
                        )

                elif cur_position.direction == BacktestingDirection.SHORT:
                    """空仓平仓检查"""
                    adjust_stop_loss = -1
                    if cur_trading_date != cur_position.open_trading_date:
                        after_holding_low = minute_low_arr[after_holding_low_bar_index]
                        adjust_stop_loss = min(cur_position.stop_loss, after_holding_low * 1.2)

                    if cur_close >= cur_position.stop_loss:
                        close_reason = '止损'
                    elif (adjust_stop_loss > 0
                          and cur_close >= adjust_stop_loss
                    ):
                        close_reason = '动态止盈止损'
                    elif (daily_prev_is_true_gap_up_arr[i]
                          and cur_intraday_bar_index >= 30
                          and cur_return_today_open > 0
                    ):
                        close_reason = "跳空反转"
                    elif (prev_is_kline_long_arr[i]
                          and prev_kline_body_ratio_arr[i] >= 0.8
                          and prev_kline_direction_arr[i] == 1
                            # and cur_intraday_bar_index >= 5
                            # and cur_return_today_open > 0
                    ):
                        close_reason = "前一根巨型阳线"
                    # elif cur_close > cur_channel_high:
                    #     close_reason = '价格高于通道高点'
                    elif (
                            daily_allow_close_short_arr[i]
                            and holding_trading_days >= self.min_required_holding_days
                            and not cur_is_channel_compression
                            and daily_trix_pos_days_arr[i] >= 5
                    ):
                        close_reason = 'trix 信号反转平仓'
                    elif i == bar_exec_df_copy_length - 1:
                        close_reason = '最后可交易日'

                    if close_reason:
                        self.close_position(
                            self.symbol,
                            cur_close,
                            cur_datetime,
                            cur_trading_date,
                            close_reason,
                            0
                        )

    def create_kline_fig(self) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.8, 0.1],
            vertical_spacing=0.05,
            subplot_titles=["K线", "TRIX 指标"]
        )

        return fig

    def plot_indicator(self,
                       fig: go.Figure,
                       bar_df: DataFrame, ) -> go.Figure:
        # TRIX 指标
        datetime_str = bar_df['trading_date'].dt.strftime(self.datetime_formater)
        fig.add_trace(
            go.Scatter(
                x=datetime_str,
                y=bar_df['trix_hist'],
                mode='lines',
                name='TRIX-HIST',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x}</b><br>TRIX-HIST: %{y:.4f}<extra></extra>'
            ),
            row=2,
            col=1)

        # EMA5
        fig.add_trace(
            go.Scatter(
                x=datetime_str,
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
                x=datetime_str,
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
                x=datetime_str,
                y=bar_df['ema_20'],
                mode='lines',
                name='EMA20',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x}</b><br>EMA20: %{y:.4f}<extra></extra>'
            ),
            row=1,
            col=1)

        return fig


def main():
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
    report_factor_dir = Path(f"{report_root_dir}/{factor_name}")
    report_factor_dir.mkdir(parents=True, exist_ok=True)
    backtesting_overall_stat = BacktestingOverallStatistics(
        factor_name,
        report_factor_dir,
        filename,
        version
    )
    for product in G_BACKTEST_PRODUCT_LIST:
        report_factor_product_dir = report_factor_dir / product
        report_factor_product_dir.mkdir(parents=True, exist_ok=True)
        symbol_bar_1d_df = BarDataManager.load_product_from_cache(product, "1d")
        major_contract_df = ContractTool.arrange_major_contract(symbol_bar_1d_df)
        major_contract_list = major_contract_df["symbol"].unique().tolist()
        major_contract_map = {major_contract: True for major_contract in major_contract_list}
        for symbol, _ in sorted(major_contract_map.items()):
            print(f"----------{symbol}----------")
            # major_contract_start_time = major_contract_df[major_contract_df["symbol"] == symbol]["datetime"].iloc[0]
            # major_contract_end_time = major_contract_df[major_contract_df["symbol"] == symbol]["datetime"].iloc[-1]
            # major_contract_adjusted_end_time = (major_contract_end_time - pd.DateOffset(months=1)).replace(day=20)
            # print(f"{major_contract_start_time} - {major_contract_adjusted_end_time}")

            try:
                bar_1m_df = BarDataManager.load_symbol_from_cache(symbol, "1m")
                if bar_1m_df.empty:
                    print(f"{symbol}-1m dataframe is empty")
                    continue
                # mask = (bar_1m_df['datetime'] >= major_contract_start_time) & (
                #         bar_1m_df['datetime'] <= major_contract_end_time)
                # bar_1m_df = bar_1m_df.loc[mask]
                trix_vpt_research = TrixVptResearch(factor_name,
                                                    symbol,
                                                    bar_period,
                                                    str(report_factor_product_dir),
                                                    initial_capital,
                                                    True, )
                trix_vpt_research.run(bar_1m_df)
                trix_vpt_research.show_backtesting_summary()

                backtesting_overall_stat.add_backtesting_summary(product,
                                                                 trix_vpt_research.get_backtesting_summary())
                backtesting_overall_stat.add_trade_df(product,
                                                      trix_vpt_research.get_trades_df())


            except Exception as e:
                print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")




def test_jq():
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
    report_factor_dir = Path(f"{report_root_dir}/{factor_name}/{version}")
    report_factor_dir.mkdir(parents=True, exist_ok=True)
    backtesting_overall_stat = BacktestingOverallStatistics(
        factor_name,
        report_factor_dir,
        filename,
        version
    )
    for product in G_BACKTEST_PRODUCT_LIST:
        report_factor_product_dir = report_factor_dir / product
        report_factor_product_dir.mkdir(parents=True, exist_ok=True)
        symbol = f"{product}JQ00"

        print(f"----------{symbol}----------")
        try:
            bar_1m_df = BarDataManager.load_symbol_from_cache(symbol, "1m")
            if bar_1m_df.empty:
                print(f"{symbol}-1m dataframe is empty")
                continue

            trix_vpt_research = TrixVptResearch(factor_name,
                                                symbol,
                                                bar_period,
                                                str(report_factor_product_dir),
                                                initial_capital,
                                                True, )
            trix_vpt_research.run(bar_1m_df)
            trix_vpt_research.show_backtesting_summary()
            backtesting_overall_stat.add_backtesting_summary(product,
                                                             trix_vpt_research.get_backtesting_summary())
            backtesting_overall_stat.add_trade_df(product,
                                                  trix_vpt_research.get_trades_df())
        except Exception as e:
            print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")

    # brief_report_file = report_factor_dir / f"brief-summary-data-{version}.txt"
    # with open(brief_report_file, 'w', encoding='utf-8') as f:
    #     f.write(f"策略名称: {filename}\n")
    #     content = '\n'.join(str(d) for d in backtesting_summary_list)
    #     f.write(content)
    backtesting_overall_stat.analyze_summary_to_txt_file()
    backtesting_overall_stat.analyze_summary_to_excel_file()
    backtesting_overall_stat.analyze_trade_to_excel_file()


def test():
    pd.set_option('display.max_columns', None)
    symbol = "MA609"
    product = ContractTool.get_product_by_symbol(symbol)
    contract_config = get_contract_config(product)
    bar_1d_df = BarDataManager.load_symbol_from_cache(symbol, "1d")
    bar_1m_df = BarDataManager.load_symbol_from_cache(symbol, "1m")
    bar_1m_df = KLineTool.add_column_session_to_bar_df(bar_1m_df, contract_config.trading_session_time)
    resampled_bar_1d_df = KLineTool.resample_bar_df(bar_1m_df, "1d")

    resampled_bar_1d_df['datetime'] = pd.to_datetime(resampled_bar_1d_df['datetime']).dt.normalize()
    bar_1d_df['datetime'] = pd.to_datetime(bar_1d_df['datetime']).dt.normalize()

    columns = ["datetime", "open", "high", "low", "close", "volume", "open_interest", "turnover"]
    resampled_bar_1d_df = resampled_bar_1d_df[columns]
    bar_1d_df = bar_1d_df[columns]

    merged_df = pd.merge(
        resampled_bar_1d_df, bar_1d_df,
        on='datetime',
        suffixes=('_resample', '_raw'),
        how='outer'  # 只取两边都有的日期
    )
    print(merged_df)

    merged_df["open_diff"] = merged_df["open_resample"] - merged_df["open_raw"]
    merged_df["high_diff"] = merged_df["high_resample"] - merged_df["high_raw"]
    merged_df["low_diff"] = merged_df["low_resample"] - merged_df["low_raw"]
    merged_df["close_diff"] = merged_df["close_resample"] - merged_df["close_raw"]
    merged_df["volume_diff"] = merged_df["volume_resample"] - merged_df["volume_raw"]
    merged_df["open_interest_diff"] = merged_df["open_resample"] - merged_df["open_raw"]
    merged_df["turnover_diff"] = merged_df["turnover_resample"] - merged_df["turnover_raw"]

    merged_df.to_csv("resample-diff.csv", index=False)


def test2():
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
    symbol = "FG505"
    product = "FG"
    report_factor_dir = Path(f"{report_root_dir}/{factor_name}/{product}")
    report_factor_dir.mkdir(parents=True, exist_ok=True)
    backtesting_overall_stat = BacktestingOverallStatistics(
        factor_name,
        report_factor_dir,
        filename,
        version
    )
    print(f"----------{symbol}----------")
    try:
        bar_1m_df = BarDataManager.load_symbol_from_cache(symbol, "1m")
        trix_vpt_research = TrixVptResearch(factor_name,
                                            symbol,
                                            bar_period,
                                            str(report_factor_dir),
                                            100_0000,
                                            True)
        trix_vpt_research.run(bar_1m_df)
        trix_vpt_research.show_backtesting_summary()
        backtesting_overall_stat.add_backtesting_summary(product,
                                                         trix_vpt_research.get_backtesting_summary())
        backtesting_overall_stat.add_trade_df(product,
                                              trix_vpt_research.get_trades_df())

    except Exception:
        print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")


if __name__ == '__main__':
    test_jq()

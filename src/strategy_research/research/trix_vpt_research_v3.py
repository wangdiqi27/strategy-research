import re
import traceback
from datetime import time
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from strategy_research.config.backtesting import G_BACKTEST_PRODUCT_LIST
from strategy_research.config.exchange import get_contract_config
from strategy_research.factor.momentum import calc_trix
from strategy_research.factor.volatility import calc_atr
from strategy_research.factor.volume_price import calc_vpt
from strategy_research.backtesting.object import BacktestingDirection, BacktestingOverallStatistics, \
    BacktestingTradeStatistics
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
        daily_signal_bar_df = calc_atr(daily_signal_bar_df,
                                       self.atr_period, )
        daily_signal_bar_df = calc_atr(daily_signal_bar_df,
                                       self.channel_lookback)

        daily_signal_bar_df = calc_trix(daily_signal_bar_df,
                                        self.trix_period,
                                        self.trix_ma_period, )
        daily_signal_bar_df = calc_vpt(daily_signal_bar_df,
                                       self.vpt_ma_period, )
        daily_signal_bar_df = daily_signal_bar_df.assign(
            # price
            close_pct=lambda x: x['close'].pct_change(),
            avg_volume=lambda x: x['volume'].rolling(self.vpt_ma_period).mean().shift(1),
            vol_increased=lambda x: x['volume'] > (x['avg_volume'] * self.volume_threshold),
            prev_high=lambda x: x['high'].shift(1),
            prev_low=lambda x: x['low'].shift(1),
            prev_close=lambda x: x['close'].shift(1),
            close_up=lambda x: x['close'] > x['prev_close'],
            close_down=lambda x: x['close'] < x['prev_close'],
            channel_high=lambda x: x["high"].rolling(self.channel_lookback).max(),
            channel_low=lambda x: x["low"].rolling(self.channel_lookback).min(),
            channel_width=lambda x: x['channel_high'] - x['channel_low'],
            channel_compression=lambda x: x["channel_width"] / x["atr_20"],
            channel_compression_20=lambda x: x["channel_compression"].rolling(60).quantile(0.2),
            is_channel_compression=lambda x: x["channel_compression"] <= x["channel_compression_20"],

            cum_ret=lambda x: x["close"].pct_change(self.channel_lookback),
            volatility=lambda x: x["close_pct"].rolling(self.channel_lookback).std(),
            volatility_rank=lambda x: x["volatility"].rolling(60).rank(pct=True),

            # trix
            trix_hist=lambda x: x['trix'] - x['trix_signal'],
            prev_trix_hist=lambda x: x['trix_hist'].shift(1),
            trix_hist_mean=lambda x: x['trix_hist'].rolling(60).mean(),
            trix_hist_std=lambda x: x['trix_hist'].rolling(60).std(),
            trix_hist_q_high=lambda x: x['trix_hist'].rolling(60).quantile(0.75),
            trix_hist_q_low=lambda x: x['trix_hist'].rolling(60).quantile(0.25),
            trix_z_score=lambda x: (x['trix_hist'] - x['trix_hist_mean']) / x['trix_hist_std'],
            trix_hist_up=lambda x: (x['trix_hist'] > x['trix_hist'].shift(1))
                                   & (x['trix_hist'].shift(1) > x['trix_hist'].shift(2)),
            trix_hist_down=lambda x: (x['trix_hist'] < x['trix_hist'].shift(1))
                                     & (x['trix_hist'].shift(1) < x['trix_hist'].shift(2)),
            # vpt
            vpt_hist=lambda x: x['vpt'] - x['vpt_ma'],
            prev_vpt_hist=lambda x: x['vpt_hist'].shift(1),
            vpt_hist_mean=lambda x: x['vpt_hist'].rolling(60).mean(),
            vpt_hist_std=lambda x: x['vpt_hist'].rolling(60).std(),
            vpt_hist_q_high=lambda x: x['vpt_hist'].rolling(60).quantile(0.75),
            vpt_hist_q_low=lambda x: x['vpt_hist'].rolling(60).quantile(0.25),
            vpt_z_score=lambda x: (x['vpt_hist'] - x['vpt_hist_mean']) / x['vpt_hist_std'],
            vpt_hist_up=lambda x: (x["vpt_hist"] > x["vpt_hist"].shift(1))
                                  & (x["vpt_hist"].shift(1) > x["vpt_hist"].shift(2)),
            vpt_hist_down=lambda x: (x["vpt_hist"] < x["vpt_hist"].shift(1))
                                    & (x["vpt_hist"].shift(1) < x["vpt_hist"].shift(2)),
        )

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
            # & (daily_signal_bar_df["trix_hist_up"])
            # & (daily_signal_bar_df["trix_z_score"] > self.trix_z_score_threshold)
        )
        trix_allow_open_short = (
                (daily_signal_bar_df["trix_hist"] < daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] < 0)
            # & (daily_signal_bar_df["trix_hist_down"])
            # & (daily_signal_bar_df["trix_z_score"] < -self.trix_z_score_threshold)
        )

        # 平仓信号
        trix_allow_close_long = (
                (daily_signal_bar_df["trix_hist"] < daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] < 0)
            # & (~daily_signal_bar_df["trix_hist_up"])
            # & (daily_signal_bar_df["trix_z_score"] < 1.5)
        )
        trix_allow_close_short = (
                (daily_signal_bar_df["trix_hist"] > daily_signal_bar_df["prev_trix_hist"])
                & (daily_signal_bar_df["trix_hist"] > 0)
            # & (~daily_signal_bar_df["trix_hist_down"])
            # & (daily_signal_bar_df["trix_z_score"] > 1.5)
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
            tr=lambda x: x[f"tr_{self.atr_period}"].shift(1),
            atr=lambda x: x[f"atr_{self.atr_period}"].shift(1),
            trix=lambda x: x["trix"].shift(1),
            trix_hist=lambda x: x["trix_hist"].shift(1),
            allow_open_long=lambda x: x["allow_open_long"].shift(1),
            allow_open_short=lambda x: x["allow_open_short"].shift(1),
            allow_close_long=lambda x: x["allow_close_long"].shift(1),
            allow_close_short=lambda x: x["allow_close_short"].shift(1),
            prev_close_pct=lambda x: x["close_pct"].shift(1),
            today_open=lambda x: x["open"],
            prev_close=lambda x: x["prev_close"],
            prev_high=lambda x: x["prev_high"],
            prev_low=lambda x: x["prev_low"],
        )
        daily_exec_bar_df = daily_exec_bar_df.add_prefix("daily_")
        daily_exec_bar_df["trading_date"] = daily_signal_bar_df["trading_date"]

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

        minute_exec_bar_df.dropna(inplace=True)
        minute_exec_bar_df.reset_index(drop=True, inplace=True)
        return minute_exec_bar_df, daily_signal_bar_df

    def hybrid_backtest(self,
                        bar_exec_df: DataFrame,
                        verbose: bool = False):
        bar_exec_df_copy = bar_exec_df.copy()

        minute_close_arr = bar_exec_df_copy['close'].to_numpy()
        intraday_return_prev_daily_close_arr = bar_exec_df_copy['intraday_return_prev_daily_close'].to_numpy()
        intraday_return_today_open_arr = bar_exec_df_copy['intraday_return_today_open'].to_numpy()
        minute_intraday_bar_index_arr = bar_exec_df_copy['intraday_bar_index'].to_numpy()

        daily_atr_arr = bar_exec_df_copy['daily_atr'].to_numpy()
        daily_allow_open_long_arr = bar_exec_df_copy['daily_allow_open_long'].to_numpy()
        daily_allow_open_short_arr = bar_exec_df_copy['daily_allow_open_short'].to_numpy()
        daily_allow_close_long_arr = bar_exec_df_copy['daily_allow_close_long'].to_numpy()
        daily_allow_close_short_arr = bar_exec_df_copy['daily_allow_close_short'].to_numpy()
        daily_trix_arr = bar_exec_df_copy['daily_trix'].to_numpy()
        daily_trix_hist_arr = bar_exec_df_copy['daily_trix_hist'].to_numpy()
        datetime_arr = bar_exec_df_copy['datetime'].astype(object).to_numpy()
        trading_date_arr = bar_exec_df_copy['trading_date'].to_numpy()

        trading_date_unique_list = sorted(bar_exec_df_copy['trading_date'].unique())
        trading_date_to_index = {
            trading_date: i for i, trading_date in enumerate(trading_date_unique_list)
        }
        bar_exec_df_copy_length = len(bar_exec_df_copy)

        # 上一根Bar的交易日期
        last_trading_date = None

        # 开始逐根 Bar 混合回测
        for i in range(bar_exec_df_copy_length):
            cur_position = self.account.get_position(self.symbol)
            cur_trading_date = trading_date_arr[i]
            cur_datetime = datetime_arr[i]
            cur_price = minute_close_arr[i]
            self.account.update_latest_price(self.symbol,
                                             cur_price)

            cur_return_prev_daily_close = intraday_return_prev_daily_close_arr[i]
            cur_return_today_open = intraday_return_today_open_arr[i]
            cur_intraday_bar_index = minute_intraday_bar_index_arr[i]

            if last_trading_date is None:
                last_trading_date = cur_trading_date

            if last_trading_date != cur_trading_date:
                self.account.record_daily_equity(last_trading_date)

            last_trading_date = cur_trading_date
            if cur_position is None or cur_position.volume == 0:
                """开仓检查"""
                if (
                        daily_allow_open_long_arr[i]
                        and cur_intraday_bar_index >= 30
                        and cur_return_today_open > 0
                ):
                    """多头开仓"""
                    stop_loss = cur_price - daily_atr_arr[i] * self.stop_loss_multiplier
                    take_profit = cur_price + daily_atr_arr[i] * self.take_profit_multiplier
                    self.open_position(
                        self.symbol,
                        cur_price,
                        BacktestingDirection.LONG,
                        cur_datetime,
                        cur_trading_date,
                        stop_loss,
                        take_profit,
                        0
                    )

                elif (
                        daily_allow_open_short_arr[i]
                        and cur_intraday_bar_index >= 30
                        and cur_return_today_open < 0
                ):
                    """空头开仓"""
                    stop_loss = cur_price + daily_atr_arr[i] * self.stop_loss_multiplier
                    take_profit = cur_price - daily_atr_arr[i] * self.take_profit_multiplier
                    self.open_position(
                        self.symbol,
                        cur_price,
                        BacktestingDirection.SHORT,
                        cur_datetime,
                        cur_trading_date,
                        stop_loss,
                        take_profit,
                        0
                    )
            else:
                close_reason = None
                open_trading_date = cur_position.open_trading_date
                holding_trading_days = trading_date_to_index.get(cur_trading_date) - trading_date_to_index.get(
                    open_trading_date)
                if cur_position.direction == BacktestingDirection.LONG:
                    """多头平仓检查"""
                    if cur_price <= cur_position.stop_loss:
                        close_reason = '止损'
                    # elif cur_price >= cur_position.take_profit:
                    #     exit_reason = '止盈'
                    elif (
                            daily_allow_close_long_arr[i]
                            and holding_trading_days >= self.min_required_holding_days
                            # and cur_intraday_bar_index >= 30
                            # and cur_return_today_open <= -0.005
                    ):
                        close_reason = '信号反转平仓'
                    elif i == bar_exec_df_copy_length - 1:
                        close_reason = '最后可交易日'

                    if close_reason:
                        self.close_position(
                            self.symbol,
                            cur_price,
                            cur_datetime,
                            cur_trading_date,
                            close_reason,
                            0
                        )

                elif cur_position.direction == BacktestingDirection.SHORT:
                    """空仓平仓检查"""
                    if cur_price >= cur_position.stop_loss:
                        close_reason = '止损'
                    # elif cur_price <= cur_position.take_profit:
                    #     exit_reason = '止盈'
                    elif (
                            daily_allow_close_short_arr[i]
                            and holding_trading_days >= self.min_required_holding_days
                            # and cur_intraday_bar_index >= 30
                            # and cur_return_today_open >= 0.005
                    ):
                        close_reason = '信号反转平仓'
                    elif i == bar_exec_df_copy_length - 1:
                        close_reason = '最后可交易日'

                    if close_reason:
                        self.close_position(
                            self.symbol,
                            cur_price,
                            cur_datetime,
                            cur_trading_date,
                            close_reason,
                            0
                        )

    def create_kline_fig(self) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.8, 0.1, 0.1],
            vertical_spacing=0.05,
            subplot_titles=["K线", "TRIX 指标", "VPT 指标"]
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
                y=bar_df['trix_z_score'],
                mode='lines',
                name='TRIX-HIST',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x}</b><br>TRIX-HIST: %{y:.4f}<extra></extra>'
            ),
            row=2,
            col=1)

        # vpt 指标
        fig.add_trace(
            go.Scatter(
                x=datetime_str,
                y=bar_df['vpt_z_score'],
                mode='lines',
                name='VPT-HIST',
                line=dict(color='blue', width=1),
                hovertemplate='<b>%{x}</b><br>VPT-HIST: %{y:.4f}<extra></extra>'
            ),
            row=3,
            col=1)

        return fig


def main():
    pd.set_option('display.max_columns', None)

    report_root_dir = "factor-report"
    factor_name = "TRIX-VPT"
    bar_period = "1m"
    initial_capital = 100_0000
    report_factor_dir = Path(f"{report_root_dir}/{factor_name}")
    report_factor_dir.mkdir(parents=True, exist_ok=True)
    for product in G_BACKTEST_PRODUCT_LIST:
        report_factor_product_dir = report_factor_dir / product
        report_factor_product_dir.mkdir(parents=True, exist_ok=True)
        symbol_bar_1d_df = BarDataManager.load_product_from_cache(product, "1d")

        product_overall_statistics: BacktestingOverallStatistics = BacktestingOverallStatistics(
            factor_name,
            str(report_factor_product_dir),
        )
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

                product_overall_statistics.add_backtesting_summary(product,
                                                                   trix_vpt_research.get_trade_statistics())
                product_overall_statistics.add_bar_df(product,
                                                      trix_vpt_research.signal_bar_df)

            except Exception as e:
                print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")

        product_overall_statistics.analyze_summary_to_excel_file(product,
                                                                 ["trix"])


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
    brief_report_data_list: list[BacktestingTradeStatistics] = []
    for product in G_BACKTEST_PRODUCT_LIST:
        report_factor_product_dir = report_factor_dir / product
        report_factor_product_dir.mkdir(parents=True, exist_ok=True)
        symbol = f"{product}JQ00"

        product_overall_statistics: BacktestingOverallStatistics = BacktestingOverallStatistics(
            factor_name,
            str(report_factor_product_dir),
        )
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
            brief_report_data_list.append(trix_vpt_research.get_trade_statistics())

            product_overall_statistics.add_backtesting_summary(product,
                                                               trix_vpt_research.get_trade_statistics())
            product_overall_statistics.add_bar_df(product,
                                                  trix_vpt_research.signal_bar_df)

        except Exception as e:
            print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")

        product_overall_statistics.analyze_summary_to_excel_file(product,
                                                                 ["trix"])

    brief_report_file = report_factor_dir / f"brief-summary-data-{version}.txt"
    with open(brief_report_file, 'w', encoding='utf-8') as f:
        f.write(f"策略名称: {filename}\n")
        content = '\n'.join(str(d) for d in brief_report_data_list)
        f.write(content)


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
    report_root_dir = "factor-report"
    factor_name = "TRIX-VPT"
    bar_period = "1m"
    symbol = "FG505"
    product = "FG"
    report_factor_path = Path(f"{report_root_dir}/{factor_name}/{product}")
    report_factor_path.mkdir(parents=True, exist_ok=True)
    print(f"----------{symbol}----------")
    try:
        bar_1m_df = BarDataManager.load_symbol_from_cache(symbol, "1m")
        trix_vpt_research = TrixVptResearch(factor_name,
                                            symbol,
                                            bar_period,
                                            str(report_factor_path),
                                            100_0000,
                                            True)
        trix_vpt_research.run(bar_1m_df)
        trix_vpt_research.show_backtesting_summary()
        product_overall_statistics: BacktestingOverallStatistics = BacktestingOverallStatistics(
            factor_name,
            str(report_factor_path)
        )
        product_overall_statistics.add_backtesting_summary(product,
                                                           trix_vpt_research.get_trade_statistics())
        product_overall_statistics.add_bar_df(product,
                                              trix_vpt_research.signal_bar_df)
        product_overall_statistics.analyze_summary_to_excel_file(product,
                                                                 ["trix"])

    except Exception:
        print(f"symbol: {symbol}, Exception: {traceback.format_exc()}")


if __name__ == '__main__':
    test_jq()

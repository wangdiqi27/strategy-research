from abc import ABC, abstractmethod
from datetime import time, datetime, date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame, Series
from plotly.io import to_html
from plotly.subplots import make_subplots

from strategy_research.backtesting.object import BacktestingAccount, BacktestingTradeStatistics, BacktestingDirection, \
    BacktestingPosition, BacktestingSummary, BacktestingAllowTradeDirection
from strategy_research.config.exchange import get_contract_config
from strategy_research.tool.contract import ContractTool
from strategy_research.tool.data import BarDataManager


class PandasBacktestingBase(ABC):

    def __init__(self,
                 factor_name: str,
                 version: str,
                 symbol: str,
                 bar_period: str,
                 report_dir: str,
                 initial_capital: float,
                 bar_df: DataFrame,
                 enable_fig_daily_mode: bool = False,
                 enable_debug_mode: bool = False,
                 enable_jq_index_mode: bool = False,
                 allow_trade_direction: BacktestingAllowTradeDirection = BacktestingAllowTradeDirection.ALL,
                 ):
        self.factor_name = factor_name
        self.version = version
        self.symbol = symbol
        self.bar_period = bar_period
        self.report_dir = report_dir
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.bar_df = bar_df
        self.datetime_formater = '%Y%m%d %H:%M'
        self.enable_fig_daily_mode = enable_fig_daily_mode
        self.enable_debug_mode = enable_debug_mode
        self.enable_jq_index_mode = enable_jq_index_mode
        self.allow_trade_direction = allow_trade_direction

        self.bar_start_datetime: datetime | None = None
        self.bar_end_datetime: datetime | None = None
        self.trading_start_datetime: datetime | None = None
        self.trading_end_datetime: datetime | None = None

        self.product = ContractTool.get_product_by_symbol(self.symbol)
        self.contract_config = get_contract_config(self.product)
        self.price_tick = self.contract_config.price_tick
        self.slippage = self.contract_config.slippage
        self.multiplier = self.contract_config.multiplier
        self.margin_rate = 1
        self.commission_rate = 0

        self._signal_bar_df: DataFrame = pd.DataFrame()
        self._exec_bar_df: DataFrame = pd.DataFrame()

        self._date_top_n_major_bar_data_1d_dict: dict[date, dict] = {}
        self._symbol_bar_data_1m_dict: dict[str, dict[datetime, dict]] = {}
        self._symbol_list: list[str] = []
        self._symbol_index_dict: dict[str, int] = {}

        self.account: BacktestingAccount = BacktestingAccount(self.initial_capital,
                                                              self.symbol)

    def _get_major_contract_symbol(self,
                                   bar_datetime: datetime) -> str | None:
        top_n_major_contract = self._date_top_n_major_bar_data_1d_dict.get(bar_datetime.date(), None)
        if top_n_major_contract is None:
            return None

        return top_n_major_contract["rank1_symbol"]

    def _get_secondary_major_contract_symbol(self,
                                             bar_datetime: datetime) -> str | None:
        top_n_major_contract = self._date_top_n_major_bar_data_1d_dict.get(bar_datetime.date(), None)
        if top_n_major_contract is None:
            return None

        return top_n_major_contract["rank2_symbol"]

    def _get_major_contract_symbol_and_bar(self,
                                           bar_datetime: datetime) -> tuple[str | None, dict | None]:
        major_symbol = self._get_major_contract_symbol(bar_datetime)
        if major_symbol is None:
            return None, None

        major_bar_data = self._get_contract_bar(major_symbol, bar_datetime)

        return major_symbol, major_bar_data

    def _get_secondary_major_contract_symbol_and_bar(self,
                                                     bar_datetime: datetime) -> tuple[str | None, dict | None]:
        secondary_major_symbol = self._get_secondary_major_contract_symbol(bar_datetime)
        if secondary_major_symbol is None:
            return None, None

        secondary_major_bar_data = self._get_contract_bar(secondary_major_symbol, bar_datetime)

        return secondary_major_symbol, secondary_major_bar_data

    def _get_contract_bar(self,
                          symbol: str,
                          bar_datetime: datetime) -> dict | None:
        datetime_bar_data_1m_dict = self._symbol_bar_data_1m_dict[symbol]
        bar_data = datetime_bar_data_1m_dict.get(bar_datetime, None)
        if bar_data is None:
            return None

        return bar_data

    def _get_last_trading_day_by_symbol(self,
                                        symbol: str) -> datetime:
        datetime_bar_data_1m_dict = self._symbol_bar_data_1m_dict[symbol]
        last_trading_day = next(iter(datetime_bar_data_1m_dict.values()))["last_trading_day"]

        return last_trading_day

    def _process_major_contract(self):
        top_n_major_contract_df, symbol_top_n_major_bar_1m_df_dict = BarDataManager.load_product_top_n_major_from_cache(
            self.product, )
        top_n_major_contract_df["date"] = pd.to_datetime(top_n_major_contract_df["datetime"]).dt.date
        top_n_major_contract_df = top_n_major_contract_df.drop(columns=['datetime'])
        date_top_n_major_contract_dict = top_n_major_contract_df.set_index('date').to_dict('index')
        self._date_top_n_major_bar_data_1d_dict = date_top_n_major_contract_dict

        major_symbol_list = sorted(symbol_top_n_major_bar_1m_df_dict.keys())
        symbol_index_dict = {}
        for i in range(len(major_symbol_list)):
            symbol_index_dict[major_symbol_list[i]] = i

        self._symbol_list = major_symbol_list
        self._symbol_index_dict = symbol_index_dict

        for symbol in symbol_top_n_major_bar_1m_df_dict:
            df = symbol_top_n_major_bar_1m_df_dict[symbol]
            datetime_bar_data_dict = df.set_index('datetime').to_dict('index')
            self._symbol_bar_data_1m_dict[symbol] = datetime_bar_data_dict

    def _is_in_trading_time(self,
                            bar_datetime: datetime) -> bool:
        trading_session_time_list = self.contract_config.trading_session_time
        bar_time = bar_datetime.time()
        for session_time in trading_session_time_list:
            if session_time[0] <= bar_time <= session_time[1]:
                return True

        return False

    def _need_switch_contract(self,
                              bar_datetime: datetime,
                              position: BacktestingPosition) -> bool:
        # 移仓换月
        position_volume = position.volume
        position_symbol = position.symbol
        position_bar_data = self._get_contract_bar(position_symbol, bar_datetime)
        position_last_trading_day = position_bar_data['last_trading_day']
        position_oi = position_bar_data["open_interest"]
        major_symbol = self._get_major_contract_symbol(bar_datetime)
        major_bar_data = self._get_contract_bar(major_symbol, bar_datetime)
        major_last_trading_day = major_bar_data['last_trading_day']
        secondary_major_symbol = self._get_secondary_major_contract_symbol(bar_datetime)
        if (major_symbol != position_symbol
                and major_last_trading_day > position_last_trading_day
                and ((position_last_trading_day - bar_datetime) < timedelta(days=20) or (
                        (position_volume / position_oi) >= (1 / 1000)))):
            # 主力合约不是持仓合约，且离最后可交易日小于等于15天，或者仓位/持仓量 > 万分之一，需要移仓换月
            print(f"bar_datetime: {bar_datetime}, "
                  f"position_symbol: {position_symbol}, "
                  f"major_symbol: {major_symbol}, "
                  f"secondary_major_symbol: {secondary_major_symbol},"
                  f"less than 20: {(position_last_trading_day - bar_datetime) < timedelta(days=20)},"
                  f"great than 1/10000: {(position_volume / position_oi) >= (1 / 1000)}, "
                  f"need switch_contract")
            return True

        return False

    def _get_switch_contract_data(self,
                                  candidate_switch_contract_symbol: str,
                                  candidate_switch_contract_bar: dict,
                                  cur_datetime: datetime,
                                  delta_days: int = 60) -> tuple[str, float]:
        switch_contract_symbol = candidate_switch_contract_symbol
        switch_contract_close = candidate_switch_contract_bar["close"]
        if (candidate_switch_contract_bar["last_trading_day"] - cur_datetime) < timedelta(days=delta_days):
            symbol_index = self._symbol_index_dict[candidate_switch_contract_symbol]
            symbol_index += 1
            if symbol_index != len(self._symbol_list):
                switch_contract_symbol = self._symbol_list[symbol_index + 1]
                switch_contract_bar = self._get_contract_bar(switch_contract_symbol, cur_datetime)
                switch_contract_close = switch_contract_bar["close"]

                print(f"candidate_contract_symbol: {candidate_switch_contract_symbol}, "
                      f"candidate_switch_contract_bar.last_trading_day: {candidate_switch_contract_bar["last_trading_day"]}, "
                      f"cur_datetime: {cur_datetime}, "
                      f"switch_contract_symbol: {switch_contract_symbol}, ")

        return switch_contract_symbol, switch_contract_close

    def run(self):
        if self.enable_jq_index_mode:
            print(f"{'-' * 10} start loading major contract {'-' * 10}")
            self._process_major_contract()
            print(f"{'-' * 10} stop loading major contract {'-' * 10}")

        bar_df = self.bar_df
        print(f"{'-' * 10} start computing indicators {'-' * 10}")
        exec_bar_df, signal_bar_df = self.compute_indicators(bar_df, self.contract_config.trading_session_time)
        self.trading_start_datetime = exec_bar_df['trading_date'].iloc[0]
        self.trading_end_datetime = exec_bar_df['trading_date'].iloc[-1]
        self.bar_start_datetime = signal_bar_df["trading_date"].iloc[0]
        self.bar_end_datetime = signal_bar_df["trading_date"].iloc[-1]

        print(f"{'-' * 10} complete computing indicators {'-' * 10}")
        self._exec_bar_df = exec_bar_df
        self._signal_bar_df = signal_bar_df

        if self.enable_debug_mode:
            print(f"{'-' * 10} start saving debug data {'-' * 10}")
            root_path = Path(f"{self.report_dir}")
            self._exec_bar_df.to_csv(root_path / f"{self.symbol}-exec.csv", index=True)
            self._signal_bar_df.to_csv(root_path / f"{self.symbol}-signal.csv", index=True)
            print(f"{'-' * 10} complete saving debug data {'-' * 10}")

        print(f"{'-' * 10} start hybrid backtest {'-' * 10}")
        self.hybrid_backtest(exec_bar_df)
        print(f"{'-' * 10} complete hybrid backtest {'-' * 10}")
        trade_df = self.account.trade_recorder.get_trades_df()
        print(f"{'-' * 10} start plot  {'-' * 10}")
        kline_fig = self.create_kline_fig()
        self.plot(kline_fig,
                  signal_bar_df,
                  trade_df, )

    @abstractmethod
    def compute_indicators(self,
                           bar_df: DataFrame,
                           session_time_type: list[tuple[time, time]]) -> tuple[DataFrame, DataFrame]:
        pass

    @abstractmethod
    def hybrid_backtest(self,
                        bar_exec_df: DataFrame,
                        verbose: bool = False):
        pass

    def get_trade_statistics(self, ) -> BacktestingTradeStatistics:
        return self.account.trade_recorder.get_statistics()

    def get_trades_df(self) -> DataFrame:
        trade_df = self.account.trade_recorder.get_trades_df()

        return trade_df

    def get_all_daily_equity_as_df(self, ) -> DataFrame:
        daily_equity_df = self.account.daily_equity_recorder.get_all_daily_equity_as_df()

        return daily_equity_df

    def get_period_return_stat(self) -> tuple[list, list]:
        return self.account.daily_equity_recorder.calc_period_return()

    def get_backtesting_summary(self) -> BacktestingSummary:
        backtesting_trade_stat = self.get_trade_statistics()
        daily_equity_statistics_df = self.get_all_daily_equity_as_df()
        max_drawdown = daily_equity_statistics_df["drawdown"].min()
        yearly_return_stat_list, half_yearly_return_stat_list = self.get_period_return_stat()

        backtesting_summary = BacktestingSummary(symbol=self.symbol,
                                                 trade_stats=backtesting_trade_stat,
                                                 max_drawdown=max_drawdown,
                                                 bar_start_date=self.bar_start_datetime,
                                                 bar_end_date=self.bar_end_datetime,
                                                 trading_start_date=self.trading_start_datetime,
                                                 trading_end_date=self.trading_end_datetime,
                                                 yearly_return_stat_list=yearly_return_stat_list,
                                                 half_yearly_return_stat_list=half_yearly_return_stat_list)

        return backtesting_summary

    def show_backtesting_summary(self):
        backtesting_summary = self.get_backtesting_summary()

        print(backtesting_summary)

    @abstractmethod
    def create_kline_fig(self) -> go.Figure:
        pass

    def plot(self,
             kline_fig: go.Figure,
             bar_df: DataFrame,
             trades_df: DataFrame, ):

        if self.enable_fig_daily_mode:
            bar_df = bar_df.copy()
            trades_df = trades_df.copy()
            bar_df["trading_date"] = pd.to_datetime(bar_df["trading_date"])
            if not trades_df.empty:
                trades_df["open_trading_date"] = pd.to_datetime(trades_df["open_trading_date"])
                trades_df["close_trading_date"] = pd.to_datetime(trades_df["close_trading_date"])

        kline_fig = self._plot_bar(kline_fig,
                                   bar_df)
        kline_fig = self._plot_open_close_points(kline_fig,
                                                 bar_df,
                                                 trades_df)
        kline_fig = self.plot_indicator(kline_fig, bar_df)
        kline_fig = self.plot_update_layout(kline_fig)

        equity_fig = self._plot_equity()

        self._plot_to_html(kline_fig,
                           equity_fig,
                           False)

        print(f"{'-' * 10} complete plot  {'-' * 10}")

    def _plot_to_html(self,
                      kline_fig: go.Figure,
                      equity_fig: go.Figure | None,
                      single_file_mode: bool = True):
        root_path = Path(f"{self.report_dir}")
        root_path.mkdir(parents=True, exist_ok=True)
        kline_file_name = f"{self.symbol}-{self.factor_name}-{self.bar_period}-kline.html"
        equity_file_name = f"{self.symbol}-{self.factor_name}-{self.bar_period}-equity.html"
        if equity_fig is None:
            kline_fig.write_html(root_path / kline_file_name)
            return

        if single_file_mode:
            single_file_name = f"{self.symbol}-{self.factor_name}-{self.bar_period}-kline-equity.html"

            # 1. 将图表转换为 HTML 片段
            # full_html=False 表示只生成图表的 div 和 script，不包含完整的 html 骨架
            # include_plotlyjs='cdn' 表示通过 CDN 引入 plotly.js (只需第一个图表加载即可，避免文件过大)
            html_fig1 = to_html(
                kline_fig,
                full_html=False,
                include_plotlyjs="cdn",
            )

            # 第二个及后续图表，include_plotlyjs 设为 False，因为前面已经加载过库了
            html_fig2 = to_html(
                equity_fig,
                full_html=False,
                include_plotlyjs=False
            )

            # 2. 拼接成一个完整的 HTML 页面
            complete_html = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <title>回测综合分析报告</title>
                <style>
                    body {{ 
                        margin: 0; 
                        padding: 20px; 
                        font-family: Arial, sans-serif; 
                        background-color: #f9f9f9; 
                    }}
                    h1 {{ 
                        text-align: center; 
                        color: #333; 
                        border-bottom: 2px solid #ddd; 
                        padding-bottom: 10px;
                        margin-top: 40px;
                    }}
                </style>
            </head>
            <body>
                <h1>K线与技术指标分析</h1>
                {html_fig1}
    
                <h1>资金权益曲线</h1>
                {html_fig2}
            </body>
            </html>
            """

            output_path = root_path / single_file_name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(complete_html)
        else:
            kline_fig.write_html(root_path / kline_file_name)
            equity_fig.write_html(root_path / equity_file_name)

    def _plot_bar(self,
                  fig: go.Figure,
                  bar_df: DataFrame, ) -> go.Figure:
        """
        K 线图
        """
        if self.enable_fig_daily_mode:
            x_axis_field = "trading_date"
        else:
            x_axis_field = "datetime"

        datetime_str = bar_df[x_axis_field].dt.strftime(self.datetime_formater)
        fig.add_trace(
            go.Candlestick(
                x=datetime_str,
                open=bar_df['open'],
                high=bar_df['high'],
                low=bar_df['low'],
                close=bar_df['close'],
                name=f'{self.symbol}-{self.bar_period}',
                increasing_line_color='red',  # 国内习惯：红涨
                decreasing_line_color='green',  # 国内习惯：绿跌
                increasing_fillcolor='red',
                decreasing_fillcolor='green',
                hovertemplate='<b>%{x}</b><br>'
                              'OPEN: %{open}<br>'
                              'HIGH: %{high}<br>'
                              'LOW: %{low}<br>'
                              'CLOSE: %{close}<extra></extra>'
            ),
            row=1,
            col=1)

        return fig

    def _plot_open_close_points(self,
                                fig: go.Figure,
                                bar_df: DataFrame,
                                trades_df: DataFrame) -> go.Figure:
        """
        交易信号图
        """
        if trades_df.empty:
            return fig

        if self.enable_fig_daily_mode:
            open_x_axis_field = "open_trading_date"
            close_x_axis_field = "close_trading_date"
            bar_x_axis_field = "trading_date"
        else:
            open_x_axis_field = "open_time"
            close_x_axis_field = "close_time"
            bar_x_axis_field = "datetime"

        if self.enable_jq_index_mode:
            open_price_field = "open_index_price"
            close_price_field = "close_index_price"
        else:
            open_price_field = "open_price"
            close_price_field = "close_price"

        """ 
        绘制开仓点，在K线上添加黑点，悬浮显示文字
        """
        offset = (bar_df['high'].max() - bar_df['low'].min()) * 0.1
        open_columns = [
            "direction",
            "open_time",
            "open_trading_date",
            "open_price",
            "take_profit",
            "stop_loss",
            "open_index_price",
        ]
        open_points_df = trades_df[open_columns]
        entry_times_str = open_points_df[open_x_axis_field].dt.strftime(self.datetime_formater)
        fig.add_trace(
            go.Scatter(
                x=entry_times_str,
                y=open_points_df[open_price_field],
                mode='markers',
                name='开仓点',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='circle',
                    line=dict(width=1, color='white')  # 加点白边更好看
                ),
                # customdata 传入多个列，按顺序组成一个二维数组
                customdata=open_points_df[['take_profit', 'stop_loss']],
                # 在 hovertemplate 中用 %{customdata[index]} 提取
                hovertemplate='<b>%{x}</b><br>' +
                              '开仓价: %{y:.2f}<br>' +
                              '止盈价: %{customdata[0]}<br>' +
                              '止损价: %{customdata[1]:.2f}<br>' +
                              '<extra></extra>',  # <extra></extra> 用于隐藏右侧的 trace 名称
            ),
            row=1,
            col=1
        )

        # for open_point in open_points_df.itertuples():
        for _, open_point in open_points_df.iterrows():
            is_long = open_point["direction"] == BacktestingDirection.LONG
            color = 'red' if is_long else 'green'
            x_axis_value = open_point[open_x_axis_field]
            entry_bar = bar_df.loc[bar_df[bar_x_axis_field] == x_axis_value]
            entry_dt = entry_bar[bar_x_axis_field].iloc[0].strftime(self.datetime_formater)
            high = entry_bar["high"].iloc[0]
            entry_text = "开"
            y1 = high + offset
            """
            绘制开仓点描述性文字
            """
            fig.add_shape(
                type="line",
                x0=entry_dt, y0=high,
                x1=entry_dt, y1=y1,
                line=dict(color=color, width=1, dash="dot"),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=entry_dt,
                y=y1,
                text=entry_text,
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor=color,
                align="center",
                row=1,
                col=1,
            )

        """
        绘制平仓点，在K线上添加黑点，悬浮显示文字
        """
        exit_times_str = trades_df[close_x_axis_field].dt.strftime(self.datetime_formater)
        fig.add_trace(
            go.Scatter(
                x=exit_times_str,
                y=trades_df[close_price_field],
                mode='markers',
                name='平仓点',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='circle',
                    line=dict(width=1, color='white')  # 加点白边更好看
                ),
                # customdata 传入多个列，按顺序组成一个二维数组
                customdata=trades_df[['close_reason', 'net_profit']],
                # 在 hovertemplate 中用 %{customdata[index]} 提取
                hovertemplate='<b>%{x}</b><br>' +
                              '平仓价: %{y:.2f}<br>' +
                              '平仓原因: %{customdata[0]}<br>' +
                              '盈亏金额: %{customdata[1]:.2f}<br>' +
                              '<extra></extra>',  # <extra></extra> 用于隐藏右侧的 trace 名称
            ),
            row=1,
            col=1
        )

        """
        绘制平仓点描述性文字
        """
        for _, trade in trades_df.iterrows():
            is_long = trade["direction"] == BacktestingDirection.LONG
            color = 'red' if is_long else 'green'
            exit_bar = bar_df.loc[bar_df[bar_x_axis_field] == trade[close_x_axis_field]]
            exit_dt = exit_bar[bar_x_axis_field].iloc[0].strftime(self.datetime_formater)
            low = exit_bar["low"].iloc[0]
            exit_text = "平"
            y1 = low - offset
            fig.add_shape(
                type="line",
                x0=exit_dt,
                y0=low,
                x1=exit_dt,
                y1=y1,
                line=dict(color=color, width=1, dash="dot"),
                row=1,
                col=1
            )
            fig.add_annotation(
                x=exit_dt,
                y=y1,
                text=exit_text,
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor=color,
                align="center",
                row=1,
                col=1
            )

        trade_statistics = self.account.trade_recorder.get_statistics()
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=f"收益率：{trade_statistics.total_net_profit_pct * 100:.2f}%",
            marker=dict(color='rgba(0,0,0,0)')
        ))
        return fig

    def plot_indicator(self,
                       fig: go.Figure,
                       bar_df: DataFrame, ) -> go.Figure:
        return fig

    def plot_update_layout(self, fig: go.Figure, ) -> go.Figure:
        # 更新布局
        fig.update_layout(
            title=f'{self.symbol}-{self.bar_period}-{self.factor_name}-{self.version}',
            xaxis=dict(
                type='category',  # 👈 显式指定为分类轴，彻底杜绝时间空白
                rangeslider=dict(visible=False),  # 关闭默认的滑动条
                matches='x2',
            ),
            xaxis2=dict(
                type='category',
                matches='x',
            ),
            xaxis3=dict(
                type='category',
                matches='x',
            ),
            hovermode='x unified',
            # height=800,
            # template='plotly_white',
            template="plotly_dark",
        )

        return fig

    def _plot_equity(self) -> go.Figure | None:
        daily_equity_df = self.get_all_daily_equity_as_df()
        if daily_equity_df.empty:
            return None

        daily_equity_df["trading_date"] = pd.to_datetime(daily_equity_df["trading_date"])
        trading_date_str = daily_equity_df["trading_date"].dt.strftime(self.datetime_formater)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.4, 0.2],
            vertical_spacing=0.05,
            subplot_titles=["账户权益", "累计收益率", "回撤率"]
        )

        # 绘制账户权益曲线
        fig.add_trace(
            go.Scatter(
                x=trading_date_str,
                y=daily_equity_df['equity'],
                mode='lines',
                name='账户权益',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',  # 填充到底部更好看
                fillcolor='rgba(31, 119, 180, 0.1)',
                hovertemplate='<b>%{x}</b><br>'
                              '<br>账户权益: %{y:,.2f}<br>'
                              '<extra></extra>'
            ),
            row=1, col=1
        )

        # 绘制收益率曲线
        fig.add_trace(
            go.Scatter(
                x=trading_date_str,
                y=daily_equity_df['cumulative_return'] * 100,
                mode='lines',
                name='累计收益率 (%)',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',  # 填充到底部更好看
                fillcolor='rgba(31, 119, 180, 0.1)',
                hovertemplate='<b>%{x}</b><br>'
                              '累计收益率: %{y:,.2f}%<br>'
                              '<extra></extra>'
            ),
            row=2, col=1
        )

        # 绘制动态回撤
        fig.add_trace(
            go.Scatter(
                x=trading_date_str,
                y=daily_equity_df['drawdown'] * 100,
                mode='lines',
                name='最大回撤率 (%)',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',  # 填充到底部更好看
                fillcolor='rgba(31, 119, 180, 0.1)',
                hovertemplate='<b>%{x}</b><br>'
                              '回撤率: %{y:,.2f}%<br>'
                              '<extra></extra>'
            ),
            row=3, col=1
        )

        # 更新全局布局
        fig.update_layout(
            title_text="账户分析",
            xaxis_rangeslider_visible=False,  # 关闭底部的范围滑动条
            height=700,
            template='plotly_white',  # 使用白色清爽主题
            hovermode='x unified'  # 鼠标悬浮时显示同一X轴的所有数据
        )

        # 更新每个子图
        y_min, y_max = self._plot_get_axis_range(daily_equity_df["equity"],
                                                 1)
        fig.update_yaxes(title_text="账户权益",
                         tickformat=",.0f",
                         row=1, col=1,
                         range=[y_min, y_max], )

        y_min, y_max = self._plot_get_axis_range(daily_equity_df["cumulative_return"],
                                                 100)
        fig.update_yaxes(title_text="累计收益率 (%)",
                         row=2, col=1,
                         range=[y_min, y_max], )

        y_min, y_max = self._plot_get_axis_range(daily_equity_df["drawdown"],
                                                 100)
        fig.update_yaxes(title_text="最大回撤 (%)",
                         row=3, col=1,
                         range=[y_min, y_max], )

        return fig

    @staticmethod
    def _plot_get_axis_range(series: Series,
                             value_ratio):
        max_value = series.max() * value_ratio
        min_value = series.min() * value_ratio
        diff = max_value - min_value
        if diff > 0:
            padding = diff * 0.1
            y_min = min_value - padding
            y_max = max_value + padding
        else:
            if min_value > 0:
                y_min = min_value * 0.95
            else:
                y_min = max_value * 1.05

            if max_value > 0:
                y_max = max_value * 1.05
            else:
                y_max = max_value * 0.95

        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0

        return y_min, y_max

    def open_position(self,
                      symbol: str,
                      price: float,
                      direction: BacktestingDirection,
                      open_reason: str,
                      open_time: datetime,
                      open_trading_date: date,
                      stop_loss: float = 0,
                      take_profit: float = 0,
                      volume: float = 0,
                      index_price: float = 0):
        self.account.open_position(
            symbol,
            self.price_tick,
            self.multiplier,
            direction,
            price,
            open_reason,
            open_time,
            open_trading_date,
            stop_loss,
            take_profit,
            volume,
            self.slippage,
            self.margin_rate,
            self.commission_rate,
            index_price,
        )

    def close_position(self,
                       symbol: str,
                       price: float,
                       close_time: datetime,
                       close_trading_date: date,
                       reason: str,
                       volume: float = 0,
                       index_price: float = 0):
        self.account.close_position(
            symbol,
            price,
            close_time,
            close_trading_date,
            reason,
            volume,
            self.slippage,
            index_price,
        )

    def get_position(self, symbol) -> BacktestingPosition | None:
        return self.account.get_position(symbol)

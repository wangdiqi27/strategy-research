from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter
from pandas import DataFrame

from strategy_research.tool.contract import ContractTool

class BacktestingAllowTradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    ALL = "ALL"

class BacktestingDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class BacktestingDailyEquity:
    trading_date: date
    equity: float
    cash: float
    margin_occupied: float
    floating_pnl: float
    realized_profit: float
    commission_cost: float


class BacktestingDailyEquityRecorder:
    def __init__(self,
                 initial_capital: float):
        self.initial_capital = initial_capital
        self.daily_equity_list: list[BacktestingDailyEquity] = []

    def add(self,
            daily_equity: BacktestingDailyEquity):
        self.daily_equity_list.append(daily_equity)

    def calc_period_return(self, ) -> tuple[list, list]:
        df = self.get_all_daily_equity_as_df()
        first_date = df['trading_date'].iloc[0]
        virtual_date = first_date - pd.Timedelta(days=1)
        virtual_row = pd.DataFrame({'trading_date': [virtual_date], 'equity': [self.initial_capital]})
        df = pd.concat([virtual_row, df], ignore_index=True)

        df = df.set_index('trading_date')
        equity = df['equity']
        # ============================================================
        # 1. 自然年收益率 (基于日历 1月1日 - 12月31日)
        # ============================================================
        yearly_data = []
        for year, group in equity.groupby(equity.index.year):
            last_eq = group.iloc[-1]
            if not yearly_data:
                first_eq = equity.iloc[0]  # 取到虚拟行的 initial_capital
            else:
                first_eq = yearly_data[-1]['final-equity']

            ret = (last_eq / first_eq) - 1 if first_eq != 0 else np.nan
            yearly_data.append({
                'period': year,
                'initial-equity': first_eq,
                'final-equity': last_eq,
                'return': ret
            })

        # ============================================================
        # 2. 自然半年收益率 (基于日历 1-6月, 7-12月)
        # ============================================================
        half_labels = equity.index.year.astype(str) + np.where(equity.index.month <= 6, "-上半年", "-下半年")
        half_data = []
        for label, group in equity.groupby(half_labels):
            last_eq = group.iloc[-1]
            if not half_data:
                first_eq = equity.iloc[0]
            else:
                first_eq = half_data[-1]['final-equity']

            ret = (last_eq / first_eq) - 1 if first_eq != 0 else np.nan
            half_data.append({
                'period': label,
                'initial-equity': first_eq,
                'final-equity': last_eq,
                'return': ret,
            })

        return yearly_data, half_data

    def get_all_daily_equity_as_df(self) -> DataFrame:
        if not self.daily_equity_list:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_equity_list)
        df["trading_date"] = pd.to_datetime(df['trading_date'])
        df["daily_return"] = df['equity'].pct_change().fillna(0.0)
        df['cumulative_return'] = (df['equity'] - self.initial_capital) / self.initial_capital
        df['cum_max'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']

        return df


@dataclass
class BacktestingTradeStatistics:
    symbol: str
    initial_capital: float = 0
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0
    profit_loss_ratio: float = 0
    total_net_profit: float = 0
    avg_profit: float = 0
    avg_loss: float = 0

    max_consecutive_loss_count: int = 0
    max_consecutive_loss: float = 0

    long_count: int = 0
    short_count: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_profit_loss_ratio: float = 0.0
    short_profit_loss_ratio: float = 0.0

    total_net_profit_pct: float = field(init=False)
    max_consecutive_loss_pct: float = field(init=False)

    def __post_init__(self):
        if self.initial_capital == 0:
            self.total_net_profit_pct = 0
            self.max_consecutive_loss_pct = 0
        else:
            self.total_net_profit_pct = self.total_net_profit / self.initial_capital
            self.max_consecutive_loss_pct = self.max_consecutive_loss / self.initial_capital

    def __str__(self) -> str:
        # 金额格式化（千分位、保留2位小数）
        fmt_m = lambda x: f"{x:,.2f}" if x is not None else "0.00"
        # 百分比格式化
        fmt_p = lambda x: f"{x:.2f}%" if x is not None else "0.00%"

        report = [
            "【账户资金概况】",
            f"  初始资金 (Initial Capital) :  {fmt_m(self.initial_capital)} 元",
            f"  累计净利润 (Net Profit)    :  {fmt_m(self.total_net_profit)} 元 ({fmt_p(self.total_net_profit_pct * 100)})",
            "",
            "【交易次数与胜率】",
            f"  总交易次数 (Total Trades)  :  {self.total_trades} 次",
            f"  盈利交易数 (Win Count)     :  {self.win_count} 次",
            f"  亏损交易数 (Loss Count)    :  {self.loss_count} 次",
            f"  交易胜率 (Win Rate)        :  {fmt_p(self.win_rate * 100)}",
            "",
            "【盈亏与风险指标】",
            f"  盈亏比 (P&L Ratio)         :  {self.profit_loss_ratio:.2f}",
            f"  单次均盈 (Avg Profit)      :  {fmt_m(self.avg_profit)} 元",
            f"  单次均亏 (Avg Loss)        :  {fmt_m(self.avg_loss)} 元",
            f"  连续亏损次数 (MCL Count)     :  {self.max_consecutive_loss_count}",
            # f"  连续亏损金额 (MCL Loss)     :  {self.max_consecutive_loss} 元 ({fmt_p(self.max_consecutive_loss_pct * 100)})",
            "",
            "【多空盈亏与风险指标】",
            f"  做多总交易次数 (Total Long Trades)   :  {self.long_count}",
            f"  做多交易胜率 (Win Rate)             :  {fmt_p(self.long_win_rate * 100)}",
            f"  做多盈亏比 (P&L Ratio)              :  {self.long_profit_loss_ratio:.2f}",
            "",
            f"  做空总交易次数 (Total Long Trades)         :  {self.short_count}",
            f"  做空交易胜率 (Win Rate)                    :  {fmt_p(self.short_win_rate * 100)}",
            f"  做空盈亏比 (P&L Ratio)                     :  {self.short_profit_loss_ratio:.2f}",
            "",
        ]
        return "\n".join(report)


@dataclass
class BacktestingTrade:
    symbol: str
    direction: BacktestingDirection
    open_time: datetime
    open_trading_date: date
    close_time: datetime
    close_trading_date: date
    open_price: float
    open_reason: str
    close_price: float
    close_reason: str
    volume: float
    net_profit: float
    gross_profit: float
    take_profit: float = 0
    stop_loss: float = 0
    open_index_price: float = 0
    close_index_price: float = 0


class BacktestingTradeRecorder:
    def __init__(self,
                 initial_capital: float,
                 symbol: str, ):
        self.initial_capital: float = initial_capital
        self.trades: list[BacktestingTrade] = []
        self.symbol: str = symbol

    def add_trade(self, trade: BacktestingTrade):
        self.trades.append(trade)

    def get_trades_df(self) -> DataFrame:
        return pd.DataFrame(self.trades)

    def get_statistics(self) -> BacktestingTradeStatistics:
        if not self.trades:
            return BacktestingTradeStatistics(self.symbol)

        trades = self.trades
        total_trades = len(trades)
        win_trades = [t for t in trades if t.net_profit > 0]
        loss_trades = [t for t in trades if t.net_profit < 0]
        total_win_num = len(win_trades)
        total_loss_num = len(loss_trades)
        # 胜率
        win_rate = total_win_num / total_trades if total_trades > 0 else 0
        # 总盈亏
        total_profit = sum(t.net_profit for t in trades)
        # 平均盈利 / 平均亏损
        avg_win = sum(t.net_profit for t in win_trades) / total_win_num if total_win_num > 0 else 0
        avg_loss = abs(sum(t.net_profit for t in loss_trades) / total_loss_num) if total_loss_num > 0 else 0

        # 盈亏比
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        max_consecutive_loss_count = 0
        max_consecutive_loss = 0

        cur_consecutive_loss_count = 0
        cur_consecutive_loss = 0

        for trade in trades:
            if trade.net_profit < 0:
                cur_consecutive_loss_count += 1
                cur_consecutive_loss += -trade.net_profit
            else:
                cur_consecutive_loss_count = 0
                cur_consecutive_loss = 0

            max_consecutive_loss_count = max(max_consecutive_loss_count, cur_consecutive_loss_count)
            max_consecutive_loss = max(max_consecutive_loss, cur_consecutive_loss)

        # 单独统计做多和做空情况

        ## 做多统计
        long_trades = [t for t in trades if t.direction == BacktestingDirection.LONG]
        long_count = len(long_trades)
        long_wins = [t for t in long_trades if t.net_profit > 0]
        long_losses = [t for t in long_trades if t.net_profit < 0]
        long_win_num = len(long_wins)
        long_loss_num = len(long_losses)
        long_win_rate = long_win_num / long_count if long_count > 0 else 0

        avg_long_win = sum(t.net_profit for t in long_wins) / long_win_num if long_win_num > 0 else 0
        avg_long_loss = abs(sum(t.net_profit for t in long_losses) / long_loss_num) if long_loss_num > 0 else 0
        long_profit_loss_ratio = avg_long_win / avg_long_loss if avg_long_loss > 0 else float('inf')

        ## 做空统计
        short_trades = [t for t in trades if t.direction == BacktestingDirection.SHORT]
        short_count = len(short_trades)
        short_wins = [t for t in short_trades if t.net_profit > 0]
        short_losses = [t for t in short_trades if t.net_profit < 0]
        short_win_num = len(short_wins)
        short_loss_num = len(short_losses)
        short_win_rate = short_win_num / short_count if short_count > 0 else 0

        avg_short_win = sum(t.net_profit for t in short_wins) / short_win_num if short_win_num > 0 else 0
        avg_short_loss = abs(sum(t.net_profit for t in short_losses) / short_loss_num) if short_loss_num > 0 else 0
        short_profit_loss_ratio = avg_short_win / avg_short_loss if avg_short_loss > 0 else float('inf')

        return BacktestingTradeStatistics(
            symbol=self.symbol,
            initial_capital=self.initial_capital,
            total_trades=total_trades,
            win_count=total_win_num,
            loss_count=total_loss_num,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_net_profit=total_profit,
            avg_profit=avg_win,
            avg_loss=avg_loss,
            max_consecutive_loss_count=max_consecutive_loss_count,
            max_consecutive_loss=max_consecutive_loss,
            long_count=long_count,
            short_count=short_count,
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            long_profit_loss_ratio=long_profit_loss_ratio,
            short_profit_loss_ratio=short_profit_loss_ratio,
        )

    def show_statistics(self):
        statistics = self.get_statistics()
        print(statistics)


class BacktestingPosition:

    def __init__(self,
                 symbol: str,
                 contract_multiplier: float,
                 price_tick: float,
                 volume: float,
                 open_avg_price: float,
                 open_reason: str,
                 direction: BacktestingDirection,
                 margin_rate: float,
                 commission_rate: float,
                 open_time: datetime,
                 open_trading_date: date,
                 open_commission: float,
                 stop_loss: float = 0,
                 take_profit: float = 0,
                 open_index_price: float = 0, ):
        self.symbol: str = symbol
        self.contract_multiplier: float = contract_multiplier
        self.price_tick: float = price_tick
        self.volume: float = volume
        self.open_avg_price: float = open_avg_price
        self.open_reason: str = open_reason
        self.latest_price: float = open_avg_price
        self.direction: BacktestingDirection = direction
        self.margin_rate: float = margin_rate
        self.commission_rate: float = commission_rate
        self.open_time: datetime = open_time
        self.open_trading_date: date = open_trading_date
        self.open_commission: float = open_commission
        self.stop_loss: float = stop_loss
        self.take_profit: float = take_profit
        self.open_index_price: float = open_index_price

    @property
    def position_value(self):
        return self.latest_price * self.contract_multiplier * self.volume

    @property
    def margin_occupied(self):
        return self.position_value * self.margin_rate

    @property
    def floating_pnl(self):
        """逐笔浮盈"""
        if self.volume == 0 or self.direction == BacktestingDirection.FLAT:
            return 0

        if self.direction == BacktestingDirection.LONG:
            pnl = (self.latest_price - self.open_avg_price) * self.volume * self.contract_multiplier
        else:
            pnl = (self.open_avg_price - self.latest_price) * self.volume * self.contract_multiplier

        return pnl

    def update_latest_price(self,
                            price):
        self.latest_price = price

    @property
    def profit_loss_point(self):
        if self.volume == 0 or self.direction == BacktestingDirection.FLAT:
            return 0

        if self.direction == BacktestingDirection.LONG:
            point = self.latest_price - self.open_avg_price
        else:
            point = self.open_avg_price - self.latest_price

        return point


class BacktestingAccount:
    def __init__(self,
                 initial_capital: float,
                 symbol: str):
        self.initial_capital: float = initial_capital
        self.symbol: str = symbol
        self.positions: dict[str, BacktestingPosition] = {}
        self.trade_recorder: BacktestingTradeRecorder = BacktestingTradeRecorder(initial_capital,
                                                                                 self.symbol)
        self.daily_equity_recorder: BacktestingDailyEquityRecorder = BacktestingDailyEquityRecorder(
            self.initial_capital)
        self._realized_profit: float = 0
        self._commission_cost: float = 0

    @property
    def margin_occupied(self):
        return sum(p.margin_occupied for p in self.positions.values())

    @property
    def floating_pnl(self):
        return sum(
            p.floating_pnl for p in self.positions.values()
        )

    @property
    def equity(self):
        base_equity = self.initial_capital + self._realized_profit
        base_equity += self.floating_pnl
        base_equity -= self._commission_cost

        return base_equity

    @property
    def cash(self):
        return self.equity - self.margin_occupied

    @property
    def margin_usage(self):
        if self.equity <= 0:
            return 1

        return self.margin_occupied / self.equity

    @property
    def realized_profit(self):
        return self._realized_profit

    @property
    def commission_cost(self):
        return self._commission_cost

    def open_position(self,
                      symbol: str,
                      price_tick: float,
                      contract_multiplier: float,
                      direction: BacktestingDirection,
                      price: float,
                      open_reason: str,
                      open_time: datetime,
                      open_trading_date: date,
                      stop_loss: float = 0,
                      take_profit: float = 0,
                      volume: float = 0,
                      slippage: float = 1,
                      margin_rate: float = 1,
                      commission_rate: float = 0,
                      index_price: float = 0.0,
                      ):
        position = self.get_position(symbol, )
        if position is not None:
            print(f"not support, should close positions before reopen position, symbol: {symbol}")
            return

        if direction == BacktestingDirection.LONG:
            adjusted_open_price = price + slippage * price_tick
        elif direction == BacktestingDirection.SHORT:
            adjusted_open_price = price - slippage * price_tick
        else:
            print(f"{direction} not supported")
            return

        contract_value_unit = adjusted_open_price * contract_multiplier
        if volume == 0:
            # 按最大可开手数开仓
            volume = self.cash // (contract_value_unit * margin_rate)

        if volume == 0:
            print(f"volume {volume} can't open")
            return

        if commission_rate == 0:
            # 没有指定手续费，按1跳计算
            commission = 1 * price_tick * volume
        else:
            commission = contract_value_unit * volume * commission_rate

        self._commission_cost += commission
        self.positions[symbol] = BacktestingPosition(symbol,
                                                     contract_multiplier,
                                                     price_tick,
                                                     volume,
                                                     adjusted_open_price,
                                                     open_reason,
                                                     direction,
                                                     margin_rate,
                                                     commission_rate,
                                                     open_time,
                                                     open_trading_date,
                                                     commission,
                                                     stop_loss,
                                                     take_profit,
                                                     index_price)

    def close_position(self,
                       symbol: str,
                       price: float,
                       close_time: datetime,
                       close_trading_date: date,
                       reason: str,
                       volume: float = 0,
                       slippage: float = 1,
                       index_price: float = 0.0, ):
        position = self.get_position(symbol, )
        if position is None:
            print(f"can't find position, symbol: {symbol}")
            return

        price_tick = position.price_tick
        contract_multiplier = position.contract_multiplier
        commission_rate = position.commission_rate

        if volume == 0:
            volume = position.volume

        if volume == 0:
            print(f"close volume = 0, symbol: {symbol}")
            return

        direction = position.direction
        if direction == BacktestingDirection.LONG:
            adjusted_close_price = price - slippage * price_tick
            realized_pnl = (adjusted_close_price - position.open_avg_price) * contract_multiplier * position.volume
        elif direction == BacktestingDirection.SHORT:
            adjusted_close_price = price + slippage * price_tick
            realized_pnl = (position.open_avg_price - adjusted_close_price) * contract_multiplier * position.volume
        else:
            raise Exception(f"{direction} not supported")

        self._realized_profit += realized_pnl
        contract_value_unit = adjusted_close_price * contract_multiplier
        if commission_rate == 0:
            # 没有指定手续费，按1跳计算
            commission = 1 * price_tick * volume
        else:
            commission = contract_value_unit * volume * commission_rate

        self._commission_cost += commission

        self.trade_recorder.add_trade(BacktestingTrade(
            symbol=symbol,
            direction=position.direction,
            open_time=position.open_time,
            open_trading_date=position.open_trading_date,
            close_time=close_time,
            close_trading_date=close_trading_date,
            open_price=position.open_avg_price,
            open_reason=position.open_reason,
            close_price=adjusted_close_price,
            close_reason=reason,
            volume=position.volume,
            net_profit=realized_pnl - commission - position.open_commission,
            gross_profit=realized_pnl,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            open_index_price=position.open_index_price,
            close_index_price=index_price,
        ))

        self.positions.pop(symbol, None)

    def get_position(self, symbol: str) -> BacktestingPosition | None:
        return self.positions.get(symbol, None)

    def get_position_v2(self) -> BacktestingPosition | None:
        if len(self.positions) == 0:
            return None

        return next(iter(self.positions.values()))

    def update_latest_price(self,
                            symbol: str,
                            price: float, ):
        position = self.get_position(symbol)
        if position is None:
            return

        position.update_latest_price(price)

    def get_trade_statistics(self) -> BacktestingTradeStatistics:
        return self.trade_recorder.get_statistics()

    def record_daily_equity(self,
                            trading_date: datetime, ):
        self.daily_equity_recorder.add(BacktestingDailyEquity(
            trading_date=trading_date,
            equity=self.equity,
            cash=self.cash,
            margin_occupied=self.margin_occupied,
            floating_pnl=self.floating_pnl,
            realized_profit=self.realized_profit,
            commission_cost=self.commission_cost
        ))

    def get_all_daily_equity_as_df(self) -> DataFrame:
        return self.daily_equity_recorder.get_all_daily_equity_as_df()


@dataclass
class BacktestingSummary:
    symbol: str
    trade_stats: BacktestingTradeStatistics
    yearly_return_stat_list: list
    half_yearly_return_stat_list: list
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    # K 线的开始时间
    bar_start_date: date | None = None
    # K 线的结束时间
    bar_end_date: date | None = None
    # 计算完指标开始交易的时间
    trading_start_date: date | None = None
    # 计算完指标结束交易的时间
    trading_end_date: date | None = None

    def to_dict(self) -> dict:
        summary_dict = asdict(self.trade_stats)
        summary_dict["symbol"] = self.symbol
        summary_dict["max_drawdown"] = self.max_drawdown
        summary_dict["bar_start_date"] = self.bar_start_date
        summary_dict["bar_end_date"] = self.bar_end_date
        summary_dict["trading_start_date"] = self.trading_start_date
        summary_dict["trading_end_date"] = self.trading_end_date

        return summary_dict

    def __str__(self) -> str:
        # 金额格式化（千分位、保留2位小数）
        fmt_m = lambda x: f"{x:,.2f}" if x is not None else "0.00"
        # 百分比格式化
        fmt_p = lambda x: f"{x:.2f}%" if x is not None else "0.00%"

        yearly_stat_str_list = []
        for item in self.yearly_return_stat_list:
            yearly_stat_str_list.extend(
                [
                    f"  年份：{item['period']}",
                    f"  期初权益：{item['initial-equity']}",
                    f"  期末权益：{item['final-equity']}",
                    f"  收益率：{fmt_p(item['return'] * 100)}",
                    ""
                ]
            )

        yearly_stat_str = "\n".join(yearly_stat_str_list)
        half_yearly_stat_str_list = []
        for item in self.half_yearly_return_stat_list:
            half_yearly_stat_str_list.extend(
                [
                    f"  年份：{item['period']}",
                    f"  期初权益：{item['initial-equity']}",
                    f"  期末权益：{item['final-equity']}",
                    f"  收益率：{fmt_p(item['return'] * 100)}",
                    ""
                ]
            )

        half_yearly_stat_str = "\n".join(half_yearly_stat_str_list)
        report = [
            "=" * 60,
            f"             回测交易统计报告 [ 标的: {self.symbol or '未指定'} ]",
            "=" * 60,
            str(self.trade_stats),
            "【最大回撤统计】",
            f"  最大回撤 (Max DrawDown)  :  {fmt_p(self.max_drawdown * 100)}",
            "",
            "【交易起止时间】",
            f"  K线开始时间             :  {self.bar_start_date}",
            f"  K线结束时间             :  {self.bar_end_date}",
            f"  交易开始时间(计算完指标)  :  {self.trading_start_date}",
            f"  交易结束时间(计算完指标)  :  {self.trading_end_date}",
            "",
            "【自然年收益统计】",
            yearly_stat_str,
            "【半个自然年收益统计】",
            half_yearly_stat_str,
            "=" * 60
        ]
        return "\n".join(report)


class BacktestingOverallStatistics:

    def __init__(self,
                 factor_name: str,
                 report_path: Path,
                 strategy_file_name: str,
                 version: str):
        self.factor_name = factor_name
        self.report_path = report_path
        self.strategy_file_name = strategy_file_name
        self.version = version

        # product -> backtesting summary
        self._summary_map: dict[str, list[BacktestingSummary]] = defaultdict(list)
        # product -> backtesting trade
        self._trade_df_map: dict[str, list[DataFrame]] = defaultdict(list)

    def add_backtesting_summary(self,
                                product: str,
                                backtesting_summary: BacktestingSummary):
        self._summary_map[product].append(backtesting_summary)

    def add_trade_df(self,
                     product: str,
                     df: DataFrame):
        df = df.copy()
        self._trade_df_map[product].append(df)

    def analyze_trade_to_excel_file(self,
                                    product: str | None = None):
        if product is None:
            filename = f"all-{self.factor_name}-trades-{self.version}"
            trade_df_list = [item for sublist in self._trade_df_map.values() for item in sublist]
        else:
            trade_df_list = self._trade_df_map[product]
            filename = f"{product}-{self.factor_name}-trades-{self.version}"

        writer = pd.ExcelWriter(self.report_path / f'{filename}.xlsx', engine='xlsxwriter')
        if trade_df_list:
            for trade_df in trade_df_list:
                symbol = trade_df["symbol"].iloc[0]
                product_sheet_name = ContractTool.get_product_by_symbol(symbol)
                columns = [
                    "symbol",
                    "direction",
                    "open_time",
                    "close_time",
                    "open_price",
                    "open_reason",
                    "close_price",
                    "close_reason",
                    "volume",
                    "net_profit",
                    "gross_profit",
                    "take_profit",
                    "stop_loss",
                ]
                filter_trade_df = trade_df[columns]
                filter_trade_df['open_time'] = filter_trade_df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                filter_trade_df['close_time'] = filter_trade_df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                filter_trade_df.to_excel(writer,
                                         sheet_name=product_sheet_name,
                                         engine='xlsxwriter', )
                worksheet = writer.sheets[product_sheet_name]
                worksheet.autofit()

        writer.close()

    def analyze_summary_to_txt_file(self):
        summary_list = [item for sublist in self._summary_map.values() for item in sublist]
        brief_report_file = self.report_path / f"brief-summary-data-{self.version}.txt"
        with open(brief_report_file, 'w', encoding='utf-8') as f:
            f.write(f"策略名称: {self.strategy_file_name}\n")
            content = '\n'.join(str(d) for d in summary_list)
            f.write(content)

    def analyze_summary_to_excel_file(self,
                                      product: str | None = None, ):
        if product is None:
            filename = f"all-{self.factor_name}-stat-{self.version}"
            summary_list = [item for sublist in self._summary_map.values() for item in sublist]
        else:
            summary_list = self._summary_map[product]
            filename = f"{product}-{self.factor_name}-stat-{self.version}"

        if not summary_list:
            return

        writer = pd.ExcelWriter(self.report_path / f'{filename}.xlsx', engine='xlsxwriter')
        if summary_list:
            records = [item.to_dict() for item in summary_list]
            summary_df = pd.json_normalize(records)
            columns = [
                "symbol",
                "total_trades",
                "win_rate",
                "profit_loss_ratio",
                "total_net_profit_pct",
                "total_net_profit",
                "max_consecutive_loss_count",
                # "max_consecutive_loss",
                # "max_consecutive_loss_pct",
                "max_drawdown",
                "bar_start_date",
                "bar_end_date",
                "trading_start_date",
                "trading_end_date",
            ]
            percent_format_columns = [
                "win_rate",
                "total_net_profit_pct",
                # "max_consecutive_loss_pct",
                "max_drawdown",
            ]
            filter_summary_df = summary_df[columns]
            filter_summary_df.to_excel(writer,
                                       sheet_name="summary",
                                       engine='xlsxwriter', )
            workbook = writer.book
            worksheet = writer.sheets['summary']
            percent_format = workbook.add_format({'num_format': '0.00%'})
            for col in percent_format_columns:
                col_idx = filter_summary_df.columns.get_loc(col)
                col_idx += 1
                col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)
                worksheet.set_column(f'{col_letter}:{col_letter}', 12, percent_format)

            worksheet.autofit()

        writer.close()
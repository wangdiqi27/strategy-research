from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

import pandas as pd
import xlsxwriter
from pandas import DataFrame

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

    def get_all_daily_equity_as_df(self) -> DataFrame:
        if not self.daily_equity_list:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_equity_list)

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
            "=" * 60,
            f"             回测交易统计报告 [ 标的: {self.symbol or '未指定'} ]",
            "=" * 60,
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
            "=" * 60
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
    close_price: float
    close_reason: str
    net_profit: float
    gross_profit: float
    take_profit: float = 0
    stop_loss: float = 0


class BacktestingTradeRecorder:
    def __init__(self,
                 initial_capital: float,
                 symbol: str,):
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

        return BacktestingTradeStatistics(
            symbol=trades[0].symbol,
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
                 direction: BacktestingDirection,
                 margin_rate: float,
                 commission_rate: float,
                 open_time: datetime,
                 open_trading_date: date,
                 open_commission: float,
                 stop_loss: float = 0,
                 take_profit: float = 0, ):
        self.symbol: str = symbol
        self.contract_multiplier: float = contract_multiplier
        self.price_tick: float = price_tick
        self.volume: float = volume
        self.open_avg_price: float = open_avg_price
        self.latest_price: float = open_avg_price
        self.direction: BacktestingDirection = direction
        self.margin_rate: float = margin_rate
        self.commission_rate: float = commission_rate
        self.open_time: datetime = open_time
        self.open_trading_date: date = open_trading_date
        self.open_commission: float = open_commission
        self.stop_loss: float = stop_loss
        self.take_profit: float = take_profit

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
                      open_time: datetime,
                      open_trading_date: date,
                      stop_loss: float = 0,
                      take_profit: float = 0,
                      volume: float = 0,
                      slippage: float = 1,
                      margin_rate: float = 1,
                      commission_rate: float = 0,
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
                                                     direction,
                                                     margin_rate,
                                                     commission_rate,
                                                     open_time,
                                                     open_trading_date,
                                                     commission,
                                                     stop_loss,
                                                     take_profit, )

    def close_position(self,
                       symbol: str,
                       price: float,
                       close_time: datetime,
                       close_trading_date: date,
                       reason: str,
                       volume: float = 0,
                       slippage: float = 1, ):
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
            close_price=adjusted_close_price,
            close_reason=reason,
            net_profit=realized_pnl - commission - position.open_commission,
            gross_profit=realized_pnl,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
        ))

        self.positions.pop(symbol, None)

    def get_position(self, symbol: str) -> BacktestingPosition:
        return self.positions.get(symbol, None)

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


class BacktestingProductOverallStatistics:

    def __init__(self,
                 factor_name: str,
                 report_path: str):
        self.factor_name = factor_name
        self.report_path = report_path

        self._trade_statistics: dict[str, list[BacktestingTradeStatistics]] = defaultdict(list)
        self._equity_statistics: dict[str, list[DataFrame]] = defaultdict(list)
        self._bar_df_statistics: dict[str, list[DataFrame]] = defaultdict(list)

    def add_trade_statistics(self,
                             product: str,
                             trade_statistics: BacktestingTradeStatistics):
        self._trade_statistics[product].append(trade_statistics)

    def add_daily_equity_df(self,
                            product: str,
                            daily_equity_df: DataFrame):
        self._equity_statistics[product].append(daily_equity_df)

    def add_bar_df(self,
                   product: str,
                   df: DataFrame):
        self._bar_df_statistics[product].append(df)

    def analyze_product(self,
                        product: str,
                        indicator_fields: list[str], ):
        trade_statistics_list = self._trade_statistics[product]
        bar_indicator_df_list = self._bar_df_statistics[product]
        if not trade_statistics_list and not bar_indicator_df_list:
            return

        writer = pd.ExcelWriter(f'{self.report_path}/{product}-{self.factor_name}-stat.xlsx', engine='xlsxwriter')
        if trade_statistics_list:
            trade_statistics_df = pd.DataFrame(trade_statistics_list)
            columns = [
                "symbol",
                "total_trades",
                "win_rate",
                "profit_loss_ratio",
                "total_net_profit_pct",
                "total_net_profit",
                "max_consecutive_loss_count",
                "max_consecutive_loss",
                "max_consecutive_loss_pct",
            ]
            percent_format_columns = [
                "win_rate",
                "total_net_profit_pct",
                "max_consecutive_loss_pct",

            ]
            trade_stat_summary_df = trade_statistics_df[columns]
            trade_stat_summary_df.to_excel(writer,
                                           sheet_name="trade-statistics",
                                           engine='xlsxwriter',)
            workbook = writer.book
            worksheet = writer.sheets['trade-statistics']
            percent_format = workbook.add_format({'num_format': '0.00%'})
            for col in percent_format_columns:
                col_idx = trade_stat_summary_df.columns.get_loc(col)
                col_idx += 1
                col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)
                worksheet.set_column(f'{col_letter}:{col_letter}', 12, percent_format)

            worksheet.autofit()


        if bar_indicator_df_list:
            for name in indicator_fields:
                stat_list = []
                for bar_indicator_df in bar_indicator_df_list:
                    stat = bar_indicator_df[name].describe()
                    stat_list.append({
                        "symbol": bar_indicator_df["symbol"].iloc[0],
                        "count": stat["count"],
                        "mean": stat["mean"],
                        "std": stat["std"],
                        "min": stat["min"],
                        "max": stat["max"],
                        "25%": stat["25%"],
                        "50%": stat["50%"],
                        "75%": stat["75%"],
                    })

                stat_list_df = pd.DataFrame(stat_list)
                stat_list_df.to_excel(writer, sheet_name=name, engine='xlsxwriter')
                worksheet = writer.sheets[name]
                worksheet.autofit()

        writer.close()

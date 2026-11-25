from pathlib import Path
from zoneinfo import ZoneInfo

from vnpy.trader.constant import Interval, Exchange
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
from pandas import DataFrame

from vnpy.trader.database import get_database
from vnpy_data_ext.database import PostgresqlDatabase

from strategy_research.tool.contract import ContractTool


class BarDataManager:
    BAR_DATA_CACHE_ROOT_DIR: Path = Path.home() / "workspace" / "trading-data" / "bar-data-cache"

    _SYMBOL_BAR_1D_DF_CACHE: dict[str, DataFrame] = {}
    _SYMBOL_BAR_1M_DF_CACHE: dict[str, DataFrame] = {}

    _PRODUCT_YEAR_1D_DF_CACHE: dict[str, DataFrame] = {}

    @classmethod
    def _get_data_cache_1m_df_dir(cls) -> Path:
        return cls.BAR_DATA_CACHE_ROOT_DIR / "1m-df"

    @classmethod
    def _get_data_cache_1d_df_dir(cls) -> Path:
        return cls.BAR_DATA_CACHE_ROOT_DIR / "1d-df"

    @classmethod
    def _get_data_cache_1d_excel_dir(cls) -> Path:
        return cls.BAR_DATA_CACHE_ROOT_DIR / "1d-excel"

    @classmethod
    def _get_data_cache_1m_spread_df_dir(cls) -> Path:
        return cls.BAR_DATA_CACHE_ROOT_DIR / "1m-index-spread"

    @staticmethod
    def _load_bar_data_as_df_from_database(symbol: str,
                                           interval: Interval,
                                           start_time: datetime,
                                           end_time: datetime, ) -> DataFrame:
        """
        从数据库下载合约分钟级K线数据，以 DataFrame 返回

        :param symbol:
        :param interval:
        :param start_time:
        :param end_time:
        :return:
        """
        database: PostgresqlDatabase = get_database()
        bars = database.load_bar_data(symbol,
                                      ContractTool.get_exchange_by_symbol(symbol),
                                      interval,
                                      start_time,
                                      end_time)
        bars.sort(key=lambda bar: bar.datetime)
        raw_data_list = []
        for bar in bars:
            raw_data_list.append({"datetime": bar.datetime,
                                  "open": bar.open_price,
                                  "high": bar.high_price,
                                  "low": bar.low_price,
                                  "close": bar.close_price,
                                  "open_interest": bar.open_interest,
                                  "volume": bar.volume,
                                  "symbol": bar.symbol,
                                  "exchange": bar.exchange,
                                  "interval": bar.interval,
                                  "turnover": bar.turnover, })

        df = pd.DataFrame(raw_data_list)
        if df.empty:
            return df

        df.set_index("datetime", inplace=True)

        return df

    @classmethod
    def download_bar_df_to_files(
            cls,
            product_list: list[str],
            contract_start_year: int,
            contract_end_year: int,
            interval: Interval, ):
        """
        从数据库下载数据到本地，并存成 csv 文件

        :param product_list:
        :param contract_start_year:
        :param contract_end_year:
        :param interval:
        :return:
        """
        china_tz = ZoneInfo("Asia/Shanghai")
        # 比合约提前一年的数据
        load_data_start_date = datetime(contract_start_year - 1, 1, 1, tzinfo=china_tz)
        load_data_end_date = datetime(contract_end_year, 12, 31, tzinfo=china_tz)
        suffixes = ContractTool.generate_contract_suffix(contract_start_year, contract_end_year)
        if interval == Interval.MINUTE:
            cache_dir = cls._get_data_cache_1m_df_dir()
        else:
            cache_dir = cls._get_data_cache_1d_df_dir()

        cache_dir.mkdir(parents=True, exist_ok=True)
        for product in product_list:
            jq_symbol = f"{product}JQ00"
            symbol_list = [jq_symbol]
            for suffix in suffixes:
                symbol = ContractTool.generate_symbol_by_product_and_suffix(product, suffix)
                symbol_list.append(symbol)

            for symbol in symbol_list:
                cache_file = cache_dir / f"{symbol}.csv"
                bar_df = cls._load_bar_data_as_df_from_database(symbol,
                                                                interval,
                                                                load_data_start_date,
                                                                load_data_end_date)
                if not bar_df.empty:
                    bar_df.to_csv(cache_file, index=True)

    @classmethod
    def load_bar_df_from_file(cls,
                              symbol: str,
                              period: str):
        """
        从文件加载 bar_df 数据

        :param symbol:
        :param period:
        :return:
        """
        if period == "1m":
            cache_dir = cls._get_data_cache_1m_df_dir()
        else:
            cache_dir = cls._get_data_cache_1d_df_dir()

        try:
            cache_file = cache_dir / f"{symbol}.csv"
            bar_df = pd.read_csv(cache_file, parse_dates=["datetime"])
            if bar_df.empty:
                return pd.DataFrame()

            n_days = bar_df["datetime"].dt.normalize().nunique()
            if n_days <= 60:
                return pd.DataFrame()

            symbol = bar_df["symbol"].iloc[0]
            exchange = bar_df["exchange"].iloc[0]
            part_1, part_2 = ContractTool.get_parts_by_symbol(symbol)
            if part_2 != "JQ00":
                if exchange == "Exchange.CZCE":
                    year = int(f"202{part_2[0:1]}")
                    month = int(part_2[1:])
                else:
                    year = int(f"20{part_2[0:2]}")
                    month = int(part_2[2:])

                delivery_year_month_flag = datetime(year, month, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
                last_trading_day = bar_df.loc[bar_df["datetime"] < delivery_year_month_flag, 'datetime'].iloc[-1]
                bar_df["last_trading_day"] = last_trading_day
            else:
                last_trading_day = bar_df['datetime'].iloc[-1]
                bar_df["last_trading_day"] = last_trading_day

            return bar_df
        except Exception as e:
            print(f"period: {period}, {symbol}.csv file can't find, exception: {e}")
            return pd.DataFrame()

    @classmethod
    def load_spread_bar_df_from_file(cls,
                                     symbol: str,
                                     period: str):
        """
        从文件加载价差bar_df

        :param symbol:
        :param period:
        :return:
        """
        if period == "1m":
            cache_dir = cls._get_data_cache_1m_spread_df_dir()
        else:
            raise Exception("unsupported")

        try:
            cache_file = cache_dir / f"{symbol}.csv"
            bar_df = pd.read_csv(cache_file, parse_dates=["datetime"])

            bar_df["open_interest"] = 0
            bar_df["volume"] = 0
            bar_df["turnover"] = 0
            bar_df["volume"] = 0
            bar_df["exchange"] = Exchange.CFFEX.value
            bar_df["interval"] = Interval.MINUTE.value

            return bar_df
        except Exception as e:
            print(f"{symbol}-{period} csv file can't find, exception: {e}")
            return pd.DataFrame()

    @classmethod
    def get_symbols_by_product_from_files(cls,
                                          product: str,
                                          period: str) -> list[str]:
        """
        根据文件列出某个品种的所有合约代码

        :param product:
        :param period:
        :return:
        """
        if period == "1m":
            cache_dir = cls._get_data_cache_1m_df_dir()
        else:
            cache_dir = cls._get_data_cache_1d_df_dir()

        files = Path(cache_dir).glob(f"{product}*.csv")

        symbol_list = [f.stem for f in files]

        symbol_list = [symbol for symbol in symbol_list if symbol != f"{product}JQ00"]

        return symbol_list

    @classmethod
    def load_product_from_cache(cls,
                                product: str,
                                period: str) -> dict[str, DataFrame]:
        """
        获取该品种所有的 bar_df

        :param product:
        :param period:
        :return:
        """
        symbol_list = cls.get_symbols_by_product_from_files(product, period)
        symbol_bar_df: dict[str, DataFrame] = {}

        for symbol in symbol_list:
            bar_df = cls.load_symbol_from_cache(symbol, period)
            symbol_bar_df[symbol] = bar_df

        return symbol_bar_df

    @classmethod
    def load_symbol_from_cache(cls,
                               symbol: str,
                               period: str) -> DataFrame:
        if period == "1d":
            symbol_bar_df_cache = cls._SYMBOL_BAR_1D_DF_CACHE
        else:
            symbol_bar_df_cache = cls._SYMBOL_BAR_1M_DF_CACHE

        if symbol in symbol_bar_df_cache:
            return symbol_bar_df_cache[symbol]

        bar_df = cls.load_bar_df_from_file(symbol, period)
        symbol_bar_df_cache[symbol] = bar_df

        return bar_df

    @classmethod
    def load_spread_symbol_from_cache(cls,
                                      spread_symbol: str,
                                      period: str) -> DataFrame:
        if period == "1d":
            raise Exception("unsupported")
        else:
            symbol_bar_df_cache = cls._SYMBOL_BAR_1M_DF_CACHE

        if spread_symbol in symbol_bar_df_cache:
            return symbol_bar_df_cache[spread_symbol]

        bar_df = cls.load_spread_bar_df_from_file(spread_symbol, period)
        symbol_bar_df_cache[spread_symbol] = bar_df

        return bar_df

    @classmethod
    def load_product_by_year_from_excel(cls,
                                        product: str,
                                        year: str,
                                        period: str) -> DataFrame:
        if period == "1d":
            bar_df_cache = cls._PRODUCT_YEAR_1D_DF_CACHE
        else:
            raise Exception("not support")

        key = f"{product}-{year}"
        if key in bar_df_cache:
            return bar_df_cache[key]

        file_path = cls._get_data_cache_1d_excel_dir() / f"{key}.xlsx"
        df = pd.read_excel(file_path, engine="openpyxl", parse_dates=["交易日期"])
        df = df.rename(columns={
            "商品名称": "product",
            "合约名称": "symbol",
            "交易日期": "datetime",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "持仓量": "open_interest",
            "成交额": "turnover",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])

        numeric_cols = ['high', 'low', 'open', 'close', 'volume', 'open_interest', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                # 将列转为字符串，去掉逗号，再转回 float
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)

        selected_cols = [
            "symbol",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "turnover",
        ]

        return df[selected_cols]

    @classmethod
    def load_product_top_n_major_from_cache(cls,
                                            product: str,
                                            top_n: int = 3,
                                            t1_shift: bool = True,
                                            ) -> tuple[dict[datetime, list], dict[str, DataFrame]]:
        """主力梯队数据（含 T-1 昨持仓口径处理）。

        t1_shift=True（默认）：T-1 昨持仓口径——交易日 T 盘中取到的是
        "按 T-1 收盘 OI 排序"的梯队，对齐实盘 FuturesDailyData.pre_open_interest
        判定，无未来数据。实现 = 键整体右移一个交易日 + **日历日**填充：
        - 交易日 d 的梯队 = 前一交易日的梯队；
        - 交易日 d 与下一交易日之间的日历日（夜盘凌晨段/周末/节假日）沿用 d
          的梯队（2026-11-12 修复的"周末空洞"语义，与 trowel 的向后查找一致）；
        - 末尾交易日之后的日历日沿用最后已知梯队（回测窗口末端）。
        t1_shift=False：原始口径（交易日 T = T 当日收盘 OI 排序，含盘中
        未来数据），仅供 V1 基类的历史口径复现（use_t1_major=False）。
        """
        symbol_bar_1d_df = BarDataManager.load_product_from_cache(product, "1d")
        daily_top_n_major_list, all_top_n_symbol_list = ContractTool.get_daily_top_n_major_list(symbol_bar_1d_df,
                                                                                                top_n)

        if t1_shift:
            from datetime import timedelta as _td
            trading_dates = sorted(daily_top_n_major_list.keys())
            t1_major_list: dict = {}
            for i, d in enumerate(trading_dates):
                if i > 0:
                    # 交易日 d 自身：T-1 = 前一交易日
                    t1_major_list[d] = daily_top_n_major_list[trading_dates[i - 1]]
                # d 与下一交易日之间的日历日（含下一交易日的夜盘凌晨段）：沿用 d 的梯队
                nxt = trading_dates[i + 1] if i + 1 < len(trading_dates) else d + _td(days=10)
                fill_date = d + _td(days=1)
                while fill_date < nxt:
                    t1_major_list[fill_date] = daily_top_n_major_list[d]
                    fill_date += _td(days=1)
            # 末尾交易日之后的日历日沿用最后已知梯队（回测窗口末端的夜盘凌晨段）
            last_t1 = t1_major_list.get(trading_dates[-1])
            fill_date = trading_dates[-1] + _td(days=1)
            while fill_date <= trading_dates[-1] + _td(days=15):
                t1_major_list.setdefault(fill_date, last_t1)
                fill_date += _td(days=1)
            daily_top_n_major_list = t1_major_list

        symbol_top_n_major_bar_1m_df_dict: dict[str, DataFrame] = {}
        for symbol in all_top_n_symbol_list:
            bar_1m_df = cls.load_symbol_from_cache(symbol, "1m")
            if bar_1m_df.empty:
                print(f"{symbol} dataframe is empty")
                continue

            symbol_top_n_major_bar_1m_df_dict[symbol] = bar_1m_df

        return daily_top_n_major_list, symbol_top_n_major_bar_1m_df_dict


class KLineTool:
    @classmethod
    def _add_column_bar_index_to_bar_df(cls,
                                        product_trading_session):
        time_bar_index_record = []
        bar_index = 1
        for session_idx, (start, end) in enumerate(product_trading_session):
            current = datetime(2000, 1, 1, start.hour, start.minute)
            end_dt = datetime(2000, 1, 1, end.hour, end.minute)

            while current <= end_dt:
                t = current.time()
                time_bar_index_record.append({
                    'time': t.strftime('%H:%M'),
                    'bar_index': bar_index,
                })
                current += timedelta(minutes=1)
                bar_index += 1

        template = pd.DataFrame(time_bar_index_record)
        template = template.sort_values('bar_index').reset_index(drop=True)

        return template[['time', 'bar_index']]

    @classmethod
    def add_column_session_to_bar_df(cls,
                                     df: DataFrame,
                                     session_time_type: list[tuple[time, time]]) -> DataFrame:
        df = df.copy()
        hours = df['datetime'].dt.hour
        dates = df['datetime'].dt.date

        # 判定条件
        is_day = (hours >= 8) & (hours < 18)
        is_night_morning = (hours < 8)  # 凌晨阶段的夜盘
        df['session_type'] = np.where(is_day, 'Day', 'Night')
        df['session_date'] = dates
        # 凌晨阶段的夜盘，其场次基准日期需要减去 1 天
        df.loc[is_night_morning, 'session_date'] = (df.loc[is_night_morning, 'datetime'] - pd.Timedelta(days=1)).dt.date

        df["trading_date"] = df["session_date"]
        df.loc[df["session_type"] == 'Night', 'trading_date'] = np.nan
        df["trading_date"] = df["trading_date"].bfill()
        # 以防最后的数据是夜盘收尾
        df.dropna(subset=['trading_date'], inplace=True)

        df["time"] = df["datetime"].dt.strftime('%H:%M')
        bar_template = cls._add_column_bar_index_to_bar_df(session_time_type)
        df = df.merge(bar_template, on='time', how='left')
        df.drop(columns=['time'], inplace=True)

        return df

    @classmethod
    def add_column_is_near_close_to_bar_df(cls,
                                           df: DataFrame,
                                           period: str = "1m", ) -> DataFrame:
        df = df.copy().sort_values('datetime').reset_index(drop=True)
        df['is_near_close'] = False

        # 1. 计算与下一根 bar 的时间差
        # 如果当前 bar 是某时段的最后一根，它与下一根 bar 的时间差会非常大
        df['time_to_next'] = df['datetime'].shift(-1) - df['datetime']

        # 2. 定义什么是"时段断点"
        # 正常连续的1分钟bar，time_to_next 是 1分钟。
        # 日盘结束到夜盘开始，或者夜盘结束到次日日盘，间隔通常 > 3小时。
        # 我们把间隔大于 3 小时的地方认为是时段结束。
        gap_threshold = pd.Timedelta(hours=3)

        if period == '1min':
            # 找出所有"时段最后一根bar"的索引
            session_end_indices = df[df['time_to_next'] > gap_threshold].index.tolist()
            # 如果是整个数据的最后一根，也算时段结束
            if len(df) > 0:
                session_end_indices.append(df.index[-1])

            # 对每个时段结束点，向前标记10根bar
            for end_idx in session_end_indices:
                start_idx = max(0, end_idx - 9)  # 向前推9根，加上自身共10根
                df.loc[start_idx:end_idx, 'is_near_close'] = True

        else:
            # 1小时周期下，间隔 > 3小时的就是时段结束
            session_end_indices = df[df['time_to_next'] > gap_threshold].index.tolist()
            if len(df) > 0:
                session_end_indices.append(df.index[-1])

            # 1小时周期直接标记这一根
            df.loc[session_end_indices, 'is_near_close'] = True

        # 清理临时列
        df = df.drop(columns=['time_to_next'])

        return df

    @classmethod
    def resample_bar_df(cls,
                        df: DataFrame,
                        period: str) -> DataFrame:
        bar_size = 0
        if period == "5min":
            bar_size = 5
        elif period == "10min":
            bar_size = 10
        elif period == "15min":
            bar_size = 15
        elif period == "30min":
            bar_size = 30
        elif period == "1h":
            bar_size = 60
        elif period == "4h":
            bar_size = 240
        elif period == "1d":
            bar_size = 24 * 60 * 60

        df = df.copy()
        df['group_idx'] = (df['bar_index'] - 1) // bar_size
        agg_dict = {
            'datetime': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'open_interest': 'last',
            'volume': 'sum',
            'turnover': 'sum',
            'symbol': 'first',
            'exchange': 'first',
            'interval': 'first',
            'session_type': 'first',
            'session_date': 'first',
            'last_trading_day': 'last',
        }
        group_keys = ['trading_date', 'group_idx']
        resampled = df.groupby(group_keys, as_index=False).agg(agg_dict)
        resampled['datetime'] = pd.to_datetime(resampled['datetime']) + pd.Timedelta(minutes=1)
        resampled.drop(columns=['group_idx'], inplace=True)

        return resampled

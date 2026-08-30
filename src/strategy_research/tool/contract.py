import re

import pandas as pd
from pandas import DataFrame
from vnpy.trader.constant import Exchange

from strategy_research.config.exchange import get_contract_config


class ContractTool:
    @staticmethod
    def get_exchange_by_product(product: str) -> Exchange:
        contract_config = get_contract_config(product)
        return contract_config.exchange

    @classmethod
    def get_product_by_symbol(cls,
                              symbol: str) -> str:
        contract = symbol.split('.')[0]

        # 规则1：加权指数 — 品种 + JQ00
        m = re.match(r'^([a-zA-Z]+)JQ00$', contract)
        if m:
            return m.group(1)

        # 规则2: 套利合约 — 品种+数字-品种+数字
        m = re.match(r'^([a-zA-Z]+)\d+-([a-zA-Z]+)\d+$', contract)
        if m:
            return m.group(1)

        # 规则3: 普通合约 — 品种 + 数字
        m = re.match(r'^([a-zA-Z]+)\d+$', contract)
        if m:
            return m.group(1)

        return contract

    @classmethod
    def get_exchange_by_symbol(cls, symbol: str) -> Exchange:
        product = cls.get_product_by_symbol(symbol)
        return cls.get_exchange_by_product(product)

    @classmethod
    def generate_symbol_by_product_and_suffix(cls,
                                              product: str,
                                              suffix: str):
        exchange = cls.get_exchange_by_product(product)
        if exchange == Exchange.CZCE:
            suffix = suffix[1:]

        symbol = f"{product}{suffix}"

        return symbol

    @staticmethod
    def generate_contract_suffix(start_year: int,
                                 end_year: int) -> list:
        contract_suffix_list = [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ]

        # 生成期间所有的的合约
        suffixes = []
        for contract_prefix in range(start_year, end_year + 2):
            contract_prefix = str(contract_prefix)[2:]
            for contract_suffix in contract_suffix_list:
                suffixes.append(f"{contract_prefix}{contract_suffix}")

        return suffixes

    @staticmethod
    def get_major_contract_by_open_interest(symbol_bar_1d_df: dict[str, DataFrame]) -> DataFrame:
        all_dfs = []
        for symbol, df in symbol_bar_1d_df.items():
            df_copy = df.copy()
            all_dfs.append(df_copy)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        idx = merged_df.groupby("datetime")["open_interest"].idxmax()
        major_contract_df = merged_df.loc[idx].sort_values("datetime").reset_index(drop=True)

        return major_contract_df

    @staticmethod
    def get_top_n_contract_by_open_interest(symbol_bar_1d_df: dict[str, DataFrame]) -> DataFrame:
        records = []

        for symbol, df in symbol_bar_1d_df.items():
            if df.empty:
                continue

            data = df[["datetime", "symbol", "open_interest"]].copy()
            records.append(data)

        if not records:
            return DataFrame(
                columns=[
                    "datetime",
                    "rank1_symbol", "rank1_oi",
                    "rank2_symbol", "rank2_oi",
                    "rank3_symbol", "rank3_oi",
                ]
            )

        # 合并所有合约
        all_df = pd.concat(records, ignore_index=True)

        # 同一天按照持仓量从大到小排序
        all_df = all_df.sort_values(
            ["datetime", "open_interest"],
            ascending=[True, False],
        )

        # 每天取前三名
        top3 = all_df.groupby("datetime", sort=True).head(3)

        # 添加排名
        top3["rank"] = (
                top3.groupby("datetime").cumcount() + 1
        )

        # 转成宽表
        result = top3.pivot(
            index="datetime",
            columns="rank",
            values=["symbol", "open_interest"],
        )

        # 调整列名
        result.columns = [
            f"rank{rank}_{'symbol' if field == 'symbol' else 'oi'}"
            for field, rank in result.columns
        ]

        result = result.reset_index()

        # 保证列顺序
        result = result[
            [
                "datetime",
                "rank1_symbol", "rank1_oi",
                "rank2_symbol", "rank2_oi",
                "rank3_symbol", "rank3_oi",
            ]
        ]

        return result

    @staticmethod
    def get_parts_by_symbol(symbol: str) -> tuple[str | None, str | None]:
        contract = symbol.split('.')[0]

        # 规则1：加权指数 — 品种 + JQ00
        m = re.match(r'^([a-zA-Z]+)JQ00$', contract)
        if m:
            return m.group(1), "JQ00"

        # # 规则2: 套利合约 — 品种+数字-品种+数字
        # m = re.match(r'^([a-zA-Z]+)\d+-([a-zA-Z]+)\d+$', contract)
        # if m:
        #     return m.group(1)

        # 规则3: 普通合约 — 品种 + 数字
        m = re.match(r'^([a-zA-Z]+)(\d+)$', contract)
        if m:
            return m.group(1), m.group(2)

        return None, None

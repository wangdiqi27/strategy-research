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

    @staticmethod
    def get_product_by_symbol(symbol: str) -> str:
        product = re.match(r'[a-zA-Z]+', symbol).group()

        return product

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
    def arrange_major_contract(symbol_bar_1d_df: dict[str, DataFrame]) -> DataFrame:
        all_dfs = []
        for symbol, df in symbol_bar_1d_df.items():
            df_copy = df.copy()
            all_dfs.append(df_copy)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        idx = merged_df.groupby("datetime")["open_interest"].idxmax()
        major_contract_df = merged_df.loc[idx].sort_values("datetime").reset_index(drop=True)

        return major_contract_df

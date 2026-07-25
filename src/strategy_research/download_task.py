from strategy_research.tool.trowel.config import configure_mode
configure_mode("backtesting", ignore_datafeed=True)


from vnpy.trader.constant import Interval

from strategy_research.config.exchange import get_all_contract_config
from strategy_research.tool.data import BarDataManager

contract_config_map_product = get_all_contract_config()

start_year = 2023
end_year = 2026
product_list = []

for product in contract_config_map_product.keys():
    product_list.append(product)

# BarDataManager.download_bar_df_to_cache(product_list,
#                                         start_year,
#                                         end_year,
#                                         Interval.DAILY)

BarDataManager.download_bar_df_to_cache(product_list,
                                        start_year,
                                        end_year,
                                        Interval.MINUTE)

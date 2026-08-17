from vnpy.trader.constant import Exchange
from vnpy_data_ext.constant import CommissionType

from strategy_research.config.exchange.object import ContractConfig, ContractCommission
from strategy_research.config.exchange.trading_session import SHFE_AU_AG_SC_SESSION, \
    SHFE_CU_SESSION, SHFE_NORMAL_DAY_NIGHT_SESSION, CFF_NORMAL_DAY_SESSION

CFFEX_EXCHANGE_CONFIG: dict[str, ContractConfig] = {
    "IF": ContractConfig(
        long_margin_ratio=0.08,
        short_margin_ratio=0.08,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.23 / 10000,
            close=0.23 / 10000,
            close_yd=0.23 / 10000,
            close_td=0.23 / 10000,
        ),
        multiplier=300,
        price_tick=0.2,
        exchange=Exchange.CFFEX,
        trading_session_time=CFF_NORMAL_DAY_SESSION,
        slippage=1,
    ),
    "IH": ContractConfig(
        long_margin_ratio=0.08,
        short_margin_ratio=0.08,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.23 / 10000,
            close=0.23 / 10000,
            close_yd=0.23 / 10000,
            close_td=0.23 / 10000,
        ),
        multiplier=300,
        price_tick=0.2,
        exchange=Exchange.CFFEX,
        trading_session_time=CFF_NORMAL_DAY_SESSION,
        slippage=1,
    ),
    "IC": ContractConfig(
        long_margin_ratio=0.08,
        short_margin_ratio=0.08,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.23 / 10000,
            close=0.23 / 10000,
            close_yd=0.23 / 10000,
            close_td=0.23 / 10000,
        ),
        multiplier=200,
        price_tick=0.2,
        exchange=Exchange.CFFEX,
        trading_session_time=CFF_NORMAL_DAY_SESSION,
        slippage=1,
    ),
    "IM": ContractConfig(
        long_margin_ratio=0.08,
        short_margin_ratio=0.08,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.23 / 10000,
            close=0.23 / 10000,
            close_yd=0.23 / 10000,
            close_td=0.23 / 10000,
        ),
        multiplier=200,
        price_tick=0.2,
        exchange=Exchange.CFFEX,
        trading_session_time=CFF_NORMAL_DAY_SESSION,
        slippage=1,
    ),
}
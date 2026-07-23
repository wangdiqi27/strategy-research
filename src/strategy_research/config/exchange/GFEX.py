from vnpy.trader.constant import Exchange
from vnpy_data_ext.constant import CommissionType

from strategy_research.config.exchange.object import ContractConfig, ContractCommission
from strategy_research.config.exchange.trading_session import DCE_CZCE_NORMAL_DAY_SESSION

GFEX_EXCHANGE_CONFIG: dict[str, ContractConfig] = {
    "lc": ContractConfig(
        long_margin_ratio=0.18,
        short_margin_ratio=0.18,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.00032,
            close=0.00032,
            close_yd=0.00032,
            close_td=0.00032,
        ),
        multiplier=1,
        price_tick=20,
        exchange=Exchange.GFEX,
        trading_session_time=DCE_CZCE_NORMAL_DAY_SESSION,
        slippage=1,
    ),
    "si": ContractConfig(
        long_margin_ratio=0.1,
        short_margin_ratio=0.1,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.0001,
            close=0.0001,
            close_yd=0.0001,
            close_td=0,
        ),
        multiplier=5,
        price_tick=5,
        exchange=Exchange.GFEX,
        trading_session_time=DCE_CZCE_NORMAL_DAY_SESSION,
        slippage=1,
    ),
    "ps": ContractConfig(
        long_margin_ratio=0.13,
        short_margin_ratio=0.13,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.0005,
            close=0.0005,
            close_yd=0.0005,
            close_td=0.0005,
        ),
        multiplier=3,
        price_tick=5,
        exchange=Exchange.GFEX,
        trading_session_time=DCE_CZCE_NORMAL_DAY_SESSION,
        slippage=1,
    ),
}
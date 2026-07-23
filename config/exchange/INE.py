from vnpy.trader.constant import Exchange
from vnpy_data_ext.constant import CommissionType

from config.exchange.object import ContractConfig, ContractCommission
from config.exchange.trading_session import SHFE_AU_AG_SC_SESSION, SHFE_NORMAL_DAY_NIGHT_SESSION, \
    SHFE_NORMAL_DAY_SESSION

INE_EXCHANGE_CONFIG: dict[str, ContractConfig] = {
    "sc": ContractConfig(
        long_margin_ratio=0.22,
        short_margin_ratio=0.22,
        commission=ContractCommission(
            type=CommissionType.FIXED,
            open=40,
            close=40,
            close_yd=40,
            close_td=240,
        ),
        multiplier=1000,
        price_tick=0.1,
        exchange=Exchange.INE,
        trading_session_time=SHFE_AU_AG_SC_SESSION,
        slippage=1,
    ),
    "lu": ContractConfig(
        long_margin_ratio=0.22,
        short_margin_ratio=0.22,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.0001,
            close=0.0001,
            close_yd=0.0001,
            close_td=0.0003,
        ),
        multiplier=10,
        price_tick=1,
        exchange=Exchange.INE,
        trading_session_time=SHFE_NORMAL_DAY_NIGHT_SESSION,
        slippage=1,
    ),
    "nr": ContractConfig(
        long_margin_ratio=0.11,
        short_margin_ratio=0.11,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=2e-05,
            close=2e-05,
            close_yd=2e-05,
            close_td=0,
        ),
        multiplier=10,
        price_tick=5,
        exchange=Exchange.INE,
        trading_session_time=SHFE_NORMAL_DAY_NIGHT_SESSION,
        slippage=1,
    ),
    "ec": ContractConfig(
        long_margin_ratio=0.22,
        short_margin_ratio=0.22,
        commission=ContractCommission(
            type=CommissionType.RATIO,
            open=0.0006,
            close=0.0006,
            close_yd=0.0006,
            close_td=0.0012,
        ),
        multiplier=50,
        price_tick=0.5,
        exchange=Exchange.INE,
        trading_session_time=SHFE_NORMAL_DAY_SESSION,
        slippage=1,
    ),
}

from dataclasses import dataclass
from datetime import time

from vnpy.trader.constant import Exchange
from vnpy_data_ext.constant import CommissionType


@dataclass(frozen=True)
class ContractCommission:
    type: CommissionType
    open: float
    close: float
    close_yd: float | None = None
    close_td: float | None = None


@dataclass(frozen=True)
class ContractConfig:
    long_margin_ratio: float
    short_margin_ratio: float
    commission: ContractCommission
    multiplier: float
    price_tick: float
    exchange: Exchange
    trading_session_time: list[tuple[time, time]]
    slippage: float = 1

from strategy_research.config.exchange.CZCE import CZCE_EXCHANGE_CONFIG
from strategy_research.config.exchange.DCE import DCE_EXCHANGE_CONFIG
from strategy_research.config.exchange.GFEX import GFEX_EXCHANGE_CONFIG
from strategy_research.config.exchange.INE import INE_EXCHANGE_CONFIG
from strategy_research.config.exchange.SHFE import SHFE_EXCHANGE_CONFIG
from strategy_research.config.exchange.object import ContractConfig

_CONTRACT_CONFIGS = {
    **CZCE_EXCHANGE_CONFIG,
    **DCE_EXCHANGE_CONFIG,
    **GFEX_EXCHANGE_CONFIG,
    **INE_EXCHANGE_CONFIG,
    **SHFE_EXCHANGE_CONFIG,
}


def get_all_contract_config() -> dict[str, ContractConfig]:
    return _CONTRACT_CONFIGS


def get_contract_config(product: str) -> ContractConfig | None:
    try:
        return _CONTRACT_CONFIGS[product]
    except KeyError:
        raise ValueError(f"Unknown contract product: {product}")


__all__ = [
    "get_contract_config",
    "get_all_contract_config"
]
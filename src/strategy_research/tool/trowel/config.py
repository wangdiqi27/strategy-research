from pathlib import Path
import json
import os
import platform

MODE = ""

def is_windows() -> bool:
    return platform.system() == "Windows"

def is_test_mode() -> bool:
    return MODE == "test"

def is_prod_mode() -> bool:
    return MODE == "prod"

def is_dev_mode() -> bool:
    return MODE == "dev"

def load_config(mode: str = "") -> dict:
    global MODE
    if not mode:
        mode = MODE
    temp_path = f'.trowel/{mode}/tl_config.json'
    config = load_json(temp_path)
    return config

def save_config(config: dict, mode: str) -> None:
    temp_path = f'.trowel/{mode}/tl_config.json'
    filepath: Path = get_file_path(temp_path)
    os.makedirs(filepath.parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def configure_mode(mode: str, ignore_database = False, ignore_datafeed = False) -> None:
    global MODE
    MODE = mode
    tl_config = {}
    if mode in ("prod", "test"):
        tl_config = load_config(mode)
        apply_env_overrides(tl_config)
        save_config(tl_config, mode)
    else:
        tl_config = load_config()

    from vnpy.trader.setting import SETTINGS
    from loguru import logger
    from logging import INFO
    SETTINGS["log.active"] = True
    SETTINGS["log.level"] = INFO
    SETTINGS["log.file"] = True
    SETTINGS["log.console"] = True
    # 初始化日志
    from vnpy.trader.logger import config_logger
    config_logger(INFO, True, True)
    logger.info("log active: {}, level: {}, file: {}, console: {}".format(
        SETTINGS["log.active"], SETTINGS["log.level"], SETTINGS["log.file"], SETTINGS["log.console"]
    ))

    lang = "zh_CN.UTF-8"
    os.environ["LANGUAGE"] = lang
    os.environ["LANG"] = lang
    os.environ["LC_ALL"] = lang
    logger.info("set locale to zh")

    if not ignore_datafeed:
        configure_datafeed(tl_config)

    if not ignore_database:
        from vnpy_data_ext.utility.polyfill import patch_postgresql_database
        patch_postgresql_database(
            database=tl_config["database.database"],
            host=tl_config["database.host"],
            port=tl_config["database.port"],
            user=tl_config["database.user"],
            password=tl_config["database.password"],
            timezone=tl_config["database.timezone"]
        )

def configure_datafeed(config: dict = None) -> None:
    """
    配置数据feed
    """
    from loguru import logger

    from vnpy.trader.setting import SETTINGS
    SETTINGS["datafeed.name"] = config["datafeed.name"]
    SETTINGS["datafeed.username"] = config["datafeed.username"]
    SETTINGS["datafeed.password"] = config["datafeed.password"]
    logger.info(f"{SETTINGS['datafeed.name']} datafeed username: {SETTINGS['datafeed.username']}")
    if config["datafeed.name"].startswith("rpc_"):
        from vnpy_data_ext.utility.polyfill import patch_rpc_datafeed
        patch_rpc_datafeed(
            name=config["datafeed.name"],
            req_server=config["datafeed.req_server"],
            sub_server=config["datafeed.sub_server"],
        )

def get_ctp_settings() -> dict:
    tl_config = load_config()
    return {
        "用户名": tl_config["ctp.username"],
        "密码": tl_config["ctp.password"],
        "经纪商代码": tl_config["ctp.broker_id"],
        "交易服务器": tl_config["ctp.trade_server"],
        "行情服务器": tl_config["ctp.market_server"],
        "产品名称": tl_config["ctp.appid"],
        "授权编码": tl_config["ctp.auth_code"]
    }

def get_xt_settings() -> dict:
    tl_config = load_config()
    return {
        "rpc": tl_config["xt.rpc"],
        "req_address": tl_config["xt.rpc.req_server"],
        "sub_address": tl_config["xt.rpc.sub_server"],
        "token": tl_config["xt.token"],
        "股票市场": "是" if tl_config["xt.stock_active"] else "否",
        "期货市场": "是" if tl_config["xt.futures_active"] else "否",
        "期权市场": "是" if tl_config["xt.option_active"] else "否",
        "交易": "是" if tl_config["xt.trading"] else "否",
        "账号类型": tl_config["xt.account_type"],
        "QMT路径": tl_config["xt.path"],
        "资金账号": tl_config["xt.account"]
    }

def get_rpc_settings() -> dict:
    tl_config = load_config()
    return {
        "req_address": tl_config["rpc.req_server"],
        "sub_address": tl_config["rpc.sub_server"]
    }

def get_datafeed_settings() -> dict:
    tl_config = load_config()
    return {
        "name": tl_config["datafeed.name"],
        "username": tl_config["datafeed.username"],
        "password": tl_config["datafeed.password"],
        "req_address": tl_config["datafeed.req_server"],
        "sub_address": tl_config["datafeed.sub_server"],
    }

def get_strategies() -> list[str]:
    tl_config = load_config()
    return tl_config.get("strategies", [])

def load_json(filename: str) -> dict | None:
    filepath: Path = get_file_path(filename)

    if filepath.exists():
        with open(filepath, encoding="UTF-8") as f:
            data: dict = json.load(f)
        return data
    else:
        print(f"未加载到配置文件，请确保文件正确配置: {filepath}")
        os._exit(1)

def get_file_path(filename: str) -> Path:
    cwd: Path = Path.cwd()
    return cwd.joinpath(filename)

def apply_env_overrides(config: dict) -> dict:
    for key, default in config.items():
        env_key = key.upper().replace(".", "_")
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        config[key] = cast_value(default, raw)
    return config

def cast_value(default, raw: str):
    if isinstance(default, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.isdigit():
        return int(raw)
    return raw

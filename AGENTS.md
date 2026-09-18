# AGENTS.md

Quant strategy research repo (Python 3.13, uv-managed, src layout). Chinese futures backtesting research built on vnpy + pandas. No tests, no lint config — verification is running the research scripts.

## Commands

- Setup/deps: `uv sync` (deps include vnpy + vnpy-data-ext from private SSH git remotes `github.com/cybrook/*` — requires SSH key access).
- Run a research backtest (each version is a standalone script):
  `uv run python src/strategy_research/research/standard/trix_vpt_research_v30.py`
  Its `__main__` runs backtests for ALL, LONG, SHORT directions across all products in `G_BACKTEST_PRODUCT_LIST` (src/strategy_research/config/backtesting.py).
- Download/refresh bar data cache: `uv run python src/strategy_research/download_task.py` (edit product list / interval in the file).
- **Run everything from repo root.** Config is loaded from `.trowel/<mode>/tl_config.json` via a cwd-relative path (src/strategy_research/tool/trowel/config.py:24).

## Environment / data gotchas

- Bar data lives in a remote vnpy Postgres DB (credentials in `.trowel/backtesting/tl_config.json`, contains secrets — do not commit changes adding more). Scripts read from local CSV caches under `~/workspace/trading-data/bar-data-cache/` (`BarDataManager` in src/strategy_research/tool/data.py); empty cache files mean that product's data was never downloaded — use `download_task.py` first.
- Before importing vnpy-dependent modules, `configure_mode("backtesting", ignore_datafeed=True)` must be called (see src/strategy_research/download_task.py:1-2) — it patches the DB connection and logging.
- Symbol convention: `{PRODUCT}JQ00` = exchange-weighted index used for indicators; individual contracts (e.g. `FG605`) used for actual open/close in newer versions. CZCE uses 1-digit year codes, others 2-digit (handled in `ContractTool`).
- Reports (HTML charts, xlsx, txt summaries) are written to `factor-report/` — gitignored, safe to delete/regenerate.

## Conventions

- **README.md is the research log**, not user docs: one section per strategy version (v1–v32) describing indicator/entry/exit changes and conclusions. When creating a new version script, add a matching README section. Mark abandoned versions "废弃" in README and suffix the filename `_deprecated` (e.g. `trix_vpt_research_v31_deprecated.py`); never delete old versions.
- New strategy versions are typically copied from the closest ancestor version (README states the base version, e.g. "v30 基于 v28"); version number is parsed from the filename via regex in each script — the `vN` suffix is mandatory.
- Factors/indicators (trix, MACD, ATR, channel compression, CCL, zigzag) go in `src/strategy_research/factor/`; strategy logic subclasses `PandasBacktestingBase` (src/strategy_research/backtesting/).

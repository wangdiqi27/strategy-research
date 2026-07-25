from strategy_research.tool.data import BarDataManager

df = BarDataManager.load_bar_df_from_cache("FGJQ00", "1d")

df["close_pct"] = df["close"].pct_change()
print(df["close_pct"].rolling(60).rank())

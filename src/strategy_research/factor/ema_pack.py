"""
均线束与长期均线的相对状态：EMA5/10/20 三线全部穿越 EMA40 的首次穿越信号（逐日状态机）
"""


class EmaPackCrossStateMachine:

    def __init__(self,
                 min_regime_days: int = 30):
        """
        状态语义：
        - pack_above_run_days / pack_below_run_days：当前连续"三线全上/全下"天数（混合日清零）
        - pack_max_above_run_days：当前"未全下片段"内出现过的最长连续全上周期；
          pack_max_below_run_days 同理。片段 = 两次全下（全上）之间的行情段，
          一旦对侧片段开始，旧周期作废清零，避免已消耗的周期被复用。
        - 符号约定：三线因平滑度不同不会同日翻转（EMA5 最先穿透、EMA20 最后穿透），
          "全下"与"全上"都通过严格 < / > 判定，"混合日"（部分线夹在 EMA40 附近）不算全上/全下。
        """
        self.min_regime_days = min_regime_days

        self.pack_above_run_days = 0
        self.pack_below_run_days = 0
        self.pack_max_above_run_days = 0
        self.pack_max_below_run_days = 0
        self.pack_prev_all_above = False
        self.pack_prev_all_below = False

        self.pack_breakdown_today = False
        self.pack_breakup_today = False

    @property
    def breakdown_today(self) -> bool:
        """当日是否触发做空信号：三线首次全部跌破 EMA40，且此前片段出现过 >= min_regime_days 的连续全上周期"""
        return self.pack_breakdown_today

    @property
    def breakup_today(self) -> bool:
        """当日是否触发做多信号：三线首次全部涨超 EMA40，且此前片段出现过 >= min_regime_days 的连续全下周期"""
        return self.pack_breakup_today

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           ema40: float) -> None:
        """
        每个交易日推进一次状态机。em5/ema10/ema20/ema40 为该交易日的日线指标值，
        在 hybrid_backtest 中应传入"前一交易日收盘后"的值（shift(1) 的日线均线），避免未来数据。

        pack_above_run_days：当前连续高于 EMA40 天数，当今天全部高于 EMA40 加一，否则清零
        pack_below_run_days: 当前连续低于 EMA40 天数，当今天全部低于 EMA40 加一，否则清零
        """
        self.pack_breakdown_today = False
        self.pack_breakup_today = False

        pack_all_above = (ema5 > ema40
                          and ema10 > ema40
                          and ema20 > ema40)
        pack_all_below = (ema5 < ema40
                          and ema10 < ema40
                          and ema20 < ema40)

        # "上方→下方"一侧：全下片刻开始时，按正在结束片段的最长全上周期判定；随后旧周期作废
        if pack_all_below:
            self.pack_breakdown_today = (not self.pack_prev_all_below
                                         and self.pack_max_above_run_days >= self.min_regime_days)
            self.pack_above_run_days = 0
            self.pack_max_above_run_days = 0
        else:
            self.pack_above_run_days = self.pack_above_run_days + 1 if pack_all_above else 0
            self.pack_max_above_run_days = max(self.pack_max_above_run_days, self.pack_above_run_days)

        # "下方→上方"一侧：同理
        if pack_all_above:
            self.pack_breakup_today = (not self.pack_prev_all_above
                                       and self.pack_max_below_run_days >= self.min_regime_days)
            self.pack_below_run_days = 0
            self.pack_max_below_run_days = 0
        else:
            self.pack_below_run_days = self.pack_below_run_days + 1 if pack_all_below else 0
            self.pack_max_below_run_days = max(self.pack_max_below_run_days, self.pack_below_run_days)

        self.pack_prev_all_above = pack_all_above
        self.pack_prev_all_below = pack_all_below

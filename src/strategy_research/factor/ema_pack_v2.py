"""
均线束三线排列状态机 v2：不使用 EMA40，仅用 EMA5/10/20 的多空排列作为穿越判据。

口径：
- 多头排列 = ema5 > ema10 > ema20；空头排列 = ema5 < ema10 < ema20；其余为混合日。
- breakup_today（做多信号）：当日首次出现多头排列，且此前对侧片段出现过
  >= min_regime_days 的连续空头排列周期，且资格周期最近一次被打断至今的
  连续混合天数 <= max_mixed_gap_days（None = 不限制）。
- breakdown_today（做空信号）：同理镜像。
- 接口与 EmaPackCrossStateMachine 完全一致（on_new_trading_day / breakup_today /
  breakdown_today，参数同名），可直接替换 v56/v72 系版本中的状态机实例。

与 v1（EMA40 口径）的语义差异：
- v1 的"全上 EMA40"要求三线整体越过长期均值，趋势反转确认晚；
- v2 的"三线排列"只要短期均线族完成排序即可，反转在实践中通常出现得更早，
  属于更灵敏的确认；代价是对震荡拉锯更敏感（排列反复翻面概率更高）。
"""


class EmaPackCrossStateMachineV2:

    def __init__(self,
                 min_regime_days: int = 30,
                 max_mixed_gap_days: int | None = 5):
        """
        状态语义与 EmaPackCrossStateMachine 对齐：
        - stack_above_run_days / stack_below_run_days：当前连续"多头排列/空头排列"天数（混合日清零）
        - stack_max_above_run_days：当前"未空头排列片段"内出现过的最长连续多头排列周期，反之同理。
          片段 = 两次空头排列（多头排列）信号日之间的行情段，一旦对侧排列信号触发，旧周期作废清零。
        - stack_mixed_run_days：当前连续"混合日"（既不满足多头排列也不满足空头排列）天数。

        max_mixed_gap_days：信号资格的"新鲜度"上限 ——
        首次空头排列（首次多头排列）结算时，要求"资格周期最近一次被打断至今"的连续混合天数
        <= max_mixed_gap_days，否则视为周期过期、放弃信号。None 表示不限制。
        """
        self.min_regime_days = min_regime_days
        self.max_mixed_gap_days = max_mixed_gap_days

        self.stack_above_run_days = 0
        self.stack_below_run_days = 0
        self.stack_max_above_run_days = 0
        self.stack_max_below_run_days = 0
        self.stack_mixed_run_days = 0
        self.stack_prev_stacked_above = False
        self.stack_prev_stacked_below = False

        self.stack_breakdown_today = False
        self.stack_breakup_today = False

    @property
    def breakdown_today(self) -> bool:
        """当日是否触发做空信号：三线首次形成空头排列，且此前片段出现过 >= min_regime_days 的连续多头排列"""
        return self.stack_breakdown_today

    @property
    def breakup_today(self) -> bool:
        """当日是否触发做多信号：三线首次形成多头排列，且此前片段出现过 >= min_regime_days 的连续空头排列"""
        return self.stack_breakup_today

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           ema40: float) -> None:
        """
        每个交易日推进一次状态机。ema5/ema10/ema20/ema40 为该交易日的日线指标值，
        在 hybrid_backtest 中应传入"前一交易日收盘后"的值（shift(1) 的日线均线），避免未来数据。
        ema40 参数仅为保持接口签名一致，v2 判据不消费它。

        stack_above_run_days：当前连续满足三线多头排列天数，当今天满足加一，否则清零
        stack_below_run_days：当前连续满足三线空头排列天数，当今天满足加一，否则清零
        """
        stacked_above = (ema5 > ema10 and ema10 > ema20)
        stacked_below = (ema5 < ema10 and ema10 < ema20)
        stacked_mixed = (not stacked_above) and (not stacked_below)

        self.stack_breakdown_today = False
        self.stack_breakup_today = False

        # 混合天数计数：先以"昨日口径"参与信号决策，再推进到"今日口径"
        stack_mixed_gap_ok = (self.max_mixed_gap_days is None
                              or self.stack_mixed_run_days <= self.max_mixed_gap_days)

        # "多头排列 → 空头排列"一侧：空头排列首次成立时，按正在结束片段的最长多头排列周期判定；随后旧周期作废
        if stacked_below:
            self.stack_breakdown_today = (not self.stack_prev_stacked_below
                                          and stack_mixed_gap_ok
                                          and self.stack_max_above_run_days >= self.min_regime_days)
            self.stack_above_run_days = 0
            self.stack_max_above_run_days = 0
        else:
            self.stack_above_run_days = self.stack_above_run_days + 1 if stacked_above else 0
            self.stack_max_above_run_days = max(self.stack_max_above_run_days, self.stack_above_run_days)

        # "空头排列 → 多头排列"一侧：同理
        if stacked_above:
            self.stack_breakup_today = (not self.stack_prev_stacked_above
                                        and stack_mixed_gap_ok
                                        and self.stack_max_below_run_days >= self.min_regime_days)
            self.stack_below_run_days = 0
            self.stack_max_below_run_days = 0
        else:
            self.stack_below_run_days = self.stack_below_run_days + 1 if stacked_below else 0
            self.stack_max_below_run_days = max(self.stack_max_below_run_days, self.stack_below_run_days)

        # 混合计数推进：排列成立的日打断混合段（清零），混合日累加
        if stacked_mixed:
            self.stack_mixed_run_days += 1
        else:
            self.stack_mixed_run_days = 0

        self.stack_prev_stacked_above = stacked_above
        self.stack_prev_stacked_below = stacked_below

"""
均线束状态机 v3：排列资格周期用 EMA5/10/20 三线排列统计，反转突破确认用 EMA5 反穿 EMA20。

口径：
- 资格周期（对侧排列）：多头排列 = ema5 > ema10 > ema20，空头排列 = ema5 < ema10 < ema20；
  片段内最长连续反向排列周期 >= min_regime_days 才获得信号资格（与 v77 一致）。
- 确认事件（突破日）：EMA5 反穿 EMA20 ——
  breakup（做多）：前一交易日 EMA5 < EMA20，当日 EMA5 > EMA20（快线上穿慢线）；
  breakdown（做空）：前一交易日 EMA5 > EMA20，当日 EMA5 < EMA20。
- 反穿日属于"混合日"（EMA5 越过 EMA20 但排序未完成），不会重置对侧排列资格周期；
  每次满足资格的反穿日都是一次触发（位置模型自然去重，不会重复开仓）。
- ema40 参数仅为保持接口签名一致，v3 判据不消费它。
- 接口与 EmaPackCrossStateMachine 完全一致（on_new_trading_day / breakup_today /
  breakdown_today，参数同名），可直接替换 v56/v72 系版本中的状态机实例。

与 v1/v2 的语义差异：
- v1：确认 = 三线全穿 EMA40（最慢、最滞后的整体性确认）；
- v2：确认 = 三线排列首次成立（排列完成度作为确认）；
- v3：确认 = EMA5 反穿 EMA20（单事件、离散、最早的可反转信号日），
  资格周期仍要求对侧三线排列 ≥30 天历史（防止无趋势震荡市的反穿噪声）。
"""


class EmaPackCrossStateMachineV3:

    def __init__(self,
                 min_regime_days: int = 30,
                 max_mixed_gap_days: int | None = 5):
        """
        状态语义与 v1/v2 对齐：
        - stack_above_run_days / stack_below_run_days：当前连续"多头排列/空头排列"天数（混合日清零）
        - stack_max_above_run_days / stack_max_below_run_days：片段内最长连续排列周期；
          一旦对侧排列日成立，旧片段资格作废清零（先按"作废前"的资格判定当日信号）。
        - stack_mixed_run_days：当前连续"混合日"天数（max_mixed_gap_days 语义与 v1 一致，None 不限制）。
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

        self._prev_ema5 = None
        self._prev_ema20 = None

    @property
    def breakdown_today(self) -> bool:
        """当日是否触发做空信号：EMA5 反穿（下穿）EMA20，且此前片段多头排列 ≥ min_regime_days"""
        return self.stack_breakdown_today

    @property
    def breakup_today(self) -> bool:
        """当日是否触发做多信号：EMA5 反穿（上穿）EMA20，且此前片段空头排列 ≥ min_regime_days"""
        return self.stack_breakup_today

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           ema40: float) -> None:
        """
        每个交易日推进一次状态机。ema5/ema10/ema20/ema40 为该交易日的日线指标值，
        在 hybrid_backtest 中应传入"前一交易日收盘后"的值（shift(1) 的日线均线），避免未来数据。
        反穿判定使用"昨日 vs 今日"的 EMA5/EMA20 对比（均为已收线信息）。
        """
        stacked_above = (ema5 > ema10) and (ema10 > ema20)
        stacked_below = (ema5 < ema10) and (ema10 < ema20)
        stacked_mixed = (not stacked_above) and (not stacked_below)

        self.stack_breakdown_today = False
        self.stack_breakup_today = False

        # 混合天数计数：先以"昨日口径"参与信号决策，再推进到"今日口径"
        stack_mixed_gap_ok = (self.max_mixed_gap_days is None
                              or self.stack_mixed_run_days <= self.max_mixed_gap_days)

        # EMA5 反穿 EMA20（昨日/今日对比）
        cross_up = (self._prev_ema5 is not None
                    and self._prev_ema5 < self._prev_ema20
                    and ema5 > ema20)
        cross_down = (self._prev_ema5 is not None
                      and self._prev_ema5 > self._prev_ema20
                      and ema5 < ema20)

        # 信号判定：先按"旧资格周期"参与决策（若今日恰为对侧排列首日，资格周期尚未作废）
        if stack_mixed_gap_ok:
            self.stack_breakup_today = (cross_up
                                        and self.stack_max_below_run_days >= self.min_regime_days)
            self.stack_breakdown_today = (cross_down
                                          and self.stack_max_above_run_days >= self.min_regime_days)

        # 资格周期推进（与 v2 同序）：空头排列日作废多头资格，多头排列日作废空头资格
        if stacked_below:
            self.stack_above_run_days = 0
            self.stack_max_above_run_days = 0
        else:
            self.stack_above_run_days = self.stack_above_run_days + 1 if stacked_above else 0
            self.stack_max_above_run_days = max(self.stack_max_above_run_days, self.stack_above_run_days)

        if stacked_above:
            self.stack_below_run_days = 0
            self.stack_max_below_run_days = 0
        else:
            self.stack_below_run_days = self.stack_below_run_days + 1 if stacked_below else 0
            self.stack_max_below_run_days = max(self.stack_max_below_run_days, self.stack_below_run_days)

        # 混合计数推进：排列日成立打断混合段（清零），混合日累加
        if stacked_mixed:
            self.stack_mixed_run_days += 1
        else:
            self.stack_mixed_run_days = 0

        self.stack_prev_stacked_above = stacked_above
        self.stack_prev_stacked_below = stacked_below

        self._prev_ema5 = ema5
        self._prev_ema20 = ema20

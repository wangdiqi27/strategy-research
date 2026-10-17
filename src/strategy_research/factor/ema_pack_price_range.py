"""
在均线束穿越状态机（ema_pack.py / ema_pack_first_cross.py）基础上，增加价格区间判断：

- 锚点 = 片段极值：做多锚点为"未全上片段"（下方环境）内出现过的**最低价**，
  做空锚点为"未全下片段"（上方环境）内出现过的**最高价**。
- 突破信号日计算突破价相对锚点的变化幅度（除数为锚点价，带符号）：
  立（做多）pct = (突破价 - 片段最低价) / 片段最低价，正值 = 已从低点起涨；
  破（做空）pct = (突破价 - 片段最高价) / 片段最高价，负值 = 已从高点回撤。
- 幅度门槛 min_price_change_pct（默认 None 只记录不拦截）分方向处理：
  立信号要求 pct > +门槛（已从低点涨起门槛以上才确认）；破信号要求 pct < -门槛（已从高点回撤门槛以上才确认）。
"""
from strategy_research.factor.ema_pack import EmaPackCrossStateMachine
from strategy_research.factor.ema_pack_first_cross import EmaPackFirstCrossStateMachine


class _PriceRangeMixin:

    def __init__(self):
        self.episode_high_price = None       # 未全下片段（上方环境）内出现过的最高价 —— 破信号锚点
        self.episode_low_price = None        # 未全上片段（下方环境）内出现过的最低价 —— 立信号锚点
        self.breakdown_price_change_pct = None   # 破信号日：突破价相对片段最高价的变化（负值 = 已回撤）
        self.breakup_price_change_pct = None     # 立信号日：突破价相对片段最低价的变化（正值 = 已起涨）
        self.breakdown_suppressed_today = False  # 本日破信号因幅度不足被压制
        self.breakup_suppressed_today = False

    @property
    def signal_price_change_pct(self) -> float | None:
        """当日触发信号时，突破价相对锚点价的变化幅度（破/立共用出口）；未触发时为 None"""
        is_breakdown, is_breakup = self._signal_fired()
        if is_breakdown:
            return self.breakdown_price_change_pct
        if is_breakup:
            return self.breakup_price_change_pct
        return None

    def _signal_fired(self) -> tuple[bool, bool]:
        """
        兼容两类父类不同的属性命名：
        - pack_breakdown_today / pack_breakup_today（ema_pack.EmaPackCrossStateMachine）
        - breakdown_today / breakup_today（ema_pack_first_cross.EmaPackFirstCrossStateMachine）
        """
        is_breakdown = getattr(self, 'pack_breakdown_today', False) or getattr(self, 'breakdown_today', False)
        is_breakup = getattr(self, 'pack_breakup_today', False) or getattr(self, 'breakup_today', False)
        return is_breakdown, is_breakup

    def _update_episode_extremes(self,
                                 price: float,
                                 pack_all_above: bool,
                                 pack_all_below: bool):
        """
        片段极值推进：
        - 本日不属于"下方 environment"（非全下日）→ 并入上方环境的最高价统计；
          进入全下片段后，上方环境终结，最高价锚点清空（等下一段重新积累）。
        - 最低价对称处理（非全上日并入，进入全上片段清空）。
        """
        if pack_all_below:
            self.episode_high_price = None
        elif self.episode_high_price is None or price > self.episode_high_price:
            self.episode_high_price = price

        if pack_all_above:
            self.episode_low_price = None
        elif self.episode_low_price is None or price < self.episode_low_price:
            self.episode_low_price = price

    def _apply_price_change_gate(self,
                                 is_breakdown: bool,
                                 is_breakup: bool):
        """
        幅度门槛（分方向符号）：
        - 立信号有效 ⇔ breakup_price_change_pct > +门槛（已从低点涨起门槛以上）
        - 破信号有效 ⇔ breakdown_price_change_pct < -门槛（已从高点回撤门槛以上）
        幅度缺失（锚点未积累到有效值）时同样视为未确认而压制。
        """
        self.breakdown_suppressed_today = False
        self.breakup_suppressed_today = False
        threshold = getattr(self, 'min_price_change_pct', None)
        if threshold is None:
            return is_breakdown, is_breakup

        if is_breakdown and (self.breakdown_price_change_pct is None
                             or self.breakdown_price_change_pct >= -threshold):
            self.breakdown_suppressed_today = True
            is_breakdown = False
        if is_breakup and (self.breakup_price_change_pct is None
                           or self.breakup_price_change_pct <= threshold):
            self.breakup_suppressed_today = True
            is_breakup = False

        return is_breakdown, is_breakup

    @staticmethod
    def _pack_states(ema5: float, ema10: float, ema20: float, ema40: float) -> tuple[bool, bool]:
        pack_all_above = (ema5 > ema40
                          and ema10 > ema40
                          and ema20 > ema40)
        pack_all_below = (ema5 < ema40
                          and ema10 < ema40
                          and ema20 < ema40)
        return pack_all_above, pack_all_below


class EmaPackCrossPriceRangeStateMachine(_PriceRangeMixin, EmaPackCrossStateMachine):
    """
    三线全部穿越口径（与 ema_pack.EmaPackCrossStateMachine 相同的触发规则）+ 片段极值锚点 + 分方向幅度门槛。
    注意：本口径一个全下（全上）片段只有一次触发机会（首下/首上日），被压制的信号不再重试，
    需等对侧新片段重新积累资格。
    """

    def __init__(self,
                 min_regime_days: int = 30,
                 max_mixed_gap_days: int | None = None,
                 min_price_change_pct: float | None = None):
        EmaPackCrossStateMachine.__init__(self, min_regime_days, max_mixed_gap_days)
        _PriceRangeMixin.__init__(self)
        self.min_price_change_pct = min_price_change_pct

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           ema40: float,
                           price: float) -> None:
        # 处理顺序：父类推进（触发判定）→ 结算幅度（锚点此刻仍是"截至昨日"的片段极值）
        # → 幅度门槛 → 片段极值推进/清空
        price = float(price)
        EmaPackCrossStateMachine.on_new_trading_day(self, ema5, ema10, ema20, ema40)

        fired_breakdown, fired_breakup = self._signal_fired()
        if fired_breakdown and self.episode_high_price:
            self.breakdown_price_change_pct = (price - self.episode_high_price) / self.episode_high_price
        elif fired_breakup and self.episode_low_price:
            self.breakup_price_change_pct = (price - self.episode_low_price) / self.episode_low_price

        is_breakdown, is_breakup = self._apply_price_change_gate(fired_breakdown, fired_breakup)
        self.pack_breakdown_today = is_breakdown
        self.pack_breakup_today = is_breakup

        pack_all_above, pack_all_below = self._pack_states(ema5, ema10, ema20, ema40)
        self._update_episode_extremes(price, pack_all_above, pack_all_below)


class EmaPackFirstCrossPriceRangeStateMachine(_PriceRangeMixin, EmaPackFirstCrossStateMachine):
    """
    EMA5 首次单线穿越口径（与 ema_pack_first_cross.EmaPackFirstCrossStateMachine 相同的触发规则）
    + 片段极值锚点 + 分方向幅度门槛。
    与 Cross 口径不同：本片段内存在多次 EMA5 穿越机会，被幅度门槛压制的信号，
    consumed 资格复位，后续穿越可重新触发。
    """

    def __init__(self,
                 min_regime_days: int = 30,
                 min_price_change_pct: float | None = None):
        EmaPackFirstCrossStateMachine.__init__(self, min_regime_days)
        _PriceRangeMixin.__init__(self)
        self.min_price_change_pct = min_price_change_pct

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           ema40: float,
                           price: float) -> None:
        price = float(price)
        EmaPackFirstCrossStateMachine.on_new_trading_day(self, ema5, ema10, ema20, ema40)

        fired_breakdown, fired_breakup = self._signal_fired()
        if fired_breakdown and self.episode_high_price:
            self.breakdown_price_change_pct = (price - self.episode_high_price) / self.episode_high_price
        elif fired_breakup and self.episode_low_price:
            self.breakup_price_change_pct = (price - self.episode_low_price) / self.episode_low_price

        is_breakdown, is_breakup = self._apply_price_change_gate(fired_breakdown, fired_breakup)
        self.breakdown_today = is_breakdown
        self.breakup_today = is_breakup
        # 被幅度门槛压制的信号：consumed 复位，本片段内后续 EMA5 穿越可重试
        if getattr(self, 'breakdown_suppressed_today', False):
            self.breakdown_consumed = False
        if getattr(self, 'breakup_suppressed_today', False):
            self.breakup_consumed = False

        pack_all_above, pack_all_below = self._pack_states(ema5, ema10, ema20, ema40)
        self._update_episode_extremes(price, pack_all_above, pack_all_below)

"""
EMA 排列反转状态机（不使用 EMA40）—— 2026-10-01 口径修订：

信号定义（多头侧，空头侧完全对称）：
1. EMA5/10/20 多头排列（EMA5>EMA10>EMA20）累计 >= N 天（min_alignment_days）。
   容忍口径：EMA5 短暂反向穿越 EMA10 且已知持续 < M 天、且未反向穿越 EMA20 期间，
   排列天数冻结（不累加不清零）；持续 >= M 天则连续认定失败，run 清零、锚点清空。
2. 锚点 = 环境（e5 > e20）内**第一次**形成多头排列那天的价格；空头环境内第一次形成空头排列时同理。
3. 触发 = EMA5 首次反向穿越 EMA20：
   - 多头环境内 EMA5 下穿 EMA20 → 做空候选；
   - 空头环境内 EMA5 上穿 EMA20 → 做多候选。
   触发日综合条件 1、2：变化幅度 = (穿越日价格 - 锚点价)/锚点价（除数为锚点价，带符号），
   ---- 分方向门槛：做空要求 > +R（从多头环境起点已上涨 R% 之后破位）；
   ----           做多要求 < -R（从空头环境起点已下跌 R% 之后穿越确认）。
   门槛未满足时信号被压制（run 已终结，无重试）。

补充说明：
- 触发事件就是排列环境的终结事件（e5 越过 e20），因此无需 consumed/重试机制；
  下一次信号需要新一轮环境重新积累。
- 生产环境输入是 numpy 值：所有状态布尔值在存储前显式 bool() 强转（历史 bug：np.bool_
  与 `is True`/`is False` 的身份判断失效）。
"""


class EmaPackTurnStateMachine:

    def __init__(self,
                 min_alignment_days: int = 30,
                 max_breach_days: int = 5,
                 min_price_change_pct: float | None = None):
        """
        - min_alignment_days（N）：排列累计天数门槛（ tolerate 期间冻结不计数）
        - max_breach_days（M）：EMA5 反向穿越 EMA10 的容忍期上限；持续超过 M 天视为非短暂、
          排列连续认定失败（run/锚点清零）
        - min_price_change_pct（R）：幅度门槛（分方向：做空 > +R、做多 < -R）；None = 只记录不拦截
        """
        self.min_alignment_days = min_alignment_days
        self.max_breach_days = max_breach_days
        self.min_price_change_pct = min_price_change_pct

        self.bull_run_days = 0                  # 环境（e5>e20）内累计多头排列天数（容忍期冻结）
        self.bear_run_days = 0                  # 环境（e5<e20）内累计空头排列天数
        self.bull_breach_streak_days = 0        # e5 低于 e10（仍高于 e20）的连续天数
        self.bear_breach_streak_days = 0        # e5 高于 e10（仍低于 e20）的连续天数
        self.bull_run_anchor_price = None       # 环境内第一次多头排列日价格 —— 做空锚点
        self.bear_run_anchor_price = None       # 环境内第一次空头排列日价格 —— 做多锚点
        self.bull_env_consumed = False          # 环境（e5>e20）内是否已触发过信号
        self.bear_env_consumed = False

        self.turn_down_today = False            # 今日触发做空信号（多头环境终结）
        self.turn_up_today = False              # 今日触发做多信号（空头环境终结）
        self.turn_down_suppressed_today = False
        self.turn_up_suppressed_today = False
        self.turn_down_price_change_pct = None
        self.turn_up_price_change_pct = None

        self._prev_e5_above_ema10 = None
        self._prev_e5_above_ema20 = None        # 环境（e5>e20）的前一日状态

    @property
    def signal_price_change_pct(self) -> float | None:
        """当日触发信号时的幅度共用出口；未触发时为 None"""
        if self.turn_down_today:
            return self.turn_down_price_change_pct
        if self.turn_up_today:
            return self.turn_up_price_change_pct
        return None

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           price: float) -> None:
        """
        每个交易日推进一次状态机。调用方应传入"前一交易日收盘后"的值（shift(1) 日线均线），避免未来数据。
        """
        # 生产环境输入是 numpy 值，布尔状态必须显式 python bool（否则 `is True` 身份判断失效）
        price = float(price)
        e5_above_ema10 = bool(ema5 > ema10)
        e5_above_ema20 = bool(ema5 > ema20)
        bull_alignment = e5_above_ema10 and (ema10 > ema20)
        bear_alignment = (not e5_above_ema10) and (ema10 < ema20)

        e20_cross_below = self._prev_e5_above_ema20 is True and (not e5_above_ema20)   # 多头环境终结
        e20_cross_above = self._prev_e5_above_ema20 is False and e5_above_ema20        # 空头环境终结

        self.turn_down_today = False
        self.turn_up_today = False
        self.turn_down_suppressed_today = False
        self.turn_up_suppressed_today = False

        # 一、触发判定（锚点与 run 均为"截至昨日"的环境状态）
        # 多头环境终结：EMA5 首次下穿 EMA20 → 做空候选：累计多头排列 >= N 且幅度 > +R
        if e20_cross_below and (not self.bull_env_consumed) and self.bull_run_days >= self.min_alignment_days:
            if self.bull_run_anchor_price:
                self.turn_down_price_change_pct = (price - self.bull_run_anchor_price) / self.bull_run_anchor_price
            if (self.min_price_change_pct is None
                    or (self.turn_down_price_change_pct is not None
                        and self.turn_down_price_change_pct > self.min_price_change_pct)):
                self.turn_down_today = True
            else:
                self.turn_down_suppressed_today = True
            self.bull_env_consumed = True
        # 空头环境终结：EMA5 首次上穿 EMA20 → 做多候选：累计空头排列 >= N 且幅度 < -R
        elif e20_cross_above and (not self.bear_env_consumed) and self.bear_run_days >= self.min_alignment_days:
            if self.bear_run_anchor_price:
                self.turn_up_price_change_pct = (price - self.bear_run_anchor_price) / self.bear_run_anchor_price
            if (self.min_price_change_pct is None
                    or (self.turn_up_price_change_pct is not None
                        and self.turn_up_price_change_pct < -self.min_price_change_pct)):
                self.turn_up_today = True
            else:
                self.turn_up_suppressed_today = True
            self.bear_env_consumed = True

        # 二、环境推进（在触发结算之后；e20 穿越日的终止重置一并在此处理）
        self._advance_bull(e5_above_ema10, e5_above_ema20, bull_alignment, price, e20_cross_below)
        self._advance_bear(e5_above_ema10, e5_above_ema20, bear_alignment, price, e20_cross_above)

        self._prev_e5_above_ema10 = e5_above_ema10
        self._prev_e5_above_ema20 = e5_above_ema20

    def _advance_bull(self,
                      e5_above_ema10: bool,
                      e5_above_ema20: bool,
                      bull_alignment: bool,
                      price: float,
                      e20_cross_above: bool) -> None:
        if e20_cross_above:
            # 多头环境从零开始（EMA5 上穿 EMA20）：consumed 复位，等待新一轮连续排列的"第一次"
            self.bull_env_consumed = False
        if not e5_above_ema20:
            # 多头环境终结（触发日或未达资格的破位日）：锚点/run/streak 清零，等重新积累
            self.bull_run_days = 0
            self.bull_run_anchor_price = None
            self.bull_breach_streak_days = 0
            return
        if bull_alignment:
            # 完整排列日：容忍期结束，run 累加
            self.bull_breach_streak_days = 0
            self.bull_run_days += 1
            if self.bull_run_anchor_price is None:
                # 环境内第一次形成多头排列 → 记录锚点价
                self.bull_run_anchor_price = price
        elif e5_above_ema10:
            # EMA5 未反向穿越 EMA10（含 e10/e20 序反）：连续性保持，run 冻结
            self.bull_breach_streak_days = 0
        else:
            # EMA5 短暂反向穿越 EMA10（仍高于 EMA20）：容忍期计数
            self.bull_breach_streak_days += 1
            if self.bull_breach_streak_days >= self.max_breach_days:
                # 短暂变成了持续：连续认定失败（但环境未终结，仍可重新排列积累）
                self.bull_run_days = 0
                self.bull_run_anchor_price = None
                self.bull_breach_streak_days = 0

    def _advance_bear(self,
                      e5_above_ema10: bool,
                      e5_above_ema20: bool,
                      bear_alignment: bool,
                      price: float,
                      e20_cross_below: bool) -> None:
        if e20_cross_below:
            # 空头环境从零开始（EMA5 下穿 EMA20）：consumed 复位
            self.bear_env_consumed = False
        if e5_above_ema20:
            # 空头环境终结：清零重新积累
            self.bear_run_days = 0
            self.bear_run_anchor_price = None
            self.bear_breach_streak_days = 0
            return
        if bear_alignment:
            self.bear_breach_streak_days = 0
            self.bear_run_days += 1
            if self.bear_run_anchor_price is None:
                self.bear_run_anchor_price = price
        elif not e5_above_ema10:
            # EMA5 未反向穿越 EMA10：连续性保持，run 冻结
            self.bear_breach_streak_days = 0
        else:
            # EMA5 短暂反向穿越 EMA10（仍低于 EMA20）：容忍期计数
            self.bear_breach_streak_days += 1
            if self.bear_breach_streak_days >= self.max_breach_days:
                self.bear_run_days = 0
                self.bear_run_anchor_price = None
                self.bear_breach_streak_days = 0

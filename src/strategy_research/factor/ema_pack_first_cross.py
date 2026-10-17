"""
均线束与长期均线的相对状态：三线在 EMA40 同侧持续 >= N 天后，由 EMA5 首次单线穿越触发的信号（逐日状态机）
"""


class EmaPackFirstCrossStateMachine:
    """
    与 EmaPackCrossStateMachine（ema_pack.py）的区别 —— 对比实验用：
    - 触发线不同：本机监听**最快**的 EMA5 第一次单独穿越 EMA40 触发，
      不等 EMA10/20 也跟上（即不等"三线全部完成穿越"）；
      EmaPackCrossStateMachine 等三线全部穿越才触发，信号天然滞后数日。
    - 资格判定相同：触发前，三线持续位于 EMA40 同侧（"未穿越片段"内）的最长连续天数 >= min_regime_days。
    - 消费规则：同一片段内只允许"第一次"穿越触发（consumed 标记），
      片段被对侧真正终结（三线全部过到另一侧）后重置，重新积累资格。
    """

    def __init__(self,
                 min_regime_days: int = 30):
        self.min_regime_days = min_regime_days

        self.above_run_days = 0            # 当前连续"三线全上"天数（混合日清零）
        self.below_run_days = 0            # 当前连续"三线全下"天数
        self.max_above_run_days = 0        # 未全下片段内出现过的最长连续全上周期
        self.max_below_run_days = 0        # 未全上片段内出现过的最长连续全下周期
        self.breakdown_consumed = False    # 当前未全下片段内已触发过破信号
        self.breakup_consumed = False      # 当前未全上片段内已触发过立信号
        self.prev_ema5_above = None        # 前一交易日 EMA5 是否高于 EMA40（None 表示尚未开始）

        self.breakdown_today = False
        self.breakup_today = False

    def on_new_trading_day(self,
                           ema5: float,
                           ema10: float,
                           ema20: float,
                           ema40: float) -> None:
        """
        每个交易日推进一次状态机。调用方应传入"前一交易日收盘后"的值（shift(1) 的日线均线），避免未来数据。
        """
        ema5_above = ema5 > ema40
        pack_all_above = (ema5 > ema40
                          and ema10 > ema40
                          and ema20 > ema40)
        pack_all_below = (ema5 < ema40
                          and ema10 < ema40
                          and ema20 < ema40)

        self.breakdown_today = False
        self.breakup_today = False

        # EMA5 首次穿越检测：昨日的 EMA5 位置 vs 今日的 EMA5 位置
        ## EMA5 由上向下穿越 EMA40 → 做空候选
        if self.prev_ema5_above is not None:
            if self.prev_ema5_above and not ema5_above and not self.breakdown_consumed:
                self.breakdown_today = self.max_above_run_days >= self.min_regime_days
                if self.breakdown_today:
                    self.breakdown_consumed = True
            ## EMA5 由下向上穿越 EMA40 → 做多候选
            elif not self.prev_ema5_above and ema5_above and not self.breakup_consumed:
                self.breakup_today = self.max_below_run_days >= self.min_regime_days
                if self.breakup_today:
                    self.breakup_consumed = True

        # 片段与连续天数推进（与 EmaPackCrossStateMachine 相同的口径）
        ## "上方→下方"一侧：进入全下片段后，旧的全上周期资格作废
        if pack_all_below:
            self.above_run_days = 0
            self.max_above_run_days = 0
            self.breakdown_consumed = False
        else:
            self.above_run_days = self.above_run_days + 1 if pack_all_above else 0
            self.max_above_run_days = max(self.max_above_run_days, self.above_run_days)

        ## "下方→上方"一侧：进入全上片段后，旧的全下周期资格作废
        if pack_all_above:
            self.below_run_days = 0
            self.max_below_run_days = 0
            self.breakup_consumed = False
        else:
            self.below_run_days = self.below_run_days + 1 if pack_all_below else 0
            self.max_below_run_days = max(self.max_below_run_days, self.below_run_days)

        self.prev_ema5_above = ema5_above

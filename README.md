# skewbudget — 硬件时钟树偏差预算分配

对一棵根树的每条边选择**一个**离散缓冲选项,使所有叶节点的到达时间
落在各自的闭窗口内、全树最大偏差 `max(arrivals) - min(arrivals) <= S`,
并最小化缓冲总成本。使用**树 DP + 时间包络/成本 Pareto 状态**精确求解,
搜索预算耗尽与“证明无解”严格区分。

本轮新增(在旧契约之上扩展,旧接口全部保留):每个缓冲选项带**上升沿与
下降沿两个延迟**;根时钟给出**周期与高电平宽度**;每个叶可要求
**最小高/低脉宽**。方案必须同时满足原上升沿到达窗口、skew 约束与每个
叶的脉宽约束(不是软目标);逐叶输出两种边沿时间与高低脉宽。

## 运行环境

- Python **3.14.7**(代码未使用 3.14 专属语法,但仅在该版本验证)。
- **仅标准库**,安装 Python 后无需 `pip install` 任何东西;`requirements.txt`
  保留为空占位。
- 全程离线:无网络、无账号/密钥、无硬件、无数据库/中间件/Docker。
- 示例数据由 `demo.py` 与测试中的确定性代码直接构造。

### venv 准备(可选)

```bash
python3.14 -m venv .venv && . .venv/bin/activate   # 无第三方依赖可装
```

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试(含独立暴力枚举对拍)
python demo.py                            # 固定输入,现场计算,约毫秒级(<8s)
python demo_iteration.py                   # 本轮双沿/脉宽演示,现场计算(<90s)
```

## 接口

```python
from skewbudget import (
    Option, Edge, LeafWindow, ClockTree,   # 输入结构(构造即严格校验)
    Clock,                                 # 根时钟:周期 + 高电平宽度
    solve, validate_tree,                  # 算法
    Status, SolveResult, PruningInfo,      # 输出结构
    LeafTiming,                            # 逐叶两沿时间与脉宽
    MAX_EDGES,                             # 25
)

tree = ClockTree(
    root="r",
    edges=[
        Edge("r", "u", base_delay=5,
             options=[Option(delay=0, cost=0), Option(delay=2, cost=4)]),
        Edge("u", "a", 1, [Option(0, 0), Option(3, 2)]),
        Edge("u", "b", 4, [Option(0, 0), Option(3, 2)]),
    ],
    windows=[LeafWindow("a", lo=0, hi=12), LeafWindow("b", 0, 12)],
)
res = solve(tree, skew_limit=0)            # budget 默认 4000;budget=None 不限
res.status            # Status.OPTIMAL / INFEASIBLE / BUDGET_EXHAUSTED
res.total_cost        # 最小总成本(无可行在案解时为 None)
res.max_delay         # 最大叶到达时间(目标序第 2 关键字)
res.min_delay; res.skew
res.choices           # {边的子端点名: 选项下标}  选项下标 = 输入列表中的下标
res.leaf_arrivals     # {叶: 根到该叶 base_delay+option.delay 的逐边累加和}
res.leaf_path_edges   # {叶: 路径上的边(以子端点名标识),根→叶顺序}
res.pruning           # 被剪枝状态分类计数与预算界,见下
```

### 双沿与脉宽(本轮接口)

```python
opt = Option(delay=3, cost=2, fall_delay=1)   # 上升沿 +3,下降沿 +1
opt = Option(3, 2)                            # fall_delay 缺省 = delay(退化)
clk = Clock(period=20, high_width=8)          # 根: t=0 上升,t=8 下降
win = LeafWindow("a", lo=0, hi=14, min_high=4, min_low=4)
res = solve(tree, skew_limit=0, clock=clk)    # clock=None 时完全等同旧行为
res.leaf_timing["a"]   # LeafTiming(rise, fall, high, low),精确整数
```

时序模型(单工况,不做真实电路模拟):根的上升沿在 t=0、下降沿在
`high_width`,周期为 `period`。叶的上升到达时间 `t_r` 是路径上
`base_delay + option.delay` 的累加;下降沿绝对时间为
`high_width + sum(base_delay + option.fall_delay)`。统一表示:全部用
相对根周期 0 的绝对整数时间,`fall` 允许大于 `period`(表示落入下一
周期)。叶的高脉宽 `high = fall - rise`,低脉宽 `low = period - high`;
可行性要求每叶 `min_high <= high <= period - min_low`——上界同时排除了
“下降沿越过下一上升沿”的情形(`high > period` 即 `low < 0`,必然被
拒绝),下界排除高脉冲消失。脉宽是**硬约束**,目标序仍是字典序
`(总成本, 最大上升延迟, 选项向量)`,不把脉宽混成软目标。skew 与到达
窗口仍只作用于上升沿。所有选项两沿延迟相等(或不给 clock)时,结果与
旧单沿问题逐项一致。

### 目标与并列序

字典序最小化 `(总成本, 最大叶延迟 M, 选项下标向量)`;选项下标向量按
规范边序(根出发、子节点按名字排序的 DFS 前序)排列。同一边上
`delay` 重复时只保留该 delay 下 `cost` 最小者(完全相同保留先出现者),
其余选项顺序**不做重排**,故选项下标即调用方给定的位置。

### 三种结果状态(语义不混淆)

- `OPTIMAL`:无截断地穷举,最优值与并列序均已证明。
- `INFEASIBLE`:无任何截断且根无可行状态——**证明无解**。
- `BUDGET_EXHAUSTED`:某节点 Pareto 前沿超过 `budget` 被截断。此时若有
  可行在案解仍会返回(其可行、数值真实),但**不声称最优**;若截断后
  没有可行状态,同样返回此状态(连可行性也未证明),绝不谎报无解。

`PruningInfo` 字段:`dominated_states`(精确支配/同几何去重)、
`window_states`(窗口前缀区间交空)、`skew_states`(子树内偏差已超 S)、
`pulse_states`(脉宽前缀区间交空,仅在给定时钟时可能非零)、
`budget_states`(仅因预算上限被丢)、`cutoff_min_cost`(被丢状态中的最小
子树成本,只是**截断记录,不是最优界**)、`nodes_visited`。

## 算法与设计取舍

- **状态**:每个子树保留一个具体赋值,记录叶到达包络 `[m,M]`、窗口可行
  前缀集合 `W=[wlo,whi]`(叶窗口播种,过边平移 `−d`,兄弟合并取交,
  `None` 表示 ±∞)、成本与局部前序选项向量。
  给定时钟时另有脉宽可行前缀集合 `P=[plo,phi]`:叶按
  `[min_high − high_width, period − min_low − high_width]` 播种,过边按
  所选选项的 `−(fall_delay − delay)` 平移,兄弟合并同样取交;它与窗口
  约束结构相同,因此同一套精确支配规则保持最优性不失。
- **精确支配**:`cost≤`、包络被包含(`[m_k,M_k] ⊆ [m_c,M_c]`)、可行前缀
  窗口不更窄(`W_c ⊆ W_k`),且在目标并列序上不更差,则被支配状态可删除。
  包络取极值关于合并单调,故该规则对任意上层前缀与任意兄弟子树都成立,
  不损失最优解;状态从不做并集近似,所有输出都是精确整数。
- 子树内偏差 `M-m > S`、平移后窗口交空当场剪枝。
- 后序合并、共享干线边的成本只计一次;时间与空间随 Pareto 前沿规模增长,
  `budget` 是每节点前沿硬上限。

## 输入上限与校验语义

- 边数 ≤ **25**;每边选项数 1..64;每边至少一个选项(必须选且只选一个)。
- 所有时间/成本/S/budget 均为**整数且非负**(budget 仅要求 ≥1)。
- **拒绝 `float`**(包括恰好整数值的 float),拒绝 `bool`
  (`isinstance(True, int)` 为真但显式拒绝),`NaN`/`±Infinity` 抛
  `InvalidNumericError`;类型错误抛 `InputTypeError`;取值越界抛
  `InputValueError`;树结构错误抛带定位路径(如 `edges[2]`、`windows`)
  的 `TreeValidationError`,均为 `SkewBudgetError` 子类。非法输入直接报错,
  绝不静默纠正;构造失败不会产生半成品对象。
- 结构校验:单一根树(无自环/双父/悬空/多连通分量)、每叶恰好一个闭窗口
  (`lo<=hi`)、窗口不得挂在内部点或不存在的点。

## 误差口径

整数域内全部用 Python 任意精度整数精确计算,**没有浮点参与任何判定或
累加,因此不存在舍入误差,也不使用任何容差**;闭区间端点按包含处理,
边界冲突(如到达值恰为 `hi+1`)照常判为不可行。

## 已知限制 / 非目标

- 只实现本题契约的数学子集:单层根树、每边一个离散选项、整数时间;
  不宣称兼容任何行业标准(SDC/STA 等)。
- 单工况(single corner)理想化时序模型:两沿各自独立累加延迟,不做
  真实电路模拟、串扰、OCV 或波形退化建模。
- 不是 CLI 产品:除上述两条命令外不提供参数解析、报表或可视化。
- Pareto 前沿在最坏情形可能较大;25 边/每边 5 选项的压力实例默认预算下
  数十毫秒内完成,但这不是复杂度保证,超大前沿请依赖 `budget` 并按
  `BUDGET_EXHAUSTED` 语义解读结果。

## 测试如何独立验证

`tests/brute.py` 用 `itertools.product` **独立枚举全部组合**直接计分,
从不导入被测核心;`tests/test_dp.py` 在手工场景(共享干线、不可行窗口、
零缓冲、非凸选项、并列序、闭区间端点)与 60 组确定性随机小树上对拍最优
值、最大延迟与每个叶的累加和。`tests/test_pulse.py` 对双沿/脉宽行为做
同样独立的对拍:包含“上升时序可行但高脉冲消失”的反例、下降沿越过
下一上升沿的拒绝、闭边界端点、两沿延迟相等时退化回旧问题,以及 60 组
带随机时钟与脉宽要求的确定性随机小树。对拍之外的路径重建一致性检查
仅作补充。

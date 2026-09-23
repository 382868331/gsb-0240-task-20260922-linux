# skewbudget — 硬件时钟树偏差预算分配

对一棵根树的每条边选择**一个**离散缓冲选项,使所有叶节点的到达时间
落在各自的闭窗口内、全树最大偏差 `max(arrivals) - min(arrivals) <= S`,
并最小化缓冲总成本。使用**树 DP + 时间包络/成本 Pareto 状态**精确求解,
搜索预算耗尽与“证明无解”严格区分。

## 运行环境

- Python **3.14.7**(代码未使用 3.14 专属语法,但仅在该版本验证)。
- **仅标准库**,安装 Python 后无需 `pip install` 任何东西;`requirements.txt`
  保留为空占位。
- 全程离线:无网络、无账号/密钥、无硬件、无数据库/中间件/Docker。
- 示例数据由 `demo.py` 与测试中的确定性代码直接构造。

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试(含独立暴力枚举对拍)
python demo.py                            # 固定输入,现场计算,约毫秒级(<8s)
```

## 接口

```python
from skewbudget import (
    Option, Edge, LeafWindow, ClockTree,   # 输入结构(构造即严格校验)
    solve, validate_tree,                  # 算法
    Status, SolveResult, PruningInfo,      # 输出结构
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
`budget_states`(仅因预算上限被丢)、`cutoff_min_cost`(被丢状态中的最小
子树成本,只是**截断记录,不是最优界**)、`nodes_visited`。

## 算法与设计取舍

- **状态**:每个子树保留一个具体赋值,记录叶到达包络 `[m,M]`、窗口可行
  前缀集合 `W=[wlo,whi]`(叶窗口播种,过边平移 `−d`,兄弟合并取交,
  `None` 表示 ±∞)、成本与局部前序选项向量。
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
- 不是 CLI 产品:除上述两条命令外不提供参数解析、报表或可视化。
- Pareto 前沿在最坏情形可能较大;25 边/每边 5 选项的压力实例默认预算下
  数十毫秒内完成,但这不是复杂度保证,超大前沿请依赖 `budget` 并按
  `BUDGET_EXHAUSTED` 语义解读结果。

## 测试如何独立验证

`tests/brute.py` 用 `itertools.product` **独立枚举全部组合**直接计分,
从不导入被测核心;`tests/test_dp.py` 在手工场景(共享干线、不可行窗口、
零缓冲、非凸选项、并列序、闭区间端点)与 60 组确定性随机小树上对拍最优
值、最大延迟与每个叶的累加和。对拍之外的路径重建一致性检查仅作补充。

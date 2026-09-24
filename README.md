# skewbudget — 硬件时钟树偏差预算分配

对一棵根树的每条边选择**一个**离散缓冲选项,使所有叶节点的到达时间
落在各自的闭窗口内、全树最大偏差 `max(arrivals) - min(arrivals) <= S`,
并最小化缓冲总成本。提供两种模型:

- **单工况**(`solve`):每个选项一个延迟;树 DP + 时间包络/成本 Pareto
  状态精确求解,搜索预算耗尽与“证明无解”严格区分。
- **多工况**(`solve_multicorner`):同一**物理**选项安装一次,在多个
  工艺工况下呈现各自不同的延迟;同一组物理选择必须同时满足**所有**
  工况的叶窗口与 skew 约束,最小化物理成本。题面限定 ≤8 个选择点、
  每点 ≤3 选项(最多 3⁸=6561 个赋值),采用**完整枚举**,最优与无解
  均为证明结论,并返回逐工况验证值。

## 运行环境

- Python **3.14.7**(代码未使用 3.14 专属语法,但仅在该版本验证)。
- **仅标准库**,安装 Python 后无需 `pip install` 任何东西;`requirements.txt`
  保留为空占位。
- 全程离线:无网络、无账号/密钥、无硬件、无数据库/中间件/Docker。
- 示例数据由 `demo.py` 与测试中的确定性代码直接构造。

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试(含独立暴力枚举对拍)
python demo.py                            # 固定输入,现场计算,毫秒级(<90s)
```

## 单工况接口

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

## 多工况接口

```python
from skewbudget import (
    PhysicalOption, PhysicalEdge, Corner, MultiCornerTree,
    MultiCornerStatus, MultiCornerResult, CornerReport,
    solve_multicorner, validate_multicorner_tree,
    MAX_CHOICE_POINTS, MAX_MC_OPTIONS, MAX_CORNERS,   # 8, 3, 8
)

tree = MultiCornerTree(
    root="r",
    edges=[
        # base_delays / option.delays 都是“每工况一个值”,按 corners
        # 的位置对齐;cost 是物理成本,只计一次。
        PhysicalEdge("r", "a", base_delays=[0, 5], options=[
            PhysicalOption(cost=0, delays=[0, 0]),
            PhysicalOption(cost=6, delays=[5, 0]),
            PhysicalOption(cost=6, delays=[0, 5]),
        ]),
        PhysicalEdge("r", "b", base_delays=[5, 0], options=[ ... ]),
    ],
    corners=[
        Corner("fast", skew_limit=0,
               windows=[LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)]),
        Corner("slow", skew_limit=0,
               windows=[LeafWindow("a", 0, 9), LeafWindow("b", 0, 9)]),
    ],
)
res = solve_multicorner(tree)                 # return_all=True 可选
res.status              # MultiCornerStatus.OPTIMAL / INFEASIBLE(均为证明结论)
res.total_cost          # 最小物理成本;INFEASIBLE 时为 None
res.choices             # {边的子端点名: 物理选项下标}
res.corners             # 每个工况一个 CornerReport(按输入顺序)
res.assignments_enumerated   # 枚举的物理赋值总数(含不可行)
res.feasible_assignments     # 同时满足所有工况的赋值数
res.all_optimal_choices      # return_all=True 时:全部并列最优赋值
```

### 逐工况验证值 `CornerReport`

`name`、`skew_limit`、`feasible`(该工况窗口与 skew 是否同时满足)、
`leaf_arrivals`(该工况根→叶 base+option 延迟累加和)、
`min_delay`/`max_delay`/`skew`、`window_ok`(逐叶是否在闭窗口内)、
属性 `skew_ok`(`skew <= skew_limit`)。可用 `res.corner("fast")` 按名取。

### 多工况语义与并列序

- **共享物理选项**:每条边只装一个物理选项,其延迟在不同工况下不同;
  不允许“工况 A 装缓冲、工况 B 不装”。因此各工况不能独立求解,必须
  对同一选项向量联合判定。
- 物理成本只计一次(与多少工况、多少叶路径经过无关)。
- 最小化物理成本;并列时取规范边序(DFS 前序)下选项下标向量字典序
  最小者(单选项边固定下标 0),与单工况约定一致。**不**对工况间延迟
  再设第二关键字。
- 只有两种状态,没有截断:`OPTIMAL`(完整枚举找到最优)、`INFEASIBLE`
  (完整枚举无任何赋值同时满足所有工况——证明无解)。`INFEASIBLE` 时
  `total_cost=None`、`choices={}`、`corners=()`(枚举计数仍真实返回)。
- `return_all=True` 时 `all_optimal_choices` 返回取得最小成本的**全部**
  选项向量(含并列);默认关闭,该字段为 `()`。

## 算法与设计取舍

- **单工况状态**:每个子树保留一个具体赋值,记录叶到达包络 `[m,M]`、窗口可行
  前缀集合 `W=[wlo,whi]`(叶窗口播种,过边平移 `−d`,兄弟合并取交,
  `None` 表示 ±∞)、成本与局部前序选项向量。
- **精确支配**:`cost≤`、包络被包含(`[m_k,M_k] ⊆ [m_c,M_c]`)、可行前缀
  窗口不更窄(`W_c ⊆ W_k`),且在目标并列序上不更差,则被支配状态可删除。
  包络取极值关于合并单调,故该规则对任意上层前缀与任意兄弟子树都成立,
  不损失最优解;状态从不做并集近似,所有输出都是精确整数。
- 子树内偏差 `M-m > S`、平移后窗口交空当场剪枝。
- 后序合并、共享干线边的成本只计一次;时间与空间随 Pareto 前沿规模增长,
  `budget` 是每节点前沿硬上限。
- **多工况枚举**:单工况 Pareto DP 不能跨工况复用(同一物理选项在不同
  工况几何不同),题面又显式限定了搜索规模,故多工况直接完整枚举。
  只有 **>1 选项的边(选择点)**参与笛卡尔积,单选项边强制选择、不计
  入上限;每个赋值对每个工况做一次从根的到达时间累加(每顶点每工况
  一次),逐工况判定窗口与 skew。无任何近似或截断,3⁸=6561 上限在
  本地实测约 0.15 秒。

## 输入上限与校验语义

- **单工况**:边数 ≤ **25**;每边选项数 1..64;每边至少一个选项。
- **多工况**:边数 ≤ 25;工况数 1..**8**;每边物理选项数 1..**3**;
  其中 **>1 选项的边(选择点)≤ 8**(单选项边强制选择,不计数),
  即完整枚举上限 3⁸=6561 个赋值。
- 所有时间/成本/S/budget 均为**整数且非负**(budget 仅要求 ≥1)。
  多工况中每个 `PhysicalOption.delays`、`PhysicalEdge.base_delays` 都
  必须与工况数等长并按位置对齐,不一致在构造期报错。
- **拒绝 `float`**(包括恰好整数值的 float),拒绝 `bool`
  (`isinstance(True, int)` 为真但显式拒绝),`NaN`/`±Infinity` 抛
  `InvalidNumericError`;类型错误抛 `InputTypeError`;取值越界抛
  `InputValueError`;树结构错误抛带定位路径(如 `edges[2]`、
  `corners[1].windows`)的 `TreeValidationError`,均为 `SkewBudgetError`
  子类。非法输入直接报错,绝不静默纠正;构造失败不会产生半成品对象。
- 结构校验:单一根树(无自环/双父/悬空/多连通分量)、每个工况的每叶
  恰好一个闭窗口(`lo<=hi`)、窗口不得挂在内部点或不存在的点、工况
  名不得重复。物理选项的逐工况延迟向量完全相同时,只保留该向量下
  `cost` 最小者(与单工况按 delay 去重同构),选项下标仍指输入位置。

## 误差口径

整数域内全部用 Python 任意精度整数精确计算,**没有浮点参与任何判定或
累加,因此不存在舍入误差,也不使用任何容差**;闭区间端点按包含处理,
边界冲突(如到达值恰为 `hi+1`)照常判为不可行。

## 已知限制 / 非目标

- 只实现本题契约的数学子集:单层根树、每边一个离散选项、整数时间;
  多工况不建模共享缓冲以外的工艺扰动耦合,不提供鲁棒证书或局部锁定
  优化;不宣称兼容任何行业标准(SDC/STA 等)。
- 不是 CLI 产品:除上述两条命令外不提供参数解析、报表或可视化。
- 单工况 Pareto 前沿在最坏情形可能较大;25 边/每边 5 选项的压力实例
  默认预算下数十毫秒内完成,但这不是复杂度保证,超大前沿请依赖
  `budget` 并按 `BUDGET_EXHAUSTED` 语义解读结果。
- 多工况求解复杂度为选项数的选择点次方(上限 3⁸),被题面的 8 选择点 /
  3 选项限制所约束;超过上限的实例在构造期直接拒绝,不做降级搜索。

## 测试如何独立验证

`tests/brute.py` 用 `itertools.product` **独立枚举全部组合**直接计分,
从不导入被测核心;`tests/test_dp.py` 在手工场景(共享干线、不可行窗口、
零缓冲、非凸选项、并列序、闭区间端点)与 60 组确定性随机小树上对拍最优
值、最大延迟与每个叶的累加和。对拍之外的路径重建一致性检查仅作补充。

多工况同理:`tests/brute.py` 中的 `enumerate_multicorner` /
`optimal_multicorner_reference` 独立枚举每个物理赋值并逐工况计分
(只依赖输入模型与 `itertools.product`,不导入求解器);
`tests/test_multicorner.py` 在手算例子(共享物理选项被迫选贵的、
跨工况冲突无解、单工况 skew 失败、闭区间端点绑定、逐工况基延迟、
并列序、孤根、强制边不占选择点)与 **50 组确定性随机小树**(1–3 工况)
上对拍成本、选择、枚举总数、可行赋值数及逐工况到达/skew/窗口判定。

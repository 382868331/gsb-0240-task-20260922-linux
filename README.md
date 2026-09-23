# 硬件时钟树偏差预算分配（clocktree）

离线芯片设计算法工程的可复用库：在一棵有根时钟树上，为每条边从离散
缓冲选项中**恰好选一个**，使各叶子信号到达时间落在各自闭区间、全树叶
最大偏差不超过 `S`，并最小化缓冲总成本。使用树 DP（时间包络 + 成本
Pareto 前沿）求解，整数运算，仅依赖 Python 标准库。

## 环境与依赖

* Python **3.14.7**（3.10+ 的 `int`/dataclass 语义即可运行，开发与验收环境为 3.14.7）。
* **无第三方依赖**，无需网络、账号、密钥、数据库、Docker 或硬件。
* 核心算法全部在本仓库内实现，未调用任何第三方同类求解器。

## 运行

```bash
python -m unittest discover -s tests -v   # 必要测试（20 个用例）
python demo.py                            # 固定输入演示，约 0.1 秒内完成
```

两条命令都只做真实计算：测试含 400 组随机小树与**独立枚举参考**的
交叉核对（枚举参考在 `tests/brute_reference.py`，不导入被测求解器）；
demo 打印一个已证明最优的正常结果和两个实际触发的 `InvalidSpecification`
拒绝边界，无 `sleep`、无预录结果。

## 问题模型

```
边 u→v 实际延迟 = base_delay + 所选选项的 buffer_delay
叶子到达时间    = 根→叶路径上所有边实际延迟之和（根时间为 0）
```

* 每条边的选项 `0` 是**隐式零缓冲**（delay=0, cost=0），其后依次是用户
  给出的显式 `BufferOption(delay, cost)`；“选项序”即该顺序。
* 每边恰好选一个选项；所有 `delay/cost/base_delay/窗口/S` 都是**非负整数**
  （窗口端点可任意整数，只需 `lo <= hi`）。
* 约束（闭区间，含端点）：
  `lo_l <= arrival_l <= hi_l` 且 `max arrival - min arrival <= S`。
* 优化目标字典序：`(最小总成本, 最小最大叶到达时间, 最小选项序字典序)`。

## 公开接口

```python
from clocktree import (
    BufferOption, ClockTree, Edge, LeafWindow, solve,
    OPTIMAL, INFEASIBLE, BUDGET_EXCEEDED,
)

tree = ClockTree(
    edges=(
        Edge("trunk", "r", "m", base_delay=1,
             options=(BufferOption(2, 8), BufferOption(4, 14))),
        Edge("arm_a", "m", "a", base_delay=2,
             options=(BufferOption(1, 1), BufferOption(3, 2))),
        Edge("arm_b", "m", "b", base_delay=5),  # 只有隐式零缓冲
    ),
    windows=(LeafWindow("a", 3, 12), LeafWindow("b", 3, 12)),
    skew_bound=0,
)
sol = solve(tree)
# sol.status / choices / arrivals / total_cost / max_arrival / skew / prune
```

| 对象 | 说明 |
| --- | --- |
| `BufferOption(delay, cost)` | 显式缓冲，非负整数。 |
| `Edge(edge_id, parent, child, base_delay, options=())` | 一条树边；`options` 可空表示只有零缓冲。 |
| `LeafWindow(leaf, lo, hi)` | 叶子到达时间闭区间。 |
| `ClockTree(edges, windows, skew_bound)` | 问题实例；`edges <= 25`。 |
| `solve(tree, *, node_vector_cap=100000, work_cap=5_000_000)` | 求解。 |
| `Solution.status` | `optimal` / `infeasible` / `budget_exceeded`。 |
| `Solution.choices` | 每边选中项（边序、选项 index、base/buffer 延迟与成本）。 |
| `Solution.arrivals` | 每个叶子的根→叶路径累加到达时间。 |
| `Solution.total_cost / max_arrival / skew` | 成本、最大到达、最大偏差。 |
| `Solution.prune` | 剪枝与预算明细：界剪枝/占优剪枝条数、保留状态数、工作量、最大前沿、截断节点、`dropped_cost_lower_bound`、各节点祖先平移界 `translation_bounds`。 |

### 三种结果状态（严格区分）

* `OPTIMAL`：前沿完整构建，经占优论证证明全局最优。
* `INFEASIBLE`：**证明无解**（窗口或 S 在所有选项组合下都不可满足）。
* `BUDGET_EXCEEDED`：搜索预算耗尽、**未证明最优**。若期间找到可行解则
  照常给出（解本身合法），并在 `prune.dropped_cost_lower_bound` 报告
  “任何经过被丢弃状态的完整可行解”的成本下界；若未找到可行解，也
  **不会**误报 `infeasible`——预算耗尽与证明无解永远不会混淆。

提高 `node_vector_cap` / `work_cap` 可把更多实例推进到 `OPTIMAL`。

## 输入上限

* 边数硬上限 **25**（`ClockTree.MAX_EDGES`），超出抛 `InvalidSpecification`。
* 节点/边 id 为非空 `str`；必须构成恰好一个根的连通有根树（单根、
  无环、无重复入边、窗口恰好覆盖全部叶子且只能挂叶子）。
* `node_vector_cap` / `work_cap` 为正整数。

## 错误语义

* 所有数值字段必须是 `int`：传入 `float`（含 `NaN`、`±Infinity`）一律
  拒绝；`bool` 即使是 `int` 子类也显式拒绝。错误类型为
  `clocktree.InvalidSpecification`（同时是 `ValueError`），消息定位到
  具体对象与字段，不做静默纠正。
* 图结构、窗口覆盖、空区间、空树、重复 id 等同样抛该异常。
* 求解器无批处理、无可变共享状态；非法输入在校验阶段失败，不产生
  任何部分结果。

## 算法与设计取舍

* **树 DP 前沿**：每个节点维护 `成本 -> {子树叶相对到达向量: 边选项序列}`。
  子边选项把儿子的向量整体平移后与当前前沿做笛卡尔卷积。
* **Pareto/占优剪枝**：同一相对到达向量只保留最小成本配置——对未来的
  任意共同平移、窗口与极差约束，更便宜的配置永不劣；同成本同向量保留
  选项序字典序最小者。成本升序扫描时，已被更便宜状态实现的向量丢弃。
* **界剪枝（被剪枝界）**：祖先边对一个子树只产生共同平移 `t`。由根到
  该节点路径各边选项（独立取最小/最大）得到整数松弛区间
  `[t_min, t_max]`；向量 `x` 存活当且仅当 `range(x) <= S` 且
  `max(t_min, max_l(lo_l-x_l)) <= min(t_max, min_l(hi_l-x_l))`。
  在根处 `t=0`，该判定就是真实可行性。这是安全剪枝（只删无论祖先
  怎么选都无法挽救的状态）。
* **目标序处理**：成本为前沿主键；同成本下保留选项序字典序最小的
  配置，根处同成本再按最大到达、选项序决胜。
* **预算截断**：卷积按成本层（行主序）生成，超 `work_cap` 或单节点向量
  数超 `node_vector_cap` 时确定性地保留低成本前缀，截断沿树传播。
* 全部使用 Python `int`，无浮点，因此**不存在浮点误差/容差问题**；
  窗口为整数闭区间、判定为精确整数比较，边界（恰等于 `lo/hi/S`）
  按包含端点处理。

## 已知限制

* 一般问题随边数/选项数组合增长；25 边 × 多选项且窗口很宽时前沿可能
  触顶，此时如实返回 `BUDGET_EXCEEDED` 而不是给未证明的“最优”。
  测试覆盖 25 边单链与中等规模随机树；接口上限是边界，不保证任意
  25 边实例都在默认预算内证明最优。
* 只实现本题契约的数学子集：单一根、纯树（无环/无共享汇流单元）、
  每边离散互斥选项、整数延迟与成本；不宣称兼容 SDC/工业 STA 等完整
  行业标准（无转换时间、上升/下降沿、PVT 角、OCV 等概念）。
* 缓冲选项的 `delay` 仅表示附加路径延迟，不建模拟冲器斜率/负载效应。
* `dropped_cost_lower_bound` 是“经过被丢弃状态的解”的成本界，不保证
  小于保留解成本（当最优解完全位于保留前缀中时二者无关）。

## 目录

```
clocktree/__init__.py     # 公开 API 导出
clocktree/model.py        # 数据结构、校验、树布局编译
clocktree/solver.py       # 树 DP / Pareto 前沿 / 剪枝 / 预算语义
tests/test_validation.py  # 校验与错误语义
tests/test_solver.py      # 功能、场景与独立枚举交叉核对
tests/brute_reference.py  # 独立暴力枚举参考（不调用被测核心）
demo.py                   # 固定输入演示
```

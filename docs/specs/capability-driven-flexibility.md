# 数据驱动工艺与能力延迟绑定（步骤 A / B）

对应赛题痛点 1（换型需大量人工重新示教或编程）与痛点 2（缺乏实时任务分配）。

## Status

已交付并纳入回归。`tests/shared/test_capability_routing.py` 共 51 项测试，
其中 12 项是 A/B/C × 1/3 件 × V1/V2 拓扑的**图等价性**断言：数据驱动构图
必须与重构前手写 DAG 完全一致（节点集、依赖边、区域、节拍、资源绑定）。

V1 行为不变；V2 统一运行时（`brazing_sim/dual_line/`）已接入同一份
AND/OR 能力选择：任务在实际资源预留后确定 OR 分支，选择沿 V2 bridge
进入物理技能、事件快照和 UI。V2 场景当前仍按 `V2_DUAL_INSTALL_PROFILE`
禁用 `FIXED_GANTRY`，因为生产 XML 没有独立的焊料固定视觉执行器；该分支
会保留在候选/拒绝原因中，不会被伪装成可执行资源。

## Problem

重构前存在三个结构性问题：

1. **决策空间为零。** `task_graph_builder.py` 中 40 余处 `resources=("ARM1",)`
   把任务与资源一一硬绑定，调度器只能决定"先后顺序"，无法决定"谁来做"。
2. **工艺路线是代码。** 1060 行手写 DAG + `DEFAULT_DURATIONS` 常数表。
   增加产品或调整工序 = 改 Python + 改测试，恰是赛题痛点 1 的反面证据。
3. **节拍与产品参数脱钩。** 常数表无法表达"翅片更多所以检测更久"。

## Solution

### 三份数据契约

| 文件 | 职责 |
|---|---|
| `config/capabilities.yaml` | 工厂级能力本体：27 条能力，覆盖全部 TaskType |
| `config/routings/heat_sink_standard.yaml` | 产品工艺路线：24 道工序，含 OR 分支 |
| `config/resources.yaml`（扩展） | `process_capabilities` + `tool_classes` |

能力声明包含：`task_type`、`requires_tool_class`、`param_schema`、
`duration_model`、`preconditions` / `effects`、`zones`、`preemptive`。

### 参数化节拍

`duration_model` 是表达式而非常数：

```yaml
VISUAL_INSPECTION_FINS:
  param_schema:
    fin_count: {type: int, min: 1, max: 12}
  duration_model: "max(10.0, 4.0 + 0.8 * fin_count)"
```

由 `brazing_sim/flexible/duration_model.py` 用 `ast` 解析求值，
**只允许**数值常量、已声明参数、四则运算与 `min/max/abs/round/ceil/floor`，
不使用 `eval`。未声明参数、函数调用、下标、lambda 均在加载期报错。

`max(envelope, ...)` 中的包线是兼容性下界：A/B/C 三个现有产品的节拍
与重构前完全一致（保证图等价性），而超出包线的更大规格产品自动获得
更长节拍，无需改代码。

### 占位符与产品无关的路线

路线用 `$name` 引用计划变量，因此**一条路线服务全部产品**：

```yaml
- id: OP20_DISPENSE
  capability: MATERIAL_DISPENSING_DUAL
  params:
    path_count: $path_count          # A=10 B=8 C=14，编译期代入
    speed_m_s: $material_speed_m_s
```

可用变量由 `plan_parameter_bindings()` 从已校验的 `ProcessPlan` 导出。
引用未知变量时报错并列出全部可用变量。

### AND/OR 图与延迟绑定

`alternatives` 把"同一工序的等价替代路线"显式写进数据：

```yaml
- id: OP25_INSPECT_BRAZING
  alternatives:
    - {mode: ARM_HANDHELD, capability: VISUAL_INSPECTION_BRAZING, cost_hint: 1.0}
    - {mode: FIXED_GANTRY, capability: FIXED_VISION_BRAZING,      cost_hint: 0.7}
```

加载期强制校验：**OR 分支必须产生与主能力相同的 `effects`**，
否则不是等价替代而是工艺错误。

节点不再声明 `eligible_resources=("ARM1",)`，而是声明 `required_capability`；
候选集由 `CapabilityBinder` 在构图/派工时刻按四个条件算出：

1. 资源是否声明该能力；
2. 是否持有 `requires_tool_class` 类别的工具；
3. 操作参数是否落在资源的 `param_limits` 窗口内；
4. 当前产线剖面是否允许。

### 产线执行剖面（LineExecutionProfile）

能力数据描述**机器人**，剖面描述**场景**。这个区分是必须的：

V1 的翅片技能实现在 `arm1` 的 weld 上
（`async_line_skills._fin_pick_stages` / `_fin_place_stages` 直接引用
`arm1_grasp_{fin_id}`）。即使 Arm3 在 V2 中拥有窄夹爪，把 Arm3 作为
V1 候选会把任务派给无法执行它的 actor。

因此：

| 剖面 | FIN_ASSEMBLY 候选 | 说明 |
|---|---|---|
| `V1_SHALLOW_U_PROFILE` | ARM1 | Arm3 在 V1 仅做检测 |
| `V2_DUAL_INSTALL_PROFILE` | ARM1、ARM3 | 双安装支路 |
| `UNRESTRICTED_PROFILE` | 全部声明者 | 单元测试用 |

`ManufacturingRuntime` 默认使用 V1 剖面。

### 可解释性

每个节点的 payload 记录 `capability`、`capability_params`、
`capability_candidates`（含节拍系数与折算工期）、`capability_alternatives`
以及 `capability_rejected`（每条附中文原因）。例如 C 型 40 mm 节距：

```
ARM3 被排除：资源 ARM3：参数 fin_pitch_m=0.04 超出该资源许可范围 [0.015, 0.03]
```

`GET /orders/plan` 的 `capability_summary` 给出 `flexible_task_count`
（候选数 >1 的节点数），即调度器真实决策空间的规模。步骤 B 之前该值恒为 0。

V2 的延迟绑定在 `ManufacturingRuntime` 的实际资源预留点完成，结果写入
`selected_alternative`。`V2PhysicalExecutionBridge` 将能力模式与资源许可
一起交给 `DualLineRuntime`；因此 `DUAL_NOZZLE` / `SINGLE_TWO_PASS` 会改变
Arm2 的物理路径，`HIGH_RELIABILITY` / `FIRST_ARTICLE` 会在同一套 Arm3
相机协调、S3B 近景复核和状态/UI 中保持一致。Arm3 检测优先与已开始的单片
安装不可抢占约束仍由 V2 物理运行时的单一 `_start` 入口执行。

## 关键取舍

**工艺 vs 拓扑的分界。** 路线只描述"产品要经过哪些工艺"。托盘在
S1/S2A/S2B/S3 之间如何移载、双支路如何合流、区域锁如何互斥，属于产线拓扑，
V1 任务图仍由 `_decorate_async_line` 负责，V2 则由 `dual_line/` 的拓扑与运行时
负责。这条分界让数据层可以替换而不触碰已验证的物理时序；此前从未调用的
`_decorate_flexible_cell` 旧旋转台实现已在仓库清理中删除。

**单夹爪约束下沉为数据。** "第 i 片翅片的夹取必须等第 i-1 片装好"原先是
编译器里的隐式串行化，现在由路线的 `after_previous` 声明：

```yaml
- id: OP30_PICK_FIN
  after: [OP18_PREPARE_FIN_TOOL]
  after_previous: [OP35_INSTALL_FIN]
  per_unit_of: fin
```

**空候选集不清空绑定。** 若能力绑定得不到任何候选，保留原有绑定并写入
`capability_binding_warning`，绝不产出一个无人可执行的任务。

**常数表未删除。** `LEGACY_DURATIONS` 保留为离线回退：不提供 catalog 时
（部分单测直接构造 plan）行为完全不变。`DEFAULT_DURATIONS` 作为别名保留，
历史实验快照仍可导入。

## 换型免编程的可验证证据

新增 D 型产品所需的**全部**改动是两个 YAML 文件：

```text
config/products/product_d.yaml     9 片翅片 / 18 条路径 / 15 mm 节距 / 30 N
config/orders/order_004.yaml       订单数量、优先级、交期
```

没有一行 Python，没有一处示教点。系统自动得到：

| 项目 | A 型（既有） | D 型（新增） | 来源 |
|---|---:|---:|---|
| 任务节点数 | 36 | 44 | 路线按翅片数展开 |
| 涂覆节拍 | 24.0 s | **29.4 s** | `max(24.0, 2.4+1.5×18)` |
| 翅片检测节拍 | 10.0 s | **11.2 s** | `max(10.0, 4.0+0.8×9)` |
| 压紧节拍 | 2.0 s | **3.0 s** | `max(2.0, 0.5+2.5)` |
| 安装候选（V2） | ARM1 / ARM3 | ARM1 / ARM3 | 15 mm 在窄夹爪窗口内 |
| 可柔性调度节点 | 10 | **18** | 能力延迟绑定 |

严格校验同时未被削弱：把 D 型改成 12 片翅片会被拒绝，
原因是 `comb_insert_15mm` 只有 9 个槽位（"梳齿槽数不足"）。
柔性化增加的是**配置能力**，不是**放松约束**。

## 边界

- V2 独立运行时未接入能力绑定，留待后续。
- OR 分支已进入任务图与快照，但选择哪条分支的**成本模型**尚未接入调度器
  （属计划中的步骤 E）。当前调度器仍按加权贪心从候选集中挑选资源。
- 换型时间尚未建模与度量（步骤 D）。
- `FIXED_VISION_GANTRY` 资源已声明，但两条产线的剖面目前都未放开该分支，
  因为对应的 MuJoCo 固定门架几何尚未建模；剖面使其"不可用"这一事实显式化。

## 验证

```bash
python -m pytest tests/shared/test_capability_routing.py -q   # 51 passed
python -m pytest tests/shared tests/v1 -q                     # 223 passed
python -m pytest tests/v2 -q
```

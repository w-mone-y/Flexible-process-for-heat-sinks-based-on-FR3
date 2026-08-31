# CP-SAT 最优参照规划规格

## 目的与边界

CP-SAT 参照层回答“当前在线调度距离一个小规模最优或有界解还有多远”。它读取
`DigitalTwinSnapshot`，输出 `ReferencePlan`，但不派工、不推进任务状态，也不修改
MuJoCo actor。`ManufacturingRuntime` 仍是唯一执行权威。

当前参照对象是快照中已经展开的活动任务 DAG。任务路线在构图阶段确定；CP-SAT 在该
DAG 上优化开始时间、可选执行资源和兼容炉批。它不是尚未展开路线空间上的全工厂全局
最优证明。

## 当前模型

| 对象 | 约束或变量 |
|---|---|
| 活动任务 | 毫秒整数刻度的开始、结束和 interval |
| 前驱关系 | 后继开始时间不早于活动前驱结束时间 |
| 可选机器人/设备 | 每个任务恰选一个合格资源，资源 interval `NoOverlap` |
| 工位 | 同一工位任务 `NoOverlap` |
| 共享区域/运输通道 | `required_zones` 对应 interval `NoOverlap` |
| 已承诺任务 | `RUNNING` / `RESERVED` 固定原资源并从相对时刻 0 继续 |
| 运行中任务 | 使用名义时长减去已执行时间作为剩余时长 |
| 贯通炉 | 同配方、同周期最多三件共享一个不可拆分 batch interval |
| 交期 | 按订单优先级加权的延期变量 |

目标函数为“`100 × 加权延期 + makespan`”。`objective_value` 和 `best_bound` 是同一
复合目标的数值，不应直接解释为物理秒数；`makespan_s` 与
`weighted_tardiness_s` 分开报告。

## 确定性与求解状态

- 单线程求解：`num_search_workers = 1`；
- 显式记录随机种子和求解时限；
- `OPTIMAL`：当前模型实例已证明最优；
- `FEASIBLE`：时限内找到合法解，同时报告下界与 gap；
- `UNKNOWN`：时限内没有可报告解；
- `UNAVAILABLE`：未安装 OR-Tools，运行时继续使用现有调度器；
- `INVALID_INPUT`：任务没有可执行资源或求解结果未通过独立校验。

gap 按 `(objective - best_bound) / objective` 计算。只有 `OPTIMAL` 且 gap 为 0 时，
文档或 UI 才能使用“已证明最优”。

## 独立校验

`ReferencePlanValidator` 不读取 CP-SAT 内部变量，独立检查：

- 活动任务完整且不重复；
- 资源合格性和最短剩余时长；
- 已承诺任务没有推迟或改派；
- 前驱、资源、工位和区域无冲突；
- 炉批容量不超过三件、配方一致、成员均为炉体任务且时间窗一致；
- 报告 makespan 与任务结束时间一致。

校验失败的解不会作为可行参照返回。

## 运行与复现

安装可选依赖：

```bash
python -m pip install -e '.[optimization]'
```

运行 1 至 6 件 V2 订单：

```bash
python benchmarks/run_reference_plan.py \
  --orders A,B,C,A,B,C \
  --time-limit 10 \
  --seed 0 \
  --output benchmarks/results/local-reference-abcabc.json
```

JSON 同时保存输入、快照指纹、计划版本、求解结果、资源忙碌时长和炉批成员。相同代码、
配置、订单和 seed 应产生相同任务图和确定性求解行为；墙钟求解耗时允许有小幅波动。

## 2026-08-21 门禁实测

| 订单 | 时限 | 状态 | 活动任务 | makespan | bound | gap | 独立校验 |
|---|---:|---|---:|---:|---:|---:|---|
| `A,B,C` | 3 s | `OPTIMAL` | 98 | 430.0 s | 430.0 | 0% | 通过 |
| `A,B,C,A,B,C` | 10 s | `FEASIBLE` | 196 | 765.5 s | 711.0 | 7.12% | 通过 |

这些数值是参照任务图的预测计划，不是 MuJoCo 完整物理回放 makespan。后续 Phase 3
必须使用相同快照和目标，把在线方案与该参照并列比较。

Phase 3 的影子比较命令见 [benchmarks/README.md](../../benchmarks/README.md)。

## 当前未建模项

- 未在 CP-SAT 内重新选择尚未展开的 AND/OR 工艺路线；
- 换刀任务按现有 DAG 的显式节点建模，尚未增加任意任务排序下的序列相关 setup 矩阵；
- 六件以内每件已有独立托盘，WIP 上限不构成额外约束；等待期间的持续工位占用尚未做
  blocking interval 扩展；
- MuJoCo 连杆、工具和载荷几何碰撞不属于本层，由后续安全屏障校验；
- CP-SAT 目前是离线/旁路参照，不能直接提交计划。

这些边界将在 Phase 3 的滚动窗口候选与 Phase 5 的几何安全屏障中分别处理，不能通过
放松校验或把预测任务标记完成来规避。

# ADR-0007: 数字孪生快照与影子时长估计边界

## Status

Accepted

## Context

竞赛升级需要让调度器使用统一、可追溯的产线事实，并根据真实物理完成结果修正节拍预测。
当前项目已经有 `ManufacturingRuntime.snapshot()`、V2 `DualLineRuntime.snapshot()` 和
`SystemEvent/EventBus`，但它们分别面向兼容 API、UI 和运行时内部使用，不能直接作为优化器的
不可变输入。若让影子模型直接修改任务的 `estimated_duration`，会在没有实验隔离的情况下改变
既有调度结果，破坏 V1/V2 基线可复现性。

## Decision

- 新增 `DigitalTwinSnapshot` 作为不可变、可序列化的运行时状态边界。
- 快照由现有运行时 snapshot 适配而来，不创建第二套订单、任务或托盘状态。
- 快照保存 `source_name`、仿真时间、事件序列计划版本和状态指纹。
- 新增 `DecisionEvent` 和扩展事件类型，用于后续计划提议、提交、安全检查和估计更新。
- `ManufacturingRuntime.capture_digital_twin()` 与 V2 `DualLineRuntime.capture_digital_twin()`
  是显式捕获入口；默认捕获不发布事件，避免 UI 轮询污染事件日志。
- `ShadowDurationEstimator` 只监听真实 `TASK_STARTED` / `TASK_SUCCEEDED` 事件，当前使用
  `(task_type, resource_id)` 键和 EWMA 更新均值与波动。
- 影子估计结果只进入数字孪生快照，不修改任务 `estimated_duration`、资源状态或调度选择。
- reset 清除估计样本和未完成开始记录；同一订单和种子仍得到相同任务结果。

## Consequences

- CP-SAT 和 TwinShield-RH 可以在不侵入物理运行时的前提下消费统一快照。
- UI、headless 和实验能够引用同一状态指纹，检测状态漂移。
- 当前阶段可以比较静态节拍与影子预测，而不把预测反馈误当成已验证的性能提升。
- 后续若要让估计进入调度目标，必须先增加显式策略开关、对照实验和回退测试，不能直接
  读取影子值覆盖现有节拍。

## Invariants

- 快照内部嵌套映射和序列不可变；`as_dict()` 返回副本。
- 物理完成事件是唯一可用于更新样本的证据。
- 未知任务、负时长和无效时间戳被忽略，不污染统计。
- V1 与 V2 的旧 snapshot 字典接口保持兼容。

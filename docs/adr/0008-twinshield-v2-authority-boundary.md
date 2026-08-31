# ADR-0008: TwinShield-RH 以原子承诺窗口接管 V2 任务选择

## Status

Accepted

## Context

Phase 3 已能从不可变 `DigitalTwinSnapshot` 生成 TwinShield-RH 影子计划，但影子结果
不能直接作为物理执行许可。若运行时逐个启动候选任务，其中一个候选在资源、区域、运动
预约或 actor 启动阶段失败，就可能留下半套已启动任务；若每个 MuJoCo step 都重新求解，
还会造成大量重复回退和明显墙钟开销。

## Decision

- TwinShield-RH 只替换 V2 的 READY 任务选择，不替换 `ManufacturingRuntime`、物理技能或
  MuJoCo actor。
- `TwinShieldAuthority` 使用“规划快照 + 提交时实时状态”双重校验，整个候选窗口任一任务
  失效即全部拒绝。
- 校验范围包含快照指纹、READY、资源能力与工具、资源唯一性、工位、共享区域和当前可启动
  时间；已处于 `RESERVED` / `RUNNING` 的动作不可改派或抢占。
- 原子提交分两阶段：先为全部任务取得运动计划、资源和区域预留，再启动全部 actor。任一步
  失败会取消已启动技能、恢复托盘所有权、释放运动/资源/区域预约并把本窗口任务恢复为 READY。
- 只有完整窗口启动成功后才发布 `TASK_RESERVED`、`TASK_STARTED` 和 `PLAN_COMMITTED`。
- TwinShield 只在 READY 集合、资源、区域或工位事实变化时重规划；仿真时间推进本身不触发
  求解。
- 求解异常、快照过期、独立校验失败、空候选或提交失败都回退到现有
  `DynamicPriorityScheduler`，不得把订单置为 ERROR。
- V2 默认模式为 `AUTHORITY`；`FALLBACK` 是操作员显式回滚模式，完全绕过 TwinShield；
  `SHADOW` 与 `OFF` 保留对照用途。V1 默认仍为 `OFF`。
- 决策快照公开接管次数、回退次数、回退原因、最后候选和决策延迟 p50/p95/max。

## Consequences

- V2 在线调度拥有可测试的唯一接管入口，同时保留一条确定性的旧调度器退路。
- 多任务窗口不会因第二个任务启动失败而留下第一个任务继续运行。
- 事件触发与快照复用后，A/B/C 对照中的 TwinShield 重复回退由 3862 次降到 32 次，
  墙钟开销由约 91.4 s 降到约 22.0 s；这属于计算效率修复，不等同于物理 makespan
  大幅提升。
- 当前安全门只覆盖逻辑资源、工位、区域和既有运动预约。MuJoCo 连杆/工具/载荷真实几何
  校验仍属于 Phase 5/6，不能因本 ADR 而宣称几何安全屏障已完成。

## Invariants

- `ManufacturingRuntime` 仍是任务状态和物理完成的唯一逻辑权威。
- CP-SAT 仍是参照计划，不直接派工。
- RUNNING/RESERVED 任务不抢占、不改派。
- 原子窗口失败后无资源、区域、运动预约或托盘所有权泄漏。
- V1、Viewer/headless、pause/continue/reset 和 0.25×～32× 逻辑结果保持兼容。

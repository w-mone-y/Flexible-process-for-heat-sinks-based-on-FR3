# TwinShield-RH V2 权威派工规格

## 目标

Phase 4 让 TwinShield-RH 在 V2 的安全任务边界选择“下一组可以并行启动的任务”，同时
确保任何不确定状态都能无损回到现有动态优先级调度器。它不改变 MuJoCo 场景、轨迹、
工艺接触动作或物理完成证据。

## 控制链

```text
ManufacturingRuntime READY 集合
        │
        ├── 当前 DynamicPriorityScheduler（始终计算，作为确定性回退）
        │
        └── DigitalTwinSnapshot → TwinShieldShadowScheduler
                                → TwinShieldAuthority 实时复核
                                → 原子预留/启动整个承诺窗口
                                           │
                          失败 ─────────────┴──→ 当前调度器
```

## 模式

| 模式 | 行为 | 用途 |
|---|---|---|
| `AUTHORITY` | TwinShield 优先，任一门控失败自动回退 | V2 默认竞赛模式 |
| `FALLBACK` | 不调用 TwinShield，直接使用当前动态调度器 | 操作员回滚/公平对照 |
| `SHADOW` | 不接管；仅通过显式接口生成影子计划 | Phase 3 复现实验 |
| `OFF` | 不生成也不使用 TwinShield 决策 | V1 与兼容模式 |

命令行示例：

```bash
python brazing_line_v2.py --headless --orders A,B,C --fast \
  --twinshield-mode AUTHORITY

python brazing_line_v2.py --headless --orders A,B,C --fast \
  --twinshield-mode FALLBACK
```

## 决策门控

候选窗口只有全部满足以下条件才可提交：

1. 候选使用的快照指纹仍等于提交时指纹；
2. 所有候选任务仍为 READY，且没有工具、物理派工或运动规划 blocker；
3. 资源属于任务候选集，当前 IDLE，具备任务能力和所需工具；
4. 窗口内任务、资源、工位和共享区域均不重复冲突；
5. 既有区域 lease 不与候选冲突；
6. 候选预计开始时间不晚于当前安全边界；
7. 独立计划校验结果为 valid；
8. 并行任务不超过配置上限（当前为 3）。

RUNNING/RESERVED 任务不会进入 READY 候选，因此不能被抢占或重新绑定。

## 原子提交与回滚

原子提交先完成全窗口预检和预留，再启动 actor。若后一个 actor 启动失败，已经启动的技能
会收到 cancel，运输任务通过 `_abort_cell_state()` 恢复原工位所有权，所有 task/resource/
zone/motion reservation 一并释放。失败窗口不会写入 assignment history，也不会发布半套
任务启动事件。

提交失败后，当前动态调度器在同一安全 tick 使用此前已计算的确定性候选继续派工。规划或
校验失败只增加回退证据，不改变订单为 ERROR。

## 事件触发和性能

TwinShield 不在每个 50 ms 仿真 tick 重算。以下事实组合形成稳定决策签名：

- READY 任务及 blocker；
- 资源状态、当前任务、工具和故障；
- 区域 lease；
- 工位托盘、占用者和可运输状态。

只有签名改变才重新求解。`snapshot()["twinshield"]` 输出：

- `authority_count` / `fallback_count`；
- `last_source` / `last_fallback_reason`；
- `last_decision`（任务、资源、成本、拒绝原因和快照指纹）；
- `decision_latency_ms`（sample_count、p50、p95、maximum）。

## 复现实验

```bash
python benchmarks/run_authority_comparison.py \
  --orders A,B,C \
  --modes AUTHORITY,FALLBACK \
  --output benchmarks/results/local-authority-abc.json
```

该命令运行完整 MuJoCo headless 流程，并比较物理 makespan、吞吐、Arm1 空闲、跨臂重叠、
墙钟耗时、接管/回退次数和决策延迟。预测 objective 不替代物理 makespan。

## Phase 4 边界

- 已完成：V2 默认接管、原子提交、确定性回退、事件触发、CLI 回滚、延迟与决策证据。
- 尚未完成：真实连杆/工具/载荷几何碰撞强制门控（Phase 5/6）。
- 当前 A/B/C 实测只产生约 0.03% makespan 改善；因此第四阶段证明了安全接管和可回滚，
  不能宣称已经达到规划书中“六订单平均下降 10%”的最终算法目标。
- 六订单 `A,B,C,A,B,C` 完成 6/6，最终决策延迟 p50 为 32.82 ms、p95 为
  42.99 ms、最大 45.82 ms；满足 Phase 4 的在线决策延迟门禁。

# 性能基准

本目录保存 V2、V2-Serial、正式 V1 与早期 V1 的可复现实测工具和精选结果。基准同时区分：

- **仿真 makespan**：由订单完成事件给出的仿真时间，衡量产线流程效率。
- **墙钟时间**：同一台电脑从启动进程到输出最终状态的真实运行耗时，衡量程序计算效率。
- **仿真吞吐**：`完成件数 ÷ 仿真 makespan`，不会使用调度器预计时长替代实测值。

## 复现 2026-07-29 对照

先为两个历史版本建立只读 worktree：

```bash
git worktree add --detach /tmp/fr3-v1-baseline v1.0.0
git worktree add --detach /tmp/fr3-v1-early a31730f
```

再从当前仓库根目录运行：

```bash
python benchmarks/compare_versions.py \
  --v1-root /tmp/fr3-v1-baseline \
  --v1-early-root /tmp/fr3-v1-early \
  --profile full \
  --runs 1 \
  --output-dir benchmarks/results/local-reproduction
```

完整物理动作一次运行需要数分钟。需要快速检查命令、数据解析和版本能力矩阵时，可将
`--profile full` 改为 `--profile fast`；快速模式结果不能替代完整运动基准。

## 公平性边界

### V2 / V2-Serial 对照

`V2-Serial` 使用与 V2 完全相同的 MuJoCo 场景、订单几何、运输和炉体规则，
但关闭 Arm3 的翅片安装支路，并将安装任务按 Arm1 顺序执行。它是衡量“第二条
安装支路和动态派工带来的收益”的控制组；V1 不参与这组性能百分比。

```bash
python brazing_line_v2.py --headless --orders A,A,A --fast --benchmark-mode SERIAL
```

- 三个版本使用相同 A 产品；三件 A 均只执行一次炉批（支持时）。
- 混合场景固定为 A/B/C 各一件、相同优先级。
- 早期 V1 只支持 A，且完整运动会触发炉体与 Arm2 的安全接触停机，所以多件与混合场景
  标记为“不适用”，不会用快进结果补造完成时间。
- V2 现阶段状态仍标记为 `CONTROL_PLANE_REHEARSAL`；报告既展示已完成的物理动画与事件
  门控，也明确保留 Phase 2 真实 TCP/抓取验收边界。

精选报告见 [2026-07-29 V1/V2 对照](results/2026-07-29-v1-v2/comparison.md)。

## CP-SAT 小规模参照

Phase 2 的旁路求解器可为 1 至 6 件 V2 订单输出 objective、best bound、gap、独立
校验结果、资源占用和炉批成员：

```bash
python benchmarks/run_reference_plan.py \
  --orders A,B,C,A,B,C \
  --time-limit 10 \
  --seed 0 \
  --output benchmarks/results/local-reference-abcabc.json
```

该结果是当前数字孪生任务图的预测参照，不会派工，也不能替代完整 MuJoCo 物理回放
makespan。模型、状态解释和已知边界见
[CP-SAT 最优参照规划规格](../docs/specs/cp-sat-reference-planning.md)。

## TwinShield-RH 影子比较

Phase 3 以同一快照同时生成当前派工边界、TwinShield-RH 影子计划和 CP-SAT 参照：

```bash
python benchmarks/run_shadow_comparison.py \
  --orders A,B,C \
  --time-limit 10 \
  --seed 0 \
  --output benchmarks/results/local-shadow-abc.json
```

命令会检查影子求解前后的实际派工记录没有变化。`shadow_schedule` 中的 `selected`
是下一安全窗口的建议动作，`rejected` 是候选被资源、工具、WIP、区域或窗口容量拒绝
的原因；它们不是物理执行许可。

## TwinShield-RH V2 权威派工对照

Phase 4 使用完整 MuJoCo headless 流程，对比 TwinShield 权威模式和操作员回退模式：

```bash
python benchmarks/run_authority_comparison.py \
  --orders A,B,C \
  --modes AUTHORITY,FALLBACK \
  --output benchmarks/results/local-authority-abc.json
```

报告同时保存物理 makespan、吞吐、Arm1 空闲、跨臂重叠、墙钟耗时、原子接管次数、
回退次数和决策延迟。`FALLBACK` 完全绕过 TwinShield，因此是同一代码与物理场景下的
可复现回滚基线。详细边界见
[TwinShield-RH V2 权威派工规格](../docs/specs/twinshield-v2-authority.md)。

本阶段精选实测见
[2026-08-22 Phase 4 权威派工验收](results/2026-08-22-phase4/README.md)。

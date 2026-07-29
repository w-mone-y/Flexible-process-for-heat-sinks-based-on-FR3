# 性能基准

本目录保存 V2、正式 V1 与早期 V1 的可复现实测工具和精选结果。基准同时区分：

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

- 三个版本使用相同 A 产品；三件 A 均只执行一次炉批（支持时）。
- 混合场景固定为 A/B/C 各一件、相同优先级。
- 早期 V1 只支持 A，且完整运动会触发炉体与 Arm2 的安全接触停机，所以多件与混合场景
  标记为“不适用”，不会用快进结果补造完成时间。
- V2 现阶段状态仍标记为 `CONTROL_PLANE_REHEARSAL`；报告既展示已完成的物理动画与事件
  门控，也明确保留 Phase 2 真实 TCP/抓取验收边界。

精选报告见 [2026-07-29 V1/V2 对照](results/2026-07-29-v1-v2/comparison.md)。

# Phase 4：TwinShield-RH V2 权威派工验收

> 日期：2026-08-22  
> 模式：完整 MuJoCo headless、`--fast`、`dt=0.05 s`  
> 场景与物理 actor：V2 当前版本，未修改布局和工艺轨迹

## A/B/C 权威模式与回退模式

| 指标 | `AUTHORITY` | `FALLBACK` | 结论 |
|---|---:|---:|---|
| 完成件数 | 3/3 | 3/3 | 两种模式均完整完成 |
| 物理 makespan | 167.55 s | 167.60 s | TwinShield 改善 0.03% |
| 吞吐 | 64.458 件/仿真小时 | 64.439 件/仿真小时 | 改善 0.03% |
| Arm1 空闲 | 105.85 s | 105.90 s | 改善 0.05% |
| 多臂重叠 | 52.20 s | 52.20 s | 无回退 |
| 决策 p95 | 28.26 ms | 不适用 | 达到 <50 ms 门禁 |
| 决策最大值 | 30.03 ms | 不适用 | 达到 <200 ms 目标 |
| 墙钟耗时 | 21.99 s | 20.09 s | 权威门控仍有约 9.4% 计算开销 |

该结果证明 Phase 4 的安全接管、回滚和延迟门禁成立，但不能证明在线成本模型已经带来
显著物理提速。0.03% 差异应如实报告为“基本持平”。

## 六订单权威模式

`A,B,C,A,B,C` 完成 6/6，物理 makespan 为 270.35 s，吞吐为 79.896 件/仿真小时；
决策 p50/p95/max 分别为 32.82/42.99/45.82 ms。未出现订单 ERROR、资源/区域泄漏或
调度死锁。

## 原始证据

- [`authority-abc.json`](authority-abc.json)：A/B/C 双模式公平对照。
- [`authority-abcabc.json`](authority-abcabc.json)：六订单权威模式验收。

复现命令：

```bash
python benchmarks/run_authority_comparison.py \
  --orders A,B,C \
  --modes AUTHORITY,FALLBACK \
  --output benchmarks/results/2026-08-22-phase4/authority-abc.json

python benchmarks/run_authority_comparison.py \
  --orders A,B,C,A,B,C \
  --modes AUTHORITY \
  --output benchmarks/results/2026-08-22-phase4/authority-abcabc.json
```


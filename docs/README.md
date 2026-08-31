# 文档导航

项目文档分为“当前有效说明”和“历史追溯材料”。首次阅读请优先查看当前说明。

> [← 返回项目首页](../README.md) ·
> [快速开始](../README.md#quick-start) ·
> [源码导航](../README.md#documentation)

## 按读者选择

| 读者 | 推荐路线 |
|---|---|
| 第一次使用 | [项目首页](../README.md) → [快速开始](../README.md#quick-start) → [版本选择](../README.md#version-choice) |
| 准备二次开发 | [项目目录说明](architecture/项目目录说明.md) → [领域上下文](../CONTEXT.md) → [V2 规格](specs/v2-dual-install-line.md) |
| 关注实验数据 | [V1/V2 实测效率报告](../benchmarks/results/2026-07-29-v1-v2/comparison.md) → [`benchmarks/`](../benchmarks/) |
| 修改 MuJoCo 场景 | [视觉模型与资产说明](architecture/视觉模型与资产说明.md) → [V1 XML](../scenes/production/brazing_line.xml) / [V2 XML](../scenes/production/brazing_line_v2.xml) |

## 当前有效说明

- [特等奖导向升级实施规划](competition/2026-08-21-特等奖导向升级实施规划.md)
- [项目目录说明](architecture/项目目录说明.md)
- [视觉模型与资产说明](architecture/视觉模型与资产说明.md)
- [V2 双安装支路规格](specs/v2-dual-install-line.md)
- [数据驱动工艺与能力延迟绑定](specs/capability-driven-flexibility.md)
- [换型建模、序列相关设置时间与 KPI](specs/changeover-modelling.md)
- [V2 扰动柔性与控制台接通](specs/v2-disturbance-flexibility.md)
- [数字孪生与影子估计决策](adr/0007-digital-twin-shadow-boundary.md)
- [CP-SAT 最优参照规划规格](specs/cp-sat-reference-planning.md)
- [TwinShield-RH V2 权威派工规格](specs/twinshield-v2-authority.md)
- [TwinShield V2 权威边界 ADR](adr/0008-twinshield-v2-authority-boundary.md)
- [V1/V2 实测效率报告](../benchmarks/results/2026-07-29-v1-v2/comparison.md)
- [Arm2 在整体流程中的任务](process/Arm2%20在整体流程中的任务.md)

## 研究材料

- `research/coordinate_frames/`：初始夹取位置、旋转中心和四元数示意。
- `research/motion_planning/`：早期运动规划研究摘要。

## 历史追溯

- `history/prompts/`：阶段性提示词和需求记录。
- `history/plans/`：旧版实施方案。

历史材料可能描述已经撤销或替换的布局与流程，不代表当前代码行为。当前事实以
`README.md`、`brazing_sim/`、`scenes/production/` 和自动化测试为准。

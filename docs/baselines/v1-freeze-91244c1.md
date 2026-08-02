# V1 冻结基线：`91244c1`

- 冻结日期：2026-08-02
- 分支：`main`
- 提交：`91244c1 The lastest change`
- 用途：V1 行为回归、V1/V2 效率比较、物理轨迹参考。

## 改动前验证

以下命令均在 `/Users/w-mone-y/Downloads/机械臂跟随控制` 执行并通过：

```bash
python -m pytest tests/v1/test_physical_task_projection.py \
  tests/v1/test_async_physical_regressions.py \
  tests/v1/test_simulation_speed.py
# 30 tests

python -m pytest tests/v2/test_v2_physical_execution_gate.py \
  tests/v2/test_v2_dual_line_runtime.py \
  tests/v2/test_v2_planning_scheduling.py \
  tests/shared/test_capability_routing.py
# 74 tests

python -m pytest tests/v2/test_v2_dual_line_scene.py \
  tests/v2/test_v2_scene_adapter.py
# 74 tests
```

## 冻结约束

- V1 的 A/B/C、1～3 件、三层炉批、故障恢复、暂停/继续/reset 和倍速结果不得改变。
- V2 迁移不得修改 V1 MJCF 名称、V1 actor 公共调用方式或 V1 入口默认行为。
- 若必须修复 V1 缺陷，应先增加能复现缺陷的回归测试，并在变更记录中说明原因。


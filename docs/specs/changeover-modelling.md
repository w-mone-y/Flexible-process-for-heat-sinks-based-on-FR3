# 换型建模、序列相关设置时间与 KPI（步骤 D）

对应赛题痛点 1（生产任务切换需大量人工重新示教或编程，准备时间长）与
评选标准 2 点名的关键指标 **换型时间缩短比例**。

## Status

已交付并纳入回归：`tests/shared/test_changeover.py` 共 28 项。
默认关闭（`track_changeover=False`），因此既有 V1/V2 行为逐位不变。

## 一个必须先纠正的事实

原计划假设「存在一条无条件的 7 步换型链需要优化」。实测发现并非如此：

1. 那条 7 步链曾位于 `task_graph_builder._decorate_flexible_cell`，但该方法
   **从未被调用**；实际产线只走 `_decorate_async_line`。2026-07-31 仓库清理已删除
   这 310 行不可达旧实现，避免它继续造成“已有实体换型”的误解。
2. 因此实际产线**完全不产生换型任务**。`CONFIGURE_COMB` 是每件固定 2.0 s，
   与梳齿是否真的需要更换无关。
3. `ManufacturingRuntime` 不跟踪跨订单的已装夹具，无法区分
   「同梳齿可跳过」与「异梳齿需更换」。

即：A/B/C 需要 20/30/15 mm 三种不同梳齿，但系统从不为切换付出任何时间。
所以步骤 D 的任务是**补上缺失的换型建模**，而不是优化已有链条。

## Solution

### D-1 配置状态向量与最小动作集

`brazing_sim/changeover/config_diff.py`

```python
FixtureConfiguration(mold, comb, press, tool, program)
```

换型动作由 `目标 − 当前` 差分导出，而非手写序列：

| 场景 | 动作数 | 说明 |
|---|---:|---|
| 冷线 → A | 7 | 只装不拆 |
| A → A | **0** | 同族批量收益的物理来源 |
| A → B | 10 | 拆 3 装 6 验 1 |
| 仅换压梁 | 2 | 模具与梳齿不动 |

拆装顺序遵循物理必然性：拆卸自上而下（press→comb→mold），
安装自下而上（mold→comb→press）—— 压梁装上后无法抽出梳齿。
被改动槽位**之上**的模块即使自身不变也必须先拆下，否则够不到。

`program`（产品:配方）与产品一同变化，但**不构成物理换型**。
这正是"过去需要重新示教、现在不需要"的部分：切程序是纯数据操作。

### D-2 序列相关换型时间矩阵

`brazing_sim/changeover/setup_matrix.py`

`setup_time[from_signature][to_signature]` 进入调度成本，问题从 FJSP 升级为
**FJSP-SDST**（sequence-dependent setup times）：某订单的代价取决于**它前面跑了什么**。

`SchedulingWeights.product_changeover_cost` 这个权重**早已存在但从无人写入**，
即换型此前对调度器完全免费。现在由
`ManufacturingRuntime._annotate_changeover_cost()` 在每个 tick 填充。

实测（6 件混流订单）：

| 排序 | 换型动作 | 换型时间 |
|---|---:|---:|
| 按到达顺序 A B A B C C | 47 | 114.0 s |
| 同族批量 A A B B C C | 27 | **65.0 s** |

**节省 43.0%** —— 调度器一旦看得见换型成本，会自发这样排。

### D-3 KPI 三件套与三档基线

`brazing_sim/changeover/metrics.py`

- `changeover_seconds`
- `changeover_count`（**有效**换型次数，配置未变的不计）
- `changeover_ratio_vs_baseline` ← 赛题点名指标

三档基线把"自动化的功劳"与"排序的功劳"分开，避免用一个数字冒领两份贡献：

| 档 | 6 件混流实测 | 有效换型 |
|---|---:|---:|
| 人工示教基线 | 720.0 s | 6 |
| 自动换型（按到达顺序） | 114.0 s | 5 |
| 自动换型 ＋ 同族批量排序 | **65.0 s** | 3 |

改善：仅自动化 **84.2%**，自动化＋排序 **91.0%**，仅排序贡献 **43.0%**。

## 两个诚实性设计（务必保留）

### 时基必须统一

第一版实现给出「98.9% 改善」——**这是错的**。它把真实的 30 分钟人工示教
除以本仿真压缩后的 16 秒换型，两者不在同一时基上，比出来的是演示压缩比
而不是自动化收益。

`TeachingBaseline.demo_scale` 负责折算（本线约 1/15），
`comparable` 字段暴露该比较是否合法。修正后为 84.2% / 91.0%，可以站住。

> **写报告时优先用 `sequencing_only_ratio`（43.0%）。**
> 它是两个仿真数之比，与人工基线和演示时基都无关，是三个数里最稳健的。

### 基线不能自己拍

`TeachingBaseline` 强制声明 `source`，并用 `measured` 区分实测与估计。
`PLACEHOLDER_TEACHING_BASELINE` 明确标注 `measured=False`，
所有输出都带 `baseline_is_placeholder: true`。

赛题第九条第 1、2 款提供现场调研与 CTO 牵头的技术导师组答疑 ——
**`T_teach` 必须从企业问出来**。它是"缩短比例"的分母，自己估的数会被评委问住。
拿到实测值后替换数字并置 `measured=True`。

## 交叉校验

换型时间由两条独立路径测量，必须一致：

1. 运行时日志 `runtime.changeover_log`（构图时记录）
2. 任务图 `changeover_seconds_from_graph()`（事后扫描）

实测两者相等（89.5 s / 37 个节点）。这条校验抓到了一个真实缺陷：
`REMOVE_OLD_COMB` / `REMOVE_OLD_PRESS` 同时用于**换型拆卸**与**焊后拆解**，
早期实现把焊后拆解误计为换型时间（多算 8 个节点 / 22 s）。
现由 `is_changeover_task()` 依据 `changeover_slot` 与 `after_brazing` 标记区分，
不靠任务类型猜测。

## 边界

- 换型动作目前只影响任务图与 KPI；**MuJoCo 中龙门的可见拆装动作尚未接入**
  （`CHANGEOVER_GANTRY` 资源与场景模块库已存在，执行层适配待做）。
- `best_sequence_cost()` 是贪心同族批量，**不是证明最优** ——
  它对应"设置感知调度器的实际行为"，作为对照点合适，但不应称为最优解。
- 换型成本已进入调度成本函数，但滚动时域优化（步骤 E）尚未接入，
  当前仍是单步贪心。
- 三档基线的第一档依赖占位数据，见上文。

## 验证

```bash
python -m pytest tests/shared/test_changeover.py -q     # 28 passed
python -m pytest tests/shared tests/v1 -q               # 回归

# 亲眼看序列相关性
python -c "
from dataclasses import replace
from brazing_sim.flexible import build_preset_plan
from brazing_sim.manufacturing_runtime import ManufacturingRuntime
def total(seq):
    rt = ManufacturingRuntime(flexible_cell=True, track_changeover=True)
    for i,p in enumerate(seq):
        b = build_preset_plan(p, quantity=1)
        rt.submit_plan(replace(b, order=replace(b.order, order_id=f'O{i}_{p}')), now=0.0)
    return sum(c['nominal_seconds'] for c in rt.changeover_log)
a, b = total('ABABCC'), total('AABBCC')
print('到达顺序 %.1fs / 同族批量 %.1fs / 节省 %.1f%%' % (a, b, 100*(a-b)/a))
"
```

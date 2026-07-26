# Arm2 在整体流程中的任务

## Arm2 在整体流程中的位置

完整生产流程可以概括为：

```text
Arm1：
基板上料
→ 翅片逐片上料
→ 完成散热组件组坯

Arm3：
焊前几何检测
→ 检查基板位置、翅片数量、间距、垂直度和插入深度

Arm2：
钎料布置
→ 夹具锁紧
→ 装炉
→ 执行虚拟钎焊流程
→ 出炉转运

Arm3：
焊后检测
→ 合格下线 / 不合格返工
```

因此，Arm2 的工作开始条件是：

```text
Arm1 组坯完成
+
Arm3 焊前检测通过
```

只有当基板和翅片的几何状态满足要求后，Arm2 才进入钎焊工艺执行阶段。

------

## 3. Arm2 的完整任务链

Arm2 的完整任务链为：

```text
等待焊前检测通过
→ 获取钎料涂覆工具
→ 根据订单生成钎料路径
→ 沿翅片根部布置钎料
→ 等待 Arm3 钎料检测
→ 对不合格区域进行补涂
→ 锁紧柔性夹具
→ 更换托盘搬运工具
→ 将工装送入钎焊炉
→ 启动虚拟钎焊工艺
→ 等待钎焊炉完成预热、升温、保温和冷却
→ 将工装从炉内取出
→ 转运至 Table3 检测工位
→ 等待 Arm3 焊后检测
→ 必要时执行返工
```

------

## 4. 任务一：等待焊前检测通过

Arm1 完成基板与翅片放置后，散热组件处于组坯状态：

```text
preassembled
```

此时 Arm2 不应立即开始布置钎料，而应等待 Arm3 对组坯结果进行焊前检测。

Arm3 需要检查：

- 基板是否放置到位；
- 基板是否与夹具定位基准对齐；
- 翅片数量是否正确；
- 翅片是否漏装；
- 翅片是否插错槽位；
- 翅片间距是否符合订单要求；
- 翅片垂直度是否合格；
- 翅片底部是否与基板贴合。

检测通过后，组件状态更新为：

```text
preassembly_passed
```

检测失败时，系统生成返工任务，由 Arm1 重新调整基板或翅片。

------

## 5. 任务二：获取钎料涂覆工具

焊前检测通过后，Arm2 前往工具架获取钎料涂覆工具。

工具名称建议为：

```text
arm2_brazing_dispenser_tool
```

该工具可在仿真中表示为一个简化喷嘴、涂覆头或钎料片放置头。

工具挂载流程：

```text
Arm2 移动到工具架等待点
→ 对准钎料涂覆工具
→ 解除工具与工具架之间的 weld
→ 激活工具与 Arm2 末端之间的 weld
→ 更新 current_tool = "brazing_dispenser"
```

------

## 6. 任务三：根据订单生成钎料路径

不同订单对应不同散热组件结构，因此钎料路径也应自动生成。

若每片翅片左右两侧各布置一条钎料路径，则：

```text
钎料路径数量 = 翅片数量 × 2
```

例如 4 片翅片时，路径包括：

```text
fin_01_left_path
fin_01_right_path
fin_02_left_path
fin_02_right_path
fin_03_left_path
fin_03_right_path
fin_04_left_path
fin_04_right_path
```

路径应由订单参数自动计算，包括：

- 路径起点；
- 路径终点；
- 路径方向；
- 涂覆高度；
- 涂覆宽度；
- 目标覆盖率；
- 对应翅片编号；
- 对应基板坐标系。

建议数据结构：

```python
@dataclass
class BrazingPath:
    path_id: str
    fin_id: str
    side: str
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    target_width: float
    target_coverage: float
    status: str = "pending"
```

------

## 7. 任务四：沿翅片根部布置钎料

Arm2 使用钎料涂覆工具沿每条路径移动。

单条路径的动作流程为：

```text
移动到钎料路径起点上方
→ 下降到涂覆高度
→ 打开虚拟涂覆状态
→ 沿翅片根部匀速移动
→ 生成或显示钎料轨迹
→ 关闭虚拟涂覆状态
→ 抬起到安全高度
```

在 MuJoCo 中，钎料布置可以通过以下方式表示：

1. 在翅片根部显示细长钎料 geom；
2. 更新钎料路径状态；
3. 记录涂覆长度和覆盖率；
4. 在日志中输出当前路径完成情况。

路径状态变化：

```text
pending
→ dispensing
→ applied
```

------

## 8. 任务五：等待 Arm3 钎料检测

Arm2 完成钎料布置后，不应立即装炉，而应等待 Arm3 进行钎料检测。

Arm3 需要检查：

- 钎料路径是否完整；
- 是否存在漏涂；
- 是否存在断续；
- 钎料是否偏离翅片根部；
- 钎料覆盖率是否达到阈值；
- 是否存在局部堆料。

检测通过后：

```text
brazing_material_passed
```

检测失败时：

```text
brazing_material_failed
```

系统生成补涂任务，Arm2 根据缺陷位置重新执行局部钎料布置。

------

## 9. 任务六：局部补涂返工

当 Arm3 检测到钎料缺陷时，Arm2 执行补涂任务。

补涂流程：

```text
接收缺陷路径或缺陷区间
→ 移动到缺陷区域起点
→ 下降到涂覆高度
→ 对缺陷区域补涂钎料
→ 更新该路径覆盖率
→ 等待 Arm3 复检
```

常见缺陷类型包括：

```text
missing_material
insufficient_coverage
path_discontinuous
material_offset
material_overflow
```

------

## 10. 任务七：锁紧柔性夹具

钎料检测通过后，Arm2 触发 Table2 上的柔性夹具锁紧机构。

夹具锁紧的目的包括：

- 固定基板位置；
- 固定每片翅片位置；
- 防止装炉搬运过程中翅片倾倒；
- 保证钎焊过程中翅片与基板保持贴合；
- 保持翅片间距和垂直度。

在 MuJoCo 中，可通过 equality weld 或状态变量模拟：

```text
base_to_fixture_weld = active
fin_01_to_fixture_weld = active
fin_02_to_fixture_weld = active
...
fixture_lock_state = locked
```

状态变化：

```text
brazing_material_passed
→ fixture_locked
```

------

## 11. 任务八：更换托盘搬运工具

钎料涂覆工具不适合搬运整个工装托盘，因此 Arm2 需要更换为托盘搬运工具。

工具名称建议为：

```text
arm2_tray_transfer_tool
```

工具切换流程：

```text
Arm2 返回工具架
→ 归还 arm2_brazing_dispenser_tool
→ 对准 arm2_tray_transfer_tool
→ 激活托盘搬运工具挂载 weld
→ 更新 current_tool = "tray_transfer"
```

------

## 12. 任务九：将工装送入钎焊炉

Arm2 使用托盘搬运工具抓取装有散热组件的工装托盘，并将其送入钎焊炉。

动作流程：

```text
移动到 Table2 工装托盘接近点
→ 对准托盘抓取点
→ 激活 tray_grasp_weld
→ 抬升或推送托盘
→ 移动到钎焊炉入口
→ 将托盘送入炉腔目标位置
→ 释放托盘
→ 撤离炉口区域
```

状态变化：

```text
fixture_locked
→ in_furnace
```

此阶段需要占用共享资源：

```text
furnace_loading_zone
```

------

## 13. 任务十：启动虚拟钎焊工艺

工装进入钎焊炉后，Arm2 触发炉门关闭和钎焊工艺循环。

钎焊炉状态机建议为：

```text
IDLE
→ LOADING
→ DOOR_CLOSING
→ PREHEAT
→ HEATING
→ SOAKING
→ COOLING
→ DOOR_OPENING
→ COMPLETE
```

MuJoCo 中不模拟真实热传导，可通过以下变量表达：

```python
virtual_temperature
brazing_timer
recipe_state
furnace_door_state
```

组件状态变化：

```text
in_furnace
→ brazing
→ cooling
→ brazed
```

------

## 14. 任务十一：出炉并转运到 Table3

钎焊炉完成虚拟工艺循环后，Arm2 负责出炉和转运。

动作流程：

```text
等待炉门打开
→ 移动到炉口等待点
→ 对准炉内托盘
→ 激活 tray_grasp_weld
→ 将工装从炉内取出
→ 移动到 Table3 冷却与检测工位
→ 放置托盘
→ 释放托盘
→ 退回安全等待位
```

状态变化：

```text
brazed
→ waiting_for_post_inspection
```

随后 Arm3 开始焊后质量检测。

------

## 15. 任务十二：根据焊后检测执行返工

Arm3 焊后检测可能发现：

- 翅片倾斜；
- 基板翘曲；
- 局部钎焊连接不足；
- 钎料覆盖不足；
- 钎料溢出；
- 局部脱焊。

若缺陷可返工，Arm2 执行：

```text
补涂钎料
→ 重新锁紧夹具
→ 二次装炉
→ 二次钎焊
→ 出炉复检
```

若缺陷不可返工，则系统将产品标记为：

```text
SCRAPPED
或
MANUAL_REVIEW
```

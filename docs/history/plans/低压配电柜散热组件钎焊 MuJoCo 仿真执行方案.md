# 低压配电柜散热组件钎焊 MuJoCo 仿真执行方案

## 一、项目定位

本项目面向低压配电柜散热组件的柔性化生产过程，仿真对象为：

> **薄板散热翅片 + 散热基板 + 钎料 + 柔性定位夹具组成的散热组件。**

在真实生产中，散热组件通常需要完成基板上料、翅片定位、钎料布置、夹具压紧、炉中钎焊、冷却检测等工序。MuJoCo 不适合真实模拟金属熔化、钎料润湿、扩散和热传导，因此本方案采用：

> **几何运动仿真 + 工艺状态机 + 虚拟钎焊炉 + 质量状态变量 + 检测与返工逻辑**

来表达低压配电柜散热组件的柔性制造过程。

项目最终目标是实现：

```text
订单输入
→ 自动生成散热片结构参数
→ Arm1 完成基板和翅片组坯
→ Arm3 执行焊前检测
→ Arm2 布置钎料并完成装炉
→ 钎焊炉执行虚拟钎焊循环
→ Arm2 出炉转运
→ Arm3 焊后检测
→ 合格下线 / 缺陷返工
```

------

## 二、产品结构假设

散热组件由以下部分组成：

```text
散热组件
├── 散热基板 base_plate
├── 多片薄板翅片 fin_01 ~ fin_N
├── 钎料 brazing_material
├── 柔性定位夹具 fixture
└── 可选安装孔、安装耳、加强板
```

### 1. 散热基板

基板是一块矩形金属薄板或中厚板，作为翅片安装基础。

仿真初始尺寸可设为：

```text
长度：0.36 m
宽度：0.22 m
厚度：0.008 m
```

在 MuJoCo 中可用 box geom 表示。

### 2. 单片翅片

翅片是薄而长的矩形板，竖直安装在基板上。

仿真初始尺寸可设为：

```text
长度：0.30 m
高度：0.06 m
厚度：0.002 m
```

多个翅片平行排列在基板上，形成类似“金属梳子”的结构：

```text
侧视图：

│ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │  ← 薄板翅片
│ │ │ │ │ │ │ │
════════════════  ← 散热基板
```

### 3. 钎料

钎料可以抽象为：

- 钎料膏；
- 钎料片；
- 钎料带；
- 预置钎料层。

在 MuJoCo 中不需要真实模拟钎料流动，可以通过细长 geom 或状态变量表示：

```text
钎料状态：
pending
→ applied
→ inspected
→ accepted / rework_required
→ brazed
```

------

## 三、订单变化设计

为了体现柔性生产，设置不同订单型号。

| 订单 | 基板尺寸 | 翅片数量 | 翅片间距 | 钎料路径   |
| ---- | -------- | -------- | -------- | ---------- |
| A 型 | 小       | 4 片     | 大       | 4 或 8 条  |
| B 型 | 中       | 6 片     | 中       | 6 或 12 条 |
| C 型 | 大       | 8 片     | 小       | 8 或 16 条 |

订单变化会导致：

```text
翅片数量变化
→ Arm1 抓取和插入次数变化

翅片间距变化
→ 夹具槽位和放置目标变化

基板尺寸变化
→ 基板抓取点和定位点变化

钎料路径变化
→ Arm2 涂覆路径变化

产品几何变化
→ Arm3 检测路径变化
```

这样可以体现比赛题目中的“任务变化感知、自主决策、路径规划和协同调度”。

------

## 四、产线布局

建议采用四个区域：

```text
┌────────────────────────────────────┐
│                                    │
│  Table1：原料区                    │
│  基板托盘、翅片料架                │
│            │                       │
│            ▼                       │
│  Table2：组坯与钎料布置工位         │
│  柔性定位夹具、钎料布置区域         │
│            │                       │
│            ▼                       │
│  Brazing Furnace：钎焊炉            │
│  虚拟升温、保温、冷却               │
│            │                       │
│            ▼                       │
│  Table3：冷却与检测工位             │
│  焊后检测、返工判断、成品下线       │
│                                    │
└────────────────────────────────────┘
```

三台机械臂围绕这几个区域工作：

```text
Arm1：靠近 Table1 和 Table2
Arm2：靠近 Table2 和钎焊炉
Arm3：靠近 Table2、Table3 和检测区
```

------

## 五、三台机械臂任务分工

## 1. Arm1：基板与翅片柔性上料组坯机械臂

Arm1 的核心任务不是普通搬运，而是：

> **根据订单完成散热基板和薄板翅片的自动组坯。**

具体任务包括：

1. 读取当前订单；
2. 从 Table1 抓取对应规格基板；
3. 将基板放入 Table2 柔性定位夹具；
4. 激活基板定位挡块；
5. 从翅片料架逐片抓取翅片；
6. 根据订单生成翅片目标槽位；
7. 将翅片插入定位梳齿或基板槽位；
8. 判断翅片是否插到位；
9. 完成所有翅片排列；
10. 请求 Arm3 进行焊前检测。

Arm1 技能函数建议：

```python
arm1_pick_base_plate(order_id)
arm1_place_base_plate_to_fixture(order_id)
arm1_pick_fin(fin_id)
arm1_place_fin_to_slot(fin_id, slot_id)
arm1_adjust_fin(fin_id)
arm1_verify_preassembly()
arm1_retreat_to_wait()
```

Arm1 动作流程：

```text
等待位
→ 原料区接近点
→ 抓取基板/翅片
→ 抬升
→ 移动到组坯夹具
→ 预插入位姿
→ 插入定位槽
→ 释放
→ 撤离
```

Arm1 末端工具建议：

```text
方案一：平行夹爪
适合抓取翅片边缘和基板夹持耳。

方案二：真空吸盘 + 窄口夹爪复合工具
吸盘抓基板，夹爪抓翅片。

方案三：工具快换
基板吸盘工具 + 翅片夹持工具。
```

第一阶段建议采用简化平行夹爪，降低实现难度。

------

## 2. Arm2：钎料布置、夹具锁紧、装炉与出炉机械臂

由于采用炉中钎焊，Arm2 不再拿焊枪沿每条翅片焊接，而是负责钎焊前后的工艺执行。

Arm2 任务包括：

1. 获取钎料涂覆工具；
2. 根据订单生成钎料路径；
3. 沿翅片根部布置钎料；
4. 等待 Arm3 检查钎料覆盖情况；
5. 对漏涂位置执行补涂；
6. 触发夹具锁紧；
7. 获取托盘搬运工具；
8. 将整套工装送入钎焊炉；
9. 启动虚拟钎焊工艺；
10. 钎焊完成后取出工件；
11. 转运到 Table3 检测区；
12. 根据检测结果执行补涂、二次钎焊或返工。

Arm2 技能函数建议：

```python
arm2_get_tool(tool_name)
arm2_return_current_tool()
arm2_apply_brazing_material(path_id)
arm2_verify_brazing_path(path_id)
arm2_lock_fixture()
arm2_transfer_fixture_to_furnace()
arm2_start_brazing_cycle(recipe_id)
arm2_remove_fixture_from_furnace()
arm2_transfer_to_inspection_station()
arm2_execute_rework(rework_task)
```

Arm2 推荐工具：

```text
工具 A：钎料涂覆头
用于模拟钎料膏、钎料带或钎料片布置。

工具 B：托盘搬运夹具
用于搬运夹具托盘和散热组件。

工具 C：备用补涂工具
用于检测失败后的局部返工。
```

第一阶段可以只实现两个工具：

```text
arm2_brazing_dispenser_tool
arm2_tray_transfer_tool
```

------

## 3. Arm3：焊前检测、钎料检测与焊后检测机械臂

Arm3 负责质量检查和返工决策。

### 焊前检测

检查内容：

- 基板是否放置到位；
- 翅片数量是否正确；
- 翅片是否漏装；
- 翅片是否放错槽；
- 翅片间距是否正确；
- 翅片是否垂直；
- 翅片底部是否贴合基板；
- 夹具是否正常打开或锁紧。

### 钎料检测

检查内容：

- 钎料路径是否完整；
- 钎料覆盖率是否达标；
- 是否存在漏涂；
- 是否存在局部堆料；
- 钎料是否涂错位置。

### 焊后检测

检查内容：

- 翅片是否倾斜；
- 基板是否翘曲；
- 翅片与基板连接状态是否合格；
- 是否存在局部未连接；
- 是否存在钎料溢出；
- 整体尺寸是否合格。

Arm3 技能函数建议：

```python
arm3_inspect_preassembly(order_id)
arm3_inspect_fin_geometry(fin_id)
arm3_inspect_brazing_material(path_id)
arm3_inspect_post_brazing(order_id)
arm3_generate_rework_task()
arm3_verify_rework()
```

------

## 六、钎焊炉仿真设计

MuJoCo 中建立一个虚拟钎焊炉对象：

```text
brazing_furnace
brazing_furnace_door
brazing_furnace_chamber
brazing_furnace_entry_site
brazing_furnace_tray_site
brazing_furnace_exit_site
```

炉门可以用 slide joint 或 hinge joint 表示。

钎焊炉状态机：

```text
IDLE
→ LOADING
→ DOOR_CLOSING
→ PREHEAT
→ HEATING
→ SOAKING
→ COOLING
→ DOOR_OPENING
→ UNLOADING
→ COMPLETE
```

虚拟温度曲线：

```text
室温
→ 预热
→ 升温
→ 保温
→ 冷却
→ 出炉
```

Python 中可定义：

```python
@dataclass
class BrazingRecipe:
    recipe_id: str
    preheat_duration: float
    heating_duration: float
    soaking_duration: float
    cooling_duration: float
    target_temperature: float
    max_temperature_error: float
```

钎焊完成判定条件：

1. 所有翅片均已安装；
2. 所有钎料路径已完成；
3. 钎料覆盖率达标；
4. 夹具已锁紧；
5. 工件已进入炉内；
6. 炉门已关闭；
7. 升温、保温、冷却过程完成；
8. 未发生设备异常。

组件状态变化：

```text
preassembled
→ brazing_material_applied
→ fixture_locked
→ in_furnace
→ brazing
→ cooling
→ brazed
→ waiting_for_inspection
```

------

## 七、MuJoCo 对象命名建议

### 产品对象

```text
heatsink_base_plate
heatsink_fin_01
heatsink_fin_02
heatsink_fin_03
heatsink_fin_04
heatsink_assembly_frame
```

### 夹具对象

```text
fixture_base
fixture_base_locator
fixture_fin_comb
fixture_side_clamp_left
fixture_side_clamp_right
fixture_top_clamp
fixture_tray
fixture_reference_site
```

### Arm1 相关 site

```text
arm1_wait_site
arm1_table1_approach_site
arm1_fixture_approach_site
arm1_base_grasp_site
arm1_fin_grasp_site
arm1_fin_insert_site
```

### Arm2 相关 site

```text
arm2_tool_changer_master
arm2_brazing_dispenser_tool
arm2_dispenser_tcp
arm2_tray_transfer_tool
arm2_tray_grasp_site
arm2_furnace_wait_site
arm2_furnace_load_site
arm2_furnace_unload_site
```

### Arm3 相关 site

```text
arm3_camera_body
arm3_camera
arm3_camera_optical_site
arm3_top_overview_site
arm3_fin_end_view_site
arm3_brazing_path_scan_site
arm3_post_braze_side_view_site
arm3_wait_site
```

------

## 八、抓取与固定逻辑

第一阶段建议使用 equality weld 模拟抓取、释放和工装固定。

### Arm1 抓取基板

```text
Arm1 到达基板抓取位姿
→ 判断距离和姿态误差
→ 激活 arm1_base_grasp_weld
→ 基板跟随 Arm1
→ 到达夹具
→ 关闭 arm1_base_grasp_weld
→ 激活 base_to_fixture_weld
```

### Arm1 抓取翅片

```text
Arm1 到达翅片抓取位姿
→ 激活 arm1_fin_grasp_weld
→ 搬运到目标槽位
→ 插入夹具
→ 关闭 arm1_fin_grasp_weld
→ 激活 fin_to_fixture_weld
```

### 工装锁紧

当所有翅片安装并检测通过后：

```text
fixture_lock_state = locked
fin_to_fixture_welds = active
base_to_fixture_weld = active
```

这样在搬运和装炉过程中，组件不会散开。

------

## 九、钎料路径生成

每片翅片至少对应一条钎料路径。

简化方案：

```text
每片翅片根部一条钎料路径
```

更合理方案：

```text
每片翅片左右两侧各一条钎料路径
```

路径数据结构：

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

状态变化：

```text
pending
→ dispensing
→ applied
→ inspected
→ accepted / rework_required
```

Arm2 的涂覆动作可以表现为：

```text
移动到钎料路径起点
→ 下降到涂覆高度
→ 沿翅片根部匀速运动
→ 显示钎料轨迹
→ 更新路径状态为 applied
```

------

## 十、检测视角设计

### 1. 顶部全局视角

相机从正上方俯视整个散热组件。

应拍到：

- 完整基板；
- 全部翅片；
- 翅片排列区域；
- 定位夹具；
- 基板边界。

主要检查：

- 是否漏装翅片；
- 翅片数量是否正确；
- 翅片是否错位；
- 翅片间距是否明显异常。

------

### 2. 翅片端面视角

相机沿翅片长度方向观察。

应拍到：

- 所有翅片端面；
- 翅片高度；
- 翅片垂直度；
- 翅片间距；
- 顶边是否齐平。

主要检查：

- 翅片是否倾斜；
- 是否插入不到位；
- 是否存在局部弯曲；
- 间距是否均匀。

------

### 3. 钎料路径斜视

相机以约 30°～45° 角观察翅片根部。

应拍到：

- 翅片根部；
- 基板表面；
- 钎料路径；
- 钎料连续性；
- 漏涂或堆料区域。

主要检查：

- 钎料覆盖率；
- 钎料是否断续；
- 是否有局部缺失；
- 是否涂到错误位置。

------

### 4. 焊后侧视

相机从侧面观察整体结构。

应拍到：

- 基板轮廓；
- 翅片侧面；
- 翅片与基板接合区；
- 翅片顶部高度；
- 基板是否翘曲。

主要检查：

- 焊后变形；
- 翅片倾斜；
- 局部脱焊；
- 钎料溢出。

------

## 十一、检测判定模型

第一阶段不做真实视觉识别，使用 MuJoCo 真值和状态变量。

### 翅片位置误差

```python
position_error = norm(actual_pos - target_pos)
```

### 翅片垂直度误差

```python
angle_error = angle_between(actual_fin_axis, target_fin_axis)
```

### 钎料覆盖率

```python
coverage = applied_length / target_length
```

### 综合质量评分

```python
quality_score = (
    0.35 * material_coverage_score
    + 0.25 * geometry_score
    + 0.25 * temperature_profile_score
    + 0.15 * fixture_stability_score
)
```

判定标准：

```text
quality_score ≥ 0.90：PASS
0.75 ≤ quality_score < 0.90：REWORK
quality_score < 0.75：SCRAP
```

------

## 十二、返工逻辑

### 1. 翅片位置异常

```text
Arm3 检测到翅片位置异常
→ 生成 fin_adjustment_task
→ Arm1 重新调整翅片
→ Arm3 复检
```

### 2. 钎料覆盖不足

```text
Arm3 检测到钎料漏涂
→ 生成 brazing_material_rework
→ Arm2 获取涂覆工具
→ 局部补涂
→ Arm3 复检
```

### 3. 钎焊后局部未连接

```text
Arm3 检测到局部未连接
→ 判断是否可返工
→ 若可返工：Arm2 补涂钎料并二次钎焊
→ 若不可返工：标记 SCRAP 或 MANUAL_REVIEW
```

------

## 十三、共享区域与避碰

需要设置共享资源锁：

```python
shared_resources = {
    "assembly_fixture_zone": None,
    "brazing_material_zone": None,
    "furnace_loading_zone": None,
    "inspection_zone": None,
}
```

规则：

```text
Arm1 插入翅片时，Arm2 不得进入组坯区；
Arm3 焊前检测时，Arm1 必须撤离到等待位；
Arm2 涂覆钎料时，Arm3 不得进入钎料路径区域；
Arm2 装炉时，Arm1 和 Arm3 不得进入炉口区域；
Arm3 焊后检测时，Arm2 必须退出检测区。
```

资源申请接口：

```python
request_resource(robot_id, resource_id)
release_resource(robot_id, resource_id)
```

------

## 十四、状态机设计

### 订单状态

```text
CREATED
→ MATERIAL_PREPARATION
→ PREASSEMBLING
→ PREASSEMBLY_INSPECTION
→ BRAZING_MATERIAL_APPLICATION
→ BRAZING_MATERIAL_INSPECTION
→ FIXTURE_LOCKING
→ FURNACE_LOADING
→ BRAZING
→ COOLING
→ POST_BRAZING_INSPECTION
→ COMPLETED / REWORK_REQUIRED / SCRAPPED
```

### Arm1 状态

```text
IDLE
→ GET_BASE
→ PLACE_BASE
→ GET_FIN
→ PLACE_FIN
→ ADJUST_FIN
→ FIN_COMPLETE
→ WAIT_PREASSEMBLY_INSPECTION
→ REWORK_FIN
→ RETURN_HOME
→ ERROR
```

### Arm2 状态

```text
IDLE
→ GET_DISPENSER_TOOL
→ APPLY_BRAZING_MATERIAL
→ WAIT_MATERIAL_INSPECTION
→ REWORK_MATERIAL
→ RETURN_DISPENSER_TOOL
→ LOCK_FIXTURE
→ GET_TRANSFER_TOOL
→ LOAD_FURNACE
→ START_BRAZING
→ WAIT_BRAZING_COMPLETE
→ UNLOAD_FURNACE
→ TRANSFER_TO_INSPECTION
→ RETURN_TOOL
→ ERROR
```

### Arm3 状态

```text
IDLE
→ TOP_OVERVIEW
→ FIN_GEOMETRY_SCAN
→ BRAZING_MATERIAL_SCAN
→ WAIT_BRAZING
→ POST_BRAZE_OVERVIEW
→ FIN_DEFORMATION_SCAN
→ JOINT_SCAN
→ GENERATE_RESULT
→ VERIFY_REWORK
→ RETURN_HOME
→ ERROR
```

------

## 十五、最小可运行版本

第一阶段不追求复杂，先做一个最小闭环。

### 场景对象

```text
基板：1 块
翅片：4 片
钎料路径：4 条或 8 条
夹具：1 套
钎焊炉：1 台
订单：1 个
三台 FR3：Arm1、Arm2、Arm3
```

### 执行流程

```text
1. 初始化场景
2. Arm1 抓取基板
3. Arm1 将基板放入夹具
4. Arm1 依次抓取 4 片翅片
5. Arm1 将翅片插入 4 个槽位
6. Arm3 检查翅片位置和垂直度
7. Arm2 获取钎料涂覆工具
8. Arm2 沿翅片根部布置钎料
9. Arm3 检查钎料覆盖情况
10. 夹具锁紧
11. Arm2 获取托盘搬运工具
12. Arm2 将工装送入钎焊炉
13. 炉门关闭
14. 执行虚拟预热、升温、保温和冷却
15. 炉门打开
16. Arm2 取出工装
17. Arm2 转运到 Table3
18. Arm3 执行焊后检测
19. 输出 PASS 或 REWORK
20. 输出本订单 KPI
```

------

## 十六、推荐代码结构

```text
project/
├── base_develop.py
├── base_develop.xml
├── controllers/
│   ├── arm1_controller.py
│   ├── arm2_controller.py
│   └── arm3_controller.py
├── skills/
│   ├── arm1_skills.py
│   ├── arm2_skills.py
│   └── arm3_skills.py
├── process/
│   ├── order_manager.py
│   ├── task_scheduler.py
│   ├── brazing_process.py
│   ├── furnace_controller.py
│   └── inspection_manager.py
├── managers/
│   ├── object_manager.py
│   ├── tool_manager.py
│   ├── fixture_manager.py
│   └── resource_manager.py
├── models/
│   ├── order.py
│   ├── task.py
│   ├── product_state.py
│   └── inspection_result.py
└── config/
    ├── orders.yaml
    ├── brazing_recipes.yaml
    └── inspection_profiles.yaml
```

------

## 十七、KPI 指标

建议记录：

```text
总生产节拍
单件组坯时间
单件钎料布置时间
装炉时间
出炉时间
炉利用率
Arm1 利用率
Arm2 利用率
Arm3 利用率
机械臂等待时间
共享区域冲突次数
订单切换时间
一次合格率
返工率
报废率
平均检测时间
```

------

## 十八、第一版实现顺序

```text
阶段 1：建立基板、4 片翅片和组坯夹具
阶段 2：实现 Arm1 基板与翅片搬运
阶段 3：实现 Arm3 焊前真值检测
阶段 4：实现 Arm2 钎料路径动画与状态更新
阶段 5：实现夹具锁紧和托盘搬运
阶段 6：实现钎焊炉门和虚拟工艺周期
阶段 7：实现 Arm3 焊后检测
阶段 8：实现缺陷注入与返工
阶段 9：增加多订单和动态调度
阶段 10：统计 KPI 并完成演示界面
```

------

## 十九、最终展示效果

最终在 MuJoCo 中应展示：

```text
动态订单输入
→ 自动生成不同数量和间距的翅片槽位
→ Arm1 完成基板和翅片柔性组坯
→ Arm3 完成焊前检测
→ Arm2 自动布置钎料
→ Arm3 检查钎料质量
→ Arm2 锁紧工装并装炉
→ 钎焊炉自动完成工艺循环
→ Arm2 出炉并转运
→ Arm3 焊后检测
→ 检测失败自动返工
→ 输出产品状态和产线 KPI
```

本方案的核心亮点是：

> **把低压配电柜散热组件的钎焊生产过程，抽象成可在 MuJoCo 中执行的多机械臂柔性制造闭环。**

它能够突出：

- 多规格订单切换；
- Arm1 柔性上料组坯；
- Arm2 钎料布置与炉前物流；
- Arm3 检测与返工决策；
- 钎焊炉状态机；
- 缺陷注入与质量闭环；
- 多机械臂协同调度与效能优化。
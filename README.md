# 低压配电柜散热组件钎焊仿真

基于 **MuJoCo** 和三台 **Franka Research 3 (FR3)** 的散热组件柔性制造 MVP。系统用几何运动、工艺状态机、虚拟钎焊炉和真值质量检测表达完整生产闭环，不模拟真实金属熔化、润湿或传热。

> 当前主线：`brazing_line.py` / `brazing_line.xml`
> 旧电气板装配产线仍保留；原说明见 [legacy 文档](docs/legacy_electrical_board.md)。

## MVP 工艺

```text
订单 A / B / C
→ Arm1 用吸盘把纯长方体基板放入可搬运工装托盘
→ Arm2 在无梳齿、无上压梁的裸基板上预涂全部钎焊材料
→ Arm3 材料检测
→ 漏涂时 Arm2 自动局部补涂并复检
→ 材料检测通过后安装成对的前、后梳齿模块
→ Arm1 逐片插入翅片，前后端同时进入梳齿槽
→ Arm3 翅片几何检测；不合格时 Arm1 自动调整并复检
→ 所有翅片完成后安装双短压梁，执行接触搜索 / 力爬升 / 订单力保压
→ equality weld 逻辑锁紧，进入 READY_FOR_TRANSFER
→ Table2 前向传送带把整套托盘连续送入炉内
→ 炉内停留 10 秒（仿真时钟）
→ 传送带把整套托盘连续送回 Table2 原位
→ Arm3 焊后检测
→ PASS / REWORK_REQUIRED / SCRAPPED
```

A/B/C 柔性订单配置：

| 订单 | 翅片数量 | 装配节距 | 前后梳齿模块 | 双侧材料线 |
|---|---:|---:|---|---:|
| A | 5 | 0.020 m | 20 mm | 10 |
| B | 4 | 0.030 m | 30 mm | 8 |
| C | 7 | 0.015 m | 15 mm | 14 |

其中 A 使用 `0.36 × 0.22 × 0.008 m` 基板，B 使用
`0.36 × 0.24 × 0.008 m`，C 使用 `0.34 × 0.20 × 0.008 m`。
基板 body 下只保留一个纯长方体 geom，没有安装耳、孔、边梁或装饰凸起。
路径按基板 X 边界各缩进 `0.015 m` 生成，A/B 单条长 `0.33 m`，C 单条长
`0.31 m`，目标宽度均为 `0.004 m`。

场景统一预分配 12 片翅片和 24 条路径，切换 A/B/C 或后续在容量内扩展规格时
无需重建 MuJoCo viewer。Arm2 双喷嘴中心距会在运行时调整：A/B 为 `5.0 mm`，
C 为 `4.4 mm`。

### YAML 柔性订单

`config/products/` 定义产品几何和工艺参数，`config/orders/` 定义数量、优先级和
层位偏好。加载器使用严格 YAML 模型：缺字段、未知字段、类型错误、负数、
超过 12/24 容量或料架无空层都会在机械臂运动前拒绝启动，并报出“文件 + 字段路径 + 原因”。

```bash
# 仅加载、生成和校验，不构建 MuJoCo 模型
python run_flexible_order.py --order config/orders/order_001.yaml --dry-run

# macOS 图形模式：普通 Python 入口会自动转交 mjpython
python run_flexible_order.py --order config/orders/order_002.yaml

# 无界面执行整个多件订单
python run_flexible_order.py --order config/orders/order_003.yaml --headless
```

场景视觉采用“简化碰撞体 + 高细节非碰撞外壳”分层：输送线包含密集辊床、铝型材
导轨、支腿、电机和传感器；炉体包含双层壳体、耐火炉口、加热元件、观察窗、排气和
控制柜；Arm2 双喷嘴包含法兰、阀体、料筒、软管、陶瓷套和金属喷嘴。新增细节不参与
质量、惯量或接触求解，相关选型与授权记录见
[视觉模型与资产说明](docs/视觉模型与资产说明.md)。

实时显示使用平衡质量配置：小圆柱保持 16 边平滑轮廓，四盏灯中只有主光源计算阴影，
阴影贴图为 2048，离屏抗锯齿为 2×。Arm3 相机在检测时默认 5 FPS、待机时 2 FPS；当
用户拖动、旋转或缩放 MuJoCo 主视角时，相机离屏渲染会让出图形线程，停止操作约
0.35 秒后自动恢复。因此不会通过隐藏模型或删除工业细节来换取流畅度。

### 多托盘批次模式

在不改变上述单件演示的前提下，YAML 订单可包含 1–3 件产品：

```text
托盘 1/2/3 依次在 Table2 完成基板、材料、翅片、检测和压紧
→ 短距离出料滑台
→ Z 向升降平台
→ 伸缩推叉
→ 按 ProcessPlan 分配的层位锁入炉内料架
→ 所有计划层位锁定且推叉/平台归零后才关炉门
→ 全批只执行一次 10 秒热循环
→ 按顶层、中层、底层退架
→ 三个成品缓存位
→ 逐套焊后分级
```

各套产品的状态、翅片、材料覆盖和检测结果相互独立。任一套进入
`MANUAL_REVIEW / ERROR` 时整批暂停。1/2 件订单只要其所有计划层位已锁定就可启炉，
不再要求未使用的空层被占用。退架始终按已占用物理层从高到低执行。

炉前推叉采用可见的两级伸缩造型：深色固定套筒留在升降台，橙色内臂、两根托叉和
带橡胶垫的后推梁随 `batch_pusher_joint` 连续伸出，托盘与推梁保持等速、等位移进入
料架。平台到层后先执行 `ALIGNING` 停稳，入架后再执行 `LOCKING`；每层独立的
25 mm 锁销通过真实 slide joint 滑入，红色指示灯转绿并确认锁定后推叉才允许回缩。
三层导轨各包含五根承托辊和两个黄色入口导轮，使托盘进入哪一层以及当前锁紧状态都
可以从主视角直接辨认。点击“单独运行升降入架”时，主 Viewer 会一次性切换到炉前
斜侧特写，随后仍可自由旋转和缩放，不会持续抢占用户视角。

## 环境

- Python 3.10+
- MuJoCo 3.1+
- NumPy、PyYAML
- 可选：PySide6（Qt 演示面板）

```bash
make install
make install-dev   # 同时安装 UI 与测试/格式化依赖
```

## 启动

macOS 图形模式建议使用 `mjpython`：

```bash
mjpython brazing_line.py --order A
mjpython brazing_line.py --order B
mjpython brazing_line.py --order C
mjpython brazing_line.py --order A --no-ui
mjpython brazing_line.py --batch A
```

如果 `mjpython` 在调用 `otool` 时返回状态码 69，并提示尚未同意 Xcode License，先在 macOS 终端执行：

```bash
sudo xcodebuild -license accept
otool -l "$(which python)" >/dev/null && echo "Xcode tools ready"
mjpython brazing_line.py --order A
```

该操作需要 macOS 管理员密码，只需在当前机器上完成一次。

无界面验证：

```bash
python brazing_line.py --headless --order A
python brazing_line.py --headless --order A --fast
python brazing_line.py --headless --batch A --fast
```

常用终端命令：

| 命令 | 作用 |
|---|---|
| `order_a` / `order_b` / `order_c` | 启动 A/B/C 型订单 |
| `batch_a` | 启动三套 A 型工件的三层批次 |
| `fault fin_pose fin_02` | 注入翅片偏位，触发 Arm1 调整 |
| `fault brazing_gap slot_02_left` | 注入局部漏涂，触发 Arm2 单线补涂 |
| `fault furnace_profile recoverable` | 注入可返工炉温偏差 |
| `fault furnace_profile severe` | 注入严重炉温偏差 |
| `status` | 输出当前快照 |
| `pick_place` / `inspection_1` | 单独运行 Arm1 基板取放 / Arm3 材料检测 |
| `arm2_motion` / `fin_assembly` | 单独运行 Arm2 五槽双线预涂 / Arm1 逐片安装翅片 |
| `inspection_2` | 单独运行 Arm3 翅片几何检测 |
| `furnace_cycle` | 单独运行压紧、进炉、炉内停留 10 秒和返回 Table2 |
| `rack_transfer` | 单独运行底层托盘的出料、升降、推叉和入架 |
| `stop` / `continue` | 暂停当前段 / 从当前任务继续 |
| `reset` | 恢复初始状态 |
| `help` | 显示帮助 |

## HTTP 接口

默认监听 `127.0.0.1:8766`：

- `GET /state`：订单、产品、三臂、炉体、质检与 KPI 快照
- `GET /camera.ppm`：Arm3 腕部相机画面
- `POST /order`：`{"preset":"A"}`、`B` 或 `C`
- `POST /batch`：`{"preset":"A","layers":3}`
- `POST /fault`：`type + target + severity`
- `POST /segment`：`pick_place / inspection_1 / arm2_motion / fin_assembly / inspection_2 / furnace_cycle / rack_transfer`
- `POST /stop` / `POST /continue`
- `POST /speed`：`{"action":"accelerate"}` 或 `{"action":"decelerate"}`
- `POST /reset`

Qt 面板保留六个单层分段选择，并新增“运行三层批次”和“单独运行升降入架”。
面板会同时显示当前生产层、三套产品阶段、三个层位状态、锁销、升降高度和推叉位置。
“加速 ×2”和“减速 ÷2”会在不重启当前流程的情况下调整仿真速度，支持 `0.25×`～`32×`。

## 软件结构

```text
brazing_line.py           # 新主入口、viewer/headless 主循环
run_flexible_order.py     # 严格 YAML 订单、dry-run 和 macOS 自动转交入口
brazing_line.xml          # 三臂钎焊工作站
config/                   # 产品、订单、梳齿、配方和料架 YAML
brazing_sim/
├── domain.py             # 强类型订单、产品、任务与检测状态
├── config.py             # 参数校验和产品坐标派生
├── flexible/             # 严格加载、几何生成、工装/料架分配与 ProcessPlan
├── motion.py             # FR3 控制与匀速轨迹执行
├── scene.py / tools.py   # 场景对象池、Arm1 快换与 Arm2 固定焊料枪
├── fixture.py            # 梳齿换型和真实 slide-actuator 压紧状态机
├── conveyor.py           # Table2—炉体连续输送、暂停和原位返回状态机
├── batch.py              # 三套单层流程的批次协调与统一热循环
├── batch_transfer.py     # 出料、升降、推叉、层锁与成品缓存移载 actor
├── preflight.py          # 名称、绑定、槽位与路径启动预检
├── process.py            # 订单级事件驱动协调器
├── furnace.py            # 非阻塞虚拟炉状态机
├── quality.py            # 硬门槛和综合质量评分
├── resources.py          # 共享区域租约
├── kpi.py                # 节拍、利用率、等待与返工统计
└── api.py / ui.py        # HTTP、终端和 Qt 展示
```

核心原则：

- 所有槽位、涂覆和检测姿态都由产品坐标生成，不使用世界坐标硬编码。
- 抓取、夹具和工具连接使用 `equality weld` 工艺近似。
- Table1 以 Arm1—Arm3 连线方向为左右 X 轴，台面约 `0.92 × 0.46 m`，左右长度为前后长度的 2 倍。基板水平放在左半区、长边沿 X；原料翅片放在右半区、长边同样沿 X。原料间距刻意大于产品装配节距，避免夹爪带倒相邻翅片；面向 Arm1 的一侧不设挡墙。
- Arm1 配有独立快换架、平行夹爪和吸盘工具。快换架位于 Table1 与 Table2 之间约 `(-0.49, 0.42)` 的横向间隙，两个工具沿 Y 方向排列。Table1 的第二列翅片向台面内侧收回并沿 Y 向错列，既保持换刀姿态，也与工具架立柱保留安全净距。Arm1 无需绕到 Table1 后侧。基板任务自动换上吸盘，翅片任务先归还吸盘再换上夹爪，后续翅片复用已装夹爪。换刀按“悬停—低速插接—锁定—停稳—退出”执行。
- 翅片夹取 TCP 位于翅片几何中心，确保夹持点在翅片实体内。原料基板和翅片在吸附/夹紧完成前由 Table1 临时约束保持不动，抓取约束生效后才解除原料架约束。
- Arm1 抓取 weld 生效时会锁定第 7 关节（末端绕随体 Z 轴的独立滚转通道）。基板或翅片被携带期间该关节角度和工件抓取姿态保持不变；工件释放后仍保留当前滚转分支，直到末端平滑撤离并回到安全姿态才恢复滚转自由度。
- Arm1 的非接触行程以 0.18 m/s 执行逐 10 mm IK 验证，接近取料点和放置点时降速。夹紧、吸附、放置和释放前后均有停稳段，工件接触目标后用 0.6 秒平滑对正，再启用托盘/临时梳齿约束。
- Table2 工装的固定框架、15/20/30 mm 前后梳齿模块、legacy 40 mm 模块、基板定位块和上压板都挂在 `fixture_tray` 下并随托盘移动。非当前梳齿模块隐藏且关闭碰撞，前后模块和 target site 由同一产品坐标生成。
- 工装上方只保留翅片长度方向两端的两根短压梁；原有两根装饰长梁已经删除。短压梁先通过实际滑动关节、触觉接触和保压力压住翅片，达到目标力后启用机械锁止并关闭重复接触求解，避免保压阶段上下振荡。
- Table2 与正前方炉体之间使用一条实际 slide actuator 驱动的传送机构。托盘、基板、翅片和夹具依靠托盘约束整体连续运动，进炉后按仿真时钟停留 10 秒，再沿原轨迹回到 Table2 的零位；暂停后可从当前输送位置继续。
- 三层模式使用独立的出料、Z 向升降、水平推叉和成品侧移 slide actuator。托盘所有权只在与承接工位对齐后通过 weld 切换，运动任务中不修改自由体位姿。
- Arm2 只保留永久安装在末端的叉形双喷嘴焊料枪，不再设置搬运夹具、快换工具架或换刀任务。其中心 TCP 采用蛇形连续路径依次处理各空槽位：奇数槽从 `-X` 到 `+X`，偶数槽反向，上一槽抬枪后直接横移到相邻槽，只有最后一槽完成后才返回安全位。喷嘴间距由当前订单动态设置，同步形成左右材料线，局部返工只更新缺陷单线。
- 黄色材料 capsule 关闭 MuJoCo 的 same-frame 位置优化，覆盖显示按照实际 TCP 投影从运动起点单向增长；正向轨迹不会从中点扩散，反向轨迹也会从对应的 `+X` 端开始。
- 前后梳齿在材料检测通过后才启用，双短压梁在翅片安装和几何检测完成后才启用。因此 Arm2 的整段涂覆及局部补涂期间，梳齿和压梁都隐藏且关闭碰撞。
- 上压板使用真实 Z 向 slide joint、位置执行器、浮动弹簧关节和 touch sensor。控制器非阻塞执行接触搜索、力爬升和订单配置的 1.5 秒保压；物理运行必须由 touch sensor 达到订单目标力，假时钟测试才启用确定性后备值。
- `table2_zone` 资源租约保证 Arm1、Arm2、Arm3 和工装机构同一时刻只有一个执行者进入核心区。
- 启动和订单切换都会执行 `preflight_check()`；缺失 body/site/joint/actuator/sensor/weld、梳齿不对齐或路径越界会聚合成中文错误并禁止启动。
- 真值负责检测判定，相机仅用于演示。
- 虚拟炉温为压缩演示配方，不代表真实生产参数。
- 一个 MVP 时刻只运行一个活动订单。

## 质量与返工

焊前硬门槛包括翅片位置、垂直度、根部间隙与节距；钎料检测检查覆盖率、最大漏涂和横向偏差。自动调整和补涂各最多执行两轮，仍不合格则进入 `MANUAL_REVIEW` 并禁止装炉。

焊后评分：

```text
0.35 × 覆盖 + 0.25 × 几何 + 0.25 × 温度 + 0.15 × 夹具
```

- 分数 ≥ 0.90：`PASS`
- 0.75 ≤ 分数 < 0.90：`REWORK_REQUIRED`
- 分数 < 0.75 或严重工艺越界：`SCRAPPED`

## 测试

```bash
make test
make test-unit
make test-headless
make test-batch
make test-flexible
```

测试覆盖严格 YAML 错误路径、奇/偶翅片坐标、A/B/C 路径长度、12/24 对象池、
动态喷嘴、梳齿选型、料架分配、1/2/3 件部分装架、资源互斥、炉门互锁、质量阈值、
返工上限、MJCF 命名契约、headless 正常/故障闭环、托盘连续性、暂停/继续/复位和
单次批次热循环，以及旧 `multi_arm_line.py/xml` 冒烟回归。

## 说明

两张散热片照片仅用于整体造型参考。若后续需要真实产品级复现，应补充 CAD、工程尺寸、公差、材料/钎料信息和真实炉温曲线。

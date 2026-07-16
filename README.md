# 低压配电柜散热组件钎焊仿真

基于 **MuJoCo** 和三台 **Franka Research 3 (FR3)** 的散热组件柔性制造 MVP。系统用几何运动、工艺状态机、虚拟钎焊炉和真值质量检测表达完整生产闭环，不模拟真实金属熔化、润湿或传热。

> 当前主线：`brazing_line.py` / `brazing_line.xml`
> 旧电气板装配产线仍保留；原说明见 [legacy 文档](docs/legacy_electrical_board.md)。

## MVP 工艺

```text
订单 A
→ Arm1 放置基板并逐片组装 4 片翅片
→ Arm3 焊前几何检测
→ 不合格时 Arm1 自动调整并复检
→ Arm2 沿每片翅片左右根部涂覆，共 8 条路径
→ Arm3 钎料覆盖检测
→ 漏涂时 Arm2 自动局部补涂并复检
→ 夹具锁紧与整盘装炉
→ 预热 / 升温 / 保温 / 冷却
→ 出炉转运
→ Arm3 焊后检测
→ PASS / REWORK_REQUIRED / SCRAPPED
```

A 型默认几何：

| 对象 | 默认参数 |
|---|---|
| 基板 | 0.36 × 0.22 × 0.008 m |
| 翅片 | 0.30 × 0.002 × 0.06 m |
| 翅片数量 / 节距 | 4 / 0.06 m |
| 钎料路径 | 每片左右各 1 条，共 8 条 |
| 路径宽度 / 目标覆盖率 | 0.004 m / 95% |

场景预分配 8 片翅片和 16 条路径，后续可扩展 6/8 片订单而无需重建 MuJoCo viewer。

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
mjpython brazing_line.py --order A --no-ui
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
```

常用终端命令：

| 命令 | 作用 |
|---|---|
| `order_a` | 启动 A 型订单 |
| `fault fin_pose fin_02` | 注入翅片偏位，触发 Arm1 调整 |
| `fault brazing_gap fin_02_left` | 注入局部漏涂，触发 Arm2 补涂 |
| `fault furnace_profile recoverable` | 注入可返工炉温偏差 |
| `fault furnace_profile severe` | 注入严重炉温偏差 |
| `status` | 输出当前快照 |
| `pick_place` / `inspection_1` | 单独运行 Arm1 取放 / Arm3 焊前检测 |
| `arm2_motion` / `inspection_2` | 单独运行 Arm2 八路径涂覆 / Arm3 钎料检测 |
| `stop` / `continue` | 暂停当前段 / 从当前任务继续 |
| `reset` | 恢复初始状态 |
| `help` | 显示帮助 |

## HTTP 接口

默认监听 `127.0.0.1:8766`：

- `GET /state`：订单、产品、三臂、炉体、质检与 KPI 快照
- `GET /camera.ppm`：Arm3 腕部相机画面
- `POST /order`：例如 `{"preset":"A"}`
- `POST /fault`：`type + target + severity`
- `POST /segment`：`pick_place / inspection_1 / arm2_motion / inspection_2`
- `POST /stop` / `POST /continue`
- `POST /reset`

Qt 面板提供“单独运行取放”“检测1”“Arm2运动”“检测2”四个段选择。每段会自动建立所需的 A 型前置状态并在完成后暂停，因此可以直接观察 Arm2 换长枪和 8 条路径，无需等待 Arm1/Arm3。

## 软件结构

```text
brazing_line.py           # 新主入口、viewer/headless 主循环
brazing_line.xml          # 三臂钎焊工作站
brazing_sim/
├── domain.py             # 强类型订单、产品、任务与检测状态
├── config.py             # 参数校验和产品坐标派生
├── motion.py             # FR3 控制与匀速轨迹执行
├── scene.py / tools.py   # 场景对象池、weld 与 Arm2 快换
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
- Table1 以 Arm1—Arm3 连线方向为左右 X 轴，台面约 `0.92 × 0.46 m`，左右长度恰为前后长度的 2 倍。台面中心位于约 `(-1.079, 0.508)`，相对初始布局的 Arm1 径向距离约为 `1.275` 倍，比上一版再外移约 `8.5%`；该位置保留了基板全路径的 `3 mm / 3°` 可达性余量。基板水平放在左半区，长边沿 X 轴；A 型 4 片翅片放在右半区，长边同样沿 X 轴，中心距为 `70 mm`。面向 Arm1 的一侧不设挡墙，取料路径按新槽位动态生成。
- Arm1 配有独立快换架、平行夹爪和吸盘工具。快换架位于 Table1 与 Table2 之间约 `(-0.49, 0.42)` 的横向间隙，两个工具沿 Y 方向排列，Arm1 无需绕到 Table1 后侧。基板任务自动换上吸盘，翅片任务先归还吸盘再换上夹爪，后续翅片复用已装夹爪。换刀按“悬停—低速插接—锁定—停稳—退出”执行。
- 翅片夹取 TCP 位于翅片几何中心，确保夹持点在翅片实体内。原料基板和翅片在吸附/夹紧完成前由 Table1 临时约束保持不动，抓取约束生效后才解除原料架约束。
- Arm1 抓取 weld 生效时会锁定第 7 关节（末端绕随体 Z 轴的独立滚转通道）。基板或翅片被携带期间该关节角度和工件抓取姿态保持不变；工件释放后仍保留当前滚转分支，直到末端平滑撤离并回到安全姿态才恢复滚转自由度。
- Arm1 的非接触行程以 0.18 m/s 执行逐 10 mm IK 验证的可见运动学搬运，接近取料点和放置点时降速。夹紧、吸附、放置和释放前后均有停稳段，运动插值使用零端点速度，分段 IK 保留上一段真实关节末态。工件接触目标后先停稳，再用 0.6 秒平滑微调对正产品坐标系，然后切换托盘/梳齿约束；这样既不瞬移，也不会把 IK 残余角度误差锁进夹具。Table2 托盘与梳齿夹具作为同一固定组件不进行内部碰撞求解。Arm2 涂覆和 Arm3 扫描使用 TCP-DLS。
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
```

测试覆盖配置与坐标派生、资源互斥、炉门互锁、质量阈值、返工上限、MJCF 命名契约、headless 正常/故障闭环，以及旧 `multi_arm_line.py/xml` 冒烟回归。

## 说明

两张散热片照片仅用于整体造型参考。若后续需要真实产品级复现，应补充 CAD、工程尺寸、公差、材料/钎料信息和真实炉温曲线。

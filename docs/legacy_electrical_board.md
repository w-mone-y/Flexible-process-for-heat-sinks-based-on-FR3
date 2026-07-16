# 低压电气板柔性生产仿真

基于 **MuJoCo** 的三台 **Franka Research 3 (FR3)** 机械臂柔性装配产线仿真。面向低压电气板的上料、插装压装、螺丝紧固与检测返工全流程，支持订单驱动、散乱料架感知、Arm2 自动工具快换，以及故障注入闭环返工。

---

## 产线布局

```
Table1 原料料架          Table2 中转定位治具         Table3 电气板装配台
(散乱元件)               (按类型分槽)                (插装 / 压装 / 拧螺丝)
     │                        │                           │
   Arm1 ──上料──►           Arm2 ──装配紧固──►           Arm3 检测/返工调度
```

| 机械臂 | 职责 |
|--------|------|
| **Arm1** | 从 Table1 散乱料架抓取元件，放到 Table2 对应类型缓存槽 |
| **Arm2** | 从 Table2 取件 → Table3 插装/压装 → 自动换电批拧螺丝；检测失败时返工 |
| **Arm3** | 检测元件位姿与螺丝状态；失败则生成返工任务交回 Arm2 |

---

## 环境要求

- Python 3.10+
- [MuJoCo](https://mujoco.org/) ≥ 3.1（建议使用 `mjpython` 启动带 viewer 的仿真）
- 可选：PyQt5 / PySide（控制面板 UI）

```bash
# 推荐：conda 环境
conda activate ai_learning   # 或你自己的环境名
pip install "mujoco>=3.1.0"
```

macOS 上带图形界面时请用：

```bash
mjpython multi_arm_line.py --seed 42
```

---

## 快速开始（主程序）

```bash
# 启动三臂产线（默认打开 MuJoCo viewer + HTTP 控制 + 终端命令）
mjpython multi_arm_line.py --seed 42

# 常用参数
mjpython multi_arm_line.py --seed 42 --no-ui          # 不要 Qt 面板
mjpython multi_arm_line.py --seed 42 --port 8766      # 指定 HTTP 端口
```

启动后在终端输入：

| 命令 | 说明 |
|------|------|
| `order_a` / `order_b` / `order_c` | 执行预设订单 A / B / C |
| `scatter [seed]` | 重新散乱 Table1 料架 |
| `fault screw_slot_1_a` | 注入螺丝未拧紧故障 → 触发重拧返工 |
| `fault relay_1` | 注入元件偏位故障 → 触发重压返工 |
| `stop` | 停止并清空任务 |
| `ee_axes on/off` | 显示/隐藏末端坐标轴 |
| `mocap on/off` | 显示/隐藏 mocap 目标 |
| `help` | 显示命令帮助 |

也可通过 HTTP / Qt 面板下发同样的订单与故障指令（默认 `http://127.0.0.1:8766`）。

---

## 预设订单

| 订单 | 内容概要 |
|------|----------|
| **A** | `relay_1→slot_1`（含 2 颗螺丝 + 检测）+ `terminal_1→slot_5` |
| **B** | `relay_2→slot_2`、`breaker_1→slot_4`、`terminal_1→slot_6`（含螺丝与检测） |
| **C** | 单件急单：`relay→slot_3`（含螺丝与检测） |

元件类型：继电器 `relay`、端子排 `terminal`、按钮 `button`、断路器 `breaker`。

---

## 单件工艺链

```
Arm1:  feed_pick → feed_place(Table2 类型槽)
Arm2:  get_tool(gripper_press)
       → assemble_pick → assemble_place → press → fixture_hold
       → return_tool(gripper_press)
       → get_tool(screwdriver) → screw × N → return_tool(screwdriver)
Arm3:  inspect
       └─ fail → Arm2 按原因返工（重压 / 重拧）→ 再检测
```

元件状态机：

```
raw → staged → placed → assembled → waiting_for_inspection → pass / fail
```

---

## Arm2 自动工具快换

Arm2 **不永久安装**夹爪/电批，法兰上为快换母盘，工具平时放在工具架上：

| 工具 | 名称 | 用途 | TCP |
|------|------|------|-----|
| A | `gripper_press` | 夹取 + 压装一体 | `arm2_grasp_tcp` / `arm2_press_tcp` |
| B | `screwdriver` | 自动电批 | `arm2_screwdriver_tcp` |

挂载方式：MuJoCo `equality weld` 在「工具↔法兰」与「工具↔工具架」之间互斥切换。  
压装完成后，Table3 工装 weld 临时固定元件，便于 Arm2 去换电批。

相关日志示例：

```
[Arm2] Request tool: gripper_press
[Arm2] Tool gripper_press mounted
[Arm2] Pick relay_1 from staging_relay
[Arm2] Component grasped
[Arm2] Move relay_1 to slot_1
[Arm2] Insert complete
[Arm2] Press-fit complete
[Arm2] Component fixed by board fixture
[Arm2] Return tool gripper_press
[Arm2] Request tool: screwdriver
[Arm2] Tighten screw_slot_1_a
[Arm2] screw_slot_1_a tightened
[Arm2] Return tool screwdriver
[Arm2] Assembly complete, waiting for Arm3 inspection
```

---

## 代码结构（产线核心）

| 文件 | 职责 |
|------|------|
| `multi_arm_line.py` | 主入口：调度、HTTP/UI/终端、共享区互斥、停车避让 |
| `multi_arm_line.xml` | 三臂场景、料架/治具/电气板、工具架与快换工具 |
| `object_manager.py` | 元件规格、订单、状态机、模拟相机感知、料架散乱、工装固定 |
| `skill_library.py` | 位姿数学、抓取/放置/压装/拧螺丝/检测技能步生成 |
| `arm_controller.py` | 单臂 5D IK、腕部旋转、技能执行器、自动回 home |
| `arm2_tool_manager.py` | Arm2 工具状态、取还工具技能、dock/undock weld |

### 设计要点

- **柔性**：料架位姿由 `--seed` 散乱；技能由感知位姿生成，无示教点位
- **共享区互斥**：`central` / `staging` 区令牌 + 跨臂碰撞 exclude，避免双臂同台卡死
- **空闲停车**：无任务时自动抬升回 home，不在共享区上空悬停
- **闭环返工**：检测失败按原因插入重压 / 重拧阶段

---

## 其他入口（演进 / 教学）

| 脚本 | 说明 |
|------|------|
| `flexible_pick_place.py` | 单臂柔性抓取放置（产线前身） |
| `base_develop.py` | 早期单臂开发基线 |
| `diffik*.py` / `opspace.py` | MuJoCo 差分 IK / 操作空间控制教学示例 |
| `grasp_*_control.py` | 抓取与打磨等专项实验 |

机器人模型资源：

- `franka_fr3/` — FR3 主模型（产线使用）
- `franka_emika_panda/`、`kuka_iiwa_14/`、`universal_robots_ur5e/` — Menagerie 教学模型

---

## 仿真简化说明

第一阶段以**完整工艺闭环**为目标，以下内容用逻辑近似，接口已预留扩展：

- 抓取 / 工具挂载 / 工装保持：`equality weld`
- 压装：位移 + 保持时间 + 槽位 snap
- 拧螺丝：腕部旋转圈数 + 状态更新（非真实螺纹/扭矩）
- 视觉：`ObjectManager.perceive()` 读仿真真值位姿（可替换为真实视觉）

---

## 目录一览

```
机械臂跟随控制/
├── multi_arm_line.py / .xml     # ★ 三臂产线主程序
├── object_manager.py
├── skill_library.py
├── arm_controller.py
├── arm2_tool_manager.py
├── flexible_pick_place.py / .xml
├── base_develop.py / .xml
├── franka_fr3/                  # FR3 模型与场景
├── diffik*.py / opspace.py      # IK / OSC 教学
└── README.md
```

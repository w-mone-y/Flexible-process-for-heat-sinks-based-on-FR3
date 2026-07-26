# 凸透镜体 spherical cap 打磨运动规划问题总结

## 1. 问题背景

当前任务是：

> 给定一个由两个球体重叠截取得到的双凸透镜体，使机械臂夹持该凸透镜体，并通过规划凸透镜体的运动，让其中一个球冠面 spherical cap1 依次接触打磨滚轮，从而实现类似“球冠面被均匀打磨”的效果。

该问题可以分成四个层次：

1. **几何建模**：明确凸透镜体、母球、球冠面、球心、法线之间的关系。
2. **曲面采样**：在 spherical cap1 上生成适合打磨的连续采样路径。
3. **刚体目标位姿生成**：把每个曲面采样点转换为凸透镜体的目标 body pose。
4. **机械臂映射**：通过已建立的末端—凸透镜体锁定关系，将凸透镜体目标位姿反算成机械臂末端目标位姿，再由 IK 控制机械臂跟踪。

---

## 2. 凸透镜体几何定义

用户描述的凸透镜体可以理解为：

- 由两个半径相同的球体重叠截取得到。
- 两个母球半径均为：

\[
R = 72.5\ \text{mm} = 0.0725\ \text{m}
\]

- 两个母球球心都位于凸透镜体随体坐标系的 z 轴上。
- 凸透镜体的 xOy 平面是中间对称平面。
- xOy 平面将凸透镜体分成上下两个相同的 spherical cap。

如果两个球心之间的总距离为：

\[
d = 52.5\ \text{mm}
\]

则每个母球球心到中间 xOy 平面的距离为：

\[
a = \frac{d}{2} = 26.25\ \text{mm}
\]

需要特别注意：

> 代码中的 `double_sphere_center_half_distance` 如果设置为 `0.0525`，它表示的是“半距离 52.5 mm”，也就是两个球心总距离为 105 mm。  
> 如果真实几何是“两个球心总距离 52.5 mm”，那么该参数应该是 `0.02625`。

这个参数会直接影响球冠边界、采样范围和旋转中心，因此必须确认。

---

## 3. spherical cap1 的数学表达

假设 cap1 是上侧球冠面，且它来自下方母球。则该母球球心在凸透镜体局部坐标系中可以写为：

\[
C_1^{local} = [0, 0, -a]^T
\]

球冠面上的点满足：

\[
P^{local} = C_1^{local} + R n^{local}
\]

其中：

- \(R\) 是母球半径。
- \(n^{local}\) 是从母球球心指向球冠面采样点的单位向量。
- 对于标准球面，采样点的外法线方向就是 \(n^{local}\)。

如果使用局部平面坐标 \((x, y)\)，则球冠面高度为：

\[
z = -a + \sqrt{R^2 - x^2 - y^2}
\]

其中：

\[
\rho = \sqrt{x^2 + y^2}
\]

球冠边界位于中间平面 \(z = 0\)，因此最大投影半径为：

\[
\rho_{max} = \sqrt{R^2 - a^2}
\]

对应的最大球面极角为：

\[
\theta_{max} = \arccos\left(\frac{a}{R}\right)
\]

---

## 4. 曲面采样策略：阿基米德螺线 + 等弧长步长

推荐采用阿基米德螺线覆盖 cap1：

\[
\rho = k\theta
\]

\[
x = \rho \cos\theta
\]

\[
y = \rho \sin\theta
\]

然后将 \((x, y)\) 投影到球冠面上：

\[
z = -a + \sqrt{R^2 - \rho^2}
\]

这样可以得到从中心向边缘连续扩展的螺旋打磨路径。

### 为什么不用固定角度步长？

如果简单令：

\[
\theta_{i+1} = \theta_i + \Delta\theta
\]

则平面投影上的点看似均匀，但映射到曲面后，真实曲面弧长和法线变化不一定均匀。尤其在球冠边缘区域，由于曲面相对于 xOy 平面的坡度更大，固定角度步长可能导致：

- 相邻点曲面距离变大；
- 相邻点法线夹角变大；
- 机械臂末端姿态变化突然；
- IK 抖动、翻腕，甚至到不了目标姿态。

因此更推荐使用**等弧长步长**。

### 等弧长近似公式

球冠表面上的螺线路径微分长度可以近似写为：

\[
\frac{ds}{d\theta}
= \sqrt{\frac{R^2}{R^2 - \rho^2}k^2 + \rho^2}
\]

其中：

\[
\rho = k\theta
\]

因此每一步可以根据目标弧长 \(\Delta s\) 自适应计算：

\[
\Delta\theta = \frac{\Delta s}{ds/d\theta}
\]

这样可以让相邻采样点在真实球冠面上近似等弧长。

---

## 5. 法线夹角约束

机械臂末端不仅要跟踪位置，还要跟踪姿态。如果相邻采样点的曲面法线变化太大，末端姿态会发生突变。

建议限制相邻采样点法线夹角：

\[
\Delta \alpha \leq 1^\circ \sim 2^\circ
\]

对于球面，曲面弧长与法线夹角近似满足：

\[
\Delta s \approx R \Delta\alpha
\]

其中 \(\Delta\alpha\) 使用弧度。

当 \(R = 72.5\ \text{mm}\) 时：

- \(1^\circ\) 对应弧长约为：

\[
0.0725 \times \frac{\pi}{180} \approx 1.27\ \text{mm}
\]

- \(2^\circ\) 对应弧长约为：

\[
0.0725 \times \frac{2\pi}{180} \approx 2.53\ \text{mm}
\]

因此推荐初始参数：

```text
arc_step = 0.0015 ~ 0.0025 m
max_normal_angle = 1.0° ~ 2.0°
```

如果机械臂姿态抖动或边缘丢姿态，可以减小 `arc_step` 或减小 `max_normal_angle`。

---

## 6. 从 cap1 采样点生成凸透镜体目标位姿

对于每个 cap1 采样点：

\[
P_i^{local}
\]

以及对应法线：

\[
n_i^{local} = \frac{P_i^{local} - C_1^{local}}{R}
\]

需要生成一个凸透镜体 body 的目标位姿：

\[
(p_{body}, R_{body})
\]

使得：

1. cap1 上的该采样点接触打磨滚轮附近的目标接触点；
2. 该采样点的表面法线对准打磨接触法线；
3. 凸透镜体整体姿态连续，不发生突然翻转。

假设打磨接触点为：

\[
P_{contact}^{world}
\]

打磨接触法线为：

\[
n_{contact}^{world}
\]

则需要让：

\[
R_{body} n_i^{local} = n_{contact}^{world}
\]

并且：

\[
p_{body} + R_{body} P_i^{local} = P_{contact}^{world}
\]

因此：

\[
p_{body} = P_{contact}^{world} - R_{body}P_i^{local}
\]

实际实现时，需要额外处理绕接触法线的自旋自由度。否则虽然法线对准了，但 target 坐标轴可能翻转，造成机械臂姿态不可达。

---

## 7. 映射到机械臂末端目标

步骤二完成后，代码中会记录机械臂末端 site 到凸透镜体 body 的固定关系：

```python
lock_state.site_to_body_pos = site_mat.T @ (body_pos - site_pos)
lock_state.site_to_body_mat = site_mat.T @ body_mat
```

这表示：

- `site_to_body_pos`：在末端 site 坐标系中，凸透镜体 body 原点的位置；
- `site_to_body_mat`：从末端 site 坐标系到凸透镜体 body 坐标系的相对旋转。

当规划得到目标凸透镜体 body 位姿：

```python
target_body_pos
target_body_mat
```

可以反算机械臂末端 site 目标：

```python
site_mat = target_body_mat @ lock_state.site_to_body_mat.T
site_pos = target_body_pos - site_mat @ lock_state.site_to_body_pos
```

然后将 `site_pos` 和 `site_mat` 转换成 mocap target 目标位姿，交给差分 IK 控制机械臂跟踪。

整体链路为：

```text
cap1 螺线采样点
→ 局部点 P_local 和法线 n_local
→ 目标 body 位姿 target_body_pose
→ 通过 locked transform 反算机械臂末端 site 目标
→ 设置红色 mocap target
→ IK 驱动机械臂
→ 凸透镜体按锁定关系跟随机械臂
```

---

## 8. 关于蓝色/红色 mocap target 轴线反向的问题

在实际调试中出现过：

> 采样点位置在滚轮附近，但红色 mocap target 的蓝色 z 轴方向反了，导致机械臂转不到目标姿态。

这个问题通常不是采样点错了，而是目标姿态矩阵构造时，工具轴方向或接触法线方向选错。

需要明确：

- 红色 mocap target 的蓝色轴通常对应局部 z 轴。
- 如果希望它指向滚轮，应设定目标 z 轴为滚轮方向。
- 如果希望它远离滚轮，则应设定相反方向。

注意：

> 仅仅绕接触法线旋转 180°，并不会改变蓝色 z 轴方向。  
> 如果蓝色 z 轴本身就是接触法线方向，那么绕 z 轴旋转只会改变红轴/绿轴，不会改变蓝轴。

因此如果要翻转蓝色 z 轴，必须直接翻转用于构造目标姿态的接触法线方向，例如：

```python
contact_normal_world = np.array([1.0, 0.0, 0.0])
# 或
contact_normal_world = np.array([-1.0, 0.0, 0.0])
```

而不是只改变绕法线的自旋角。

---

## 9. 推荐的实现函数结构

可以在当前 MuJoCo 控制脚本中增加以下核心函数：

```python
def build_cap1_spiral_goals(
    arc_step: float,
    loop_spacing: float,
    max_normal_angle_deg: float,
    sphere_center_offset: float,
    contact_normal_sign: float,
    lock_state: WorkpieceLockState,
) -> tuple[list[PoseCommand], list[str]]:
    """生成 cap1 阿基米德螺线打磨轨迹，并映射成机械臂末端目标。"""
```

该函数内部流程为：

1. 确定母球半径 `R`。
2. 确定母球球心 `C1_local = [0, 0, -a]`。
3. 计算 cap1 最大投影半径 `rho_max`。
4. 根据阿基米德螺线生成 \(\rho, \theta\)。
5. 使用等弧长步长自适应推进 \(\theta\)。
6. 对每个采样点计算：
   - `P_local`
   - `n_local`
7. 根据滚轮位置和接触法线生成目标 `target_body_pos, target_body_mat`。
8. 调用 `site_command_from_body_pose()` 映射成机械臂末端目标。
9. 返回 `PoseCommand` 列表供主控制循环执行。

---

## 10. 推荐的调参顺序

### 第一步：确认几何参数

先确认：

```text
两个球心距离到底是总距离 52.5 mm，还是半距离 52.5 mm？
```

对应参数：

```text
总距离 52.5 mm → sphere_center_offset = 0.02625 m
半距离 52.5 mm → sphere_center_offset = 0.0525 m
```

### 第二步：确认接触方向

观察红色 mocap target 的蓝色 z 轴：

- 如果蓝轴应该指向滚轮，就选择指向滚轮的 contact normal。
- 如果蓝轴应该远离滚轮，就选择相反方向。

如果切换选项后蓝轴没有变化，说明代码只改变了绕法线自旋角，而没有真正改变 contact normal，需要直接翻转 `contact_normal_world`。

### 第三步：从稀疏采样开始

建议先用：

```text
arc_step = 0.003 m
loop_spacing = 0.008 m
max_normal_angle = 2.0°
```

确认整体方向正确后，再逐渐加密：

```text
arc_step = 0.0015 ~ 0.002 m
loop_spacing = 0.004 ~ 0.006 m
max_normal_angle = 1.0° ~ 1.5°
```

### 第四步：检查机械臂可达性

如果边缘点机械臂转不过去，可能是：

1. 目标姿态蓝轴方向反了；
2. 工具绕法线的自旋角不合适；
3. 采样范围太靠边；
4. 相邻点间姿态变化太大；
5. 当前 IK 初值导致翻腕；
6. 打磨滚轮位置相对机械臂工作空间太极限。

---

## 11. 当前推荐方案总结

最终推荐的运动规划方法是：

```text
1. 使用凸透镜体随体坐标系定义 cap1 母球球心。
2. 在 cap1 上用阿基米德螺线生成覆盖路径。
3. 不使用固定角度步长，而使用球冠表面等弧长步长。
4. 控制相邻采样点法线夹角不超过 1°~2°。
5. 对每个采样点生成凸透镜体目标 body pose。
6. 使用步骤二建立的末端—凸透镜体锁定关系，将 body pose 反算为机械臂末端 site pose。
7. 将 site pose 发送给红色 mocap target，让差分 IK 驱动机械臂运动。
8. 椭球体/凸透镜体始终由 locked transform 跟随机械臂末端，不直接 teleport。
```

这个方案的优点是：

- 曲面覆盖连续；
- 适合打磨路径；
- 姿态变化可控；
- 可以直接接入当前 MuJoCo + mocap target + differential IK 控制框架；
- 后续可以继续扩展为双 cap 扫描、边缘加强采样、接触力控制或优化式轨迹规划。

---

## 12. 后续可扩展方向

后续如果希望进一步提高打磨效果，可以考虑：

1. **双 cap 切换扫描**  
   分别对 cap1 和 cap2 生成螺线轨迹。

2. **边缘区域加密**  
   在靠近球冠边缘处进一步减小 `arc_step`。

3. **姿态连续性优化**  
   对生成的目标姿态序列做 quaternion slerp 平滑，避免局部翻转。

4. **接触力控制**  
   在 IK 位置控制外增加法向压入量或力反馈。

5. **接触隐式轨迹优化**  
   如果希望自动寻找更优接触路径，可以参考 contact-implicit trajectory optimization 的思路，把“采样点接触滚轮”和“法线对齐”写成优化约束。


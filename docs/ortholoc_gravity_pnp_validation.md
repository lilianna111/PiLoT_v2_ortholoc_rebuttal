# OrthoLoC 中加入重力约束 PnP：方法、对照与验证记录

## 1. 目的

本文记录 OrthoLoC 的 PnP 阶段如何使用外部 roll/pitch 先验，并回答以下问题：

1. 重力先验在求解器中究竟做了什么；
2. 该做法与 UAVD4L 的实现哪里相同、哪里不同；
3. 如何证明当前实现确实使用了先验、坐标系正确，且没有破坏原始 PnP；
4. 为什么启用先验后最终整体指标可能只发生很小变化。

本改动仅影响 `run_google_roma.sh -> run_feicuiwan.sh -> main.py` 所使用的 OrthoLoC PoseLib PnP 路径。匹配器、地图裁剪、首帧初始化、GT reset 协议和评价程序均未因重力约束而替换。

## 2. 先验的定义与坐标系

当前 angle 文件每行为：

```text
image_basename roll_deg pitch_deg
```

用户已确认其 roll/pitch 与 OrthoLoC 最终输出 pose 的坐标约定一致，并且 `pitch=0` 表示纯俯视。因此，**不需要也不应额外进行机体—云台/相机外参转换**；文件来自 IMU、云台还是其他设备不影响数学使用方式。只有输入仍处于不同坐标系时，才需要先进行外参转换。

工程中用于生成裁剪姿态的相机到局部 ENU 旋转为：

```text
R_enu_from_cam = Rotation.from_euler('xyz', [pitch - 180, roll, yaw])
```

重力约束只需要世界 Up 在 OpenCV 相机坐标系中的方向，不需要 yaw。令 `e_up=[0,0,1]^T`，则相机坐标中的有符号 Up 是：

```text
u_cam = R_enu_from_cam^T e_up
      = [-sin(roll), sin(pitch-180) cos(roll), cos(pitch-180) cos(roll)]^T .
```

例如 `roll=0, pitch=0` 时，`u_cam=[0,0,-1]^T`：相机前方朝下，世界 Up 在相机后方，符合纯俯视的 OpenCV 相机坐标。

PnP 的 3D 点是减去 ECEF 原点后的局部坐标；减去原点只改变平移、不旋转坐标轴。因此世界 Up 仍须用该地图原点经纬度对应的 ECEF 局部竖直：

```text
u_world = [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]^T .
```

这里不能错误地使用 ECEF `[0,0,1]^T`。最终 PnP 输出 `R_w2c` 时，先验残差定义为：

```text
theta = angle(R_w2c u_world, u_cam).
```

它与 yaw 无关，正好约束相机的两个倾斜自由度（roll/pitch），但不约束 yaw、位置或高度。

## 3. 当前 OrthoLoC 的实现

实现位于以下文件：

- `OrthoLoC/OrthoLoC/ortholoc/gravity.py`：读取具名 RP 先验，转换为 `u_cam`；计算 ECEF Up。
- `native/gravity_pnp/gravity_pnp.cc`：独立编译的 PoseLib 2.0.5 重力门控后端。
- `OrthoLoC/OrthoLoC/ortholoc/utils/pose.py`：保持原有相机内参、`+0.5` COLMAP 像素约定、RANSAC 配置和 inlier 映射，仅在启用模式时调用重力后端。
- `main.py`：按图像 basename 取 RP 先验，以当前地图 ECEF 原点计算 `u_world`，并写入调试信息。

求解器仍复用 PoseLib 2.0.5 的 P3P、RANSAC 采样、MSAC 重投影评分、LO-RANSAC 和最终 BA。唯一加入的几何合法性条件是：

```text
theta <= threshold_deg       # 当前实验为 2 degrees
```

具体地：

1. P3P 每次产生候选 pose 后，若不满足该条件则不参与评分；
2. LO-RANSAC 优化后再次检查；若优化把合法 pose 推到门限外，则回退到优化前的合法 pose；
3. 最终 BA 后再次检查；若越界，则不接受该最终 BA 结果；
4. 全部 P3P 候选被拒绝时，返回失败，绝不返回单位位姿、传感器初值或普通视觉 PnP 的隐式 fallback；
5. 返回的 inlier mask 由**最终返回 pose**重新计算，而不是报告优化前 pose 的 mask。

这是一种**硬门限（hard gating）**，不是将 roll/pitch 强行设成测量值，也不是向 BA 添加一个加权软残差。因而视觉重投影仍决定可行集合内部的最佳 pose；传感器仅排除与其方向明显不一致的解。

## 4. UAVD4L 如何做

UAVD4L 的相关调用在 `UAVD4L-main/hloc/localize_sensloc_loftr_withgravity.py`，其本地 PoseLib fork 的候选筛选在 `UAVD4L-main/poselib/PoseLib/robust/ransac_impl.h`。

从该 RANSAC 源码可直接看到：P3P 候选生成后，若启用了 `use_gravity_axis`，它计算

```text
angle(gravity_axis, R.col(2)) < gravity_thresh
```

仅将通过角度检查的候选送入后续的重投影评分和 RANSAC 流程。外层脚本以 `gravity_threshold=2` 分别调用无重力和有重力的 `ransac_PnP_with_gravity`；它还把 sensor pose 作为 `T_init` 传给 wrapper。该仓库中 `pixlib.models.metric_utils.ransac_PnP_with_gravity` 的源码未随副本提供，故本文不对 wrapper 内部的坐标变换作未验证推断。

## 5. 相同点与不同点

| 项目 | UAVD4L 可见实现 | 当前 OrthoLoC 实现 |
|---|---|---|
| 基础求解 | PoseLib P3P + RANSAC | PoseLib 2.0.5 P3P + RANSAC |
| 先验作用时机 | P3P 候选评分前筛选 | P3P 候选评分前筛选 |
| 门限 | `gravity_thresh`，示例为 2° | `--ortholoc_gravity_threshold_deg`，当前为 2° |
| 被约束的量 | `R.col(2)` 与给定重力轴的夹角 | `R_w2c u_world` 与 RP 推导出的 `u_cam` 的夹角 |
| yaw / 位置 / 高度 | 可见 RANSAC 门控代码不使用 | 明确不使用 |
| LO / 最终 BA 后检查 | 可见 fork 的候选筛选代码未再次检查 | 再次检查；越界则回退/拒绝 |
| 无合法解的处理 | 外层脚本将失败结果替换为 `T_init` | 明确 PnP 失败，不返回传感器 pose |
| 世界坐标 | wrapper 内部变换未在该副本中可见 | 显式使用当前 ECEF 地图原点的局部 Up |

因此，两者的核心思想一致：在三点 PnP 的候选 pose 层面使用重力方向缩小搜索空间。当前实现的额外检查是有意的安全强化：它避免 LO 或最终 BA 将一个本来合格的候选优化到 2°之外后仍作为“重力约束结果”返回。

为验证这一区别，重力后端提供仅供测试的 `enforce_refinement_gravity=False` 模式，模拟“只筛选候选”的行为；生产调用不传该参数，默认始终为严格模式。

## 6. 如何证明重力约束加对了

证明分为坐标、求解器、真实数据回放三层，而不是仅观察最终精度是否提高。

### 6.1 坐标关系的单元测试

测试直接调用工程实际的 `crop/crop/transform_colmap.py`，在多个 roll、pitch、yaw 组合下验证：

```text
R_w2c @ ecef_up(lon, lat) == camera_up_from_roll_pitch(roll, pitch)
```

误差阈值为 `1e-14`。测试还验证：

- `roll=0, pitch=0` 对应 `[0,0,-1]`；
- 改变 yaw 不改变重力向量；
- ECEF 局部竖直通常不等于 ECEF Z 轴；
- 旋转整个世界坐标、平移 ECEF 原点后，求解结果按理论坐标变换保持不变。

### 6.2 求解器等价性与失败行为

`tests/test_gravity_pnp.py` 覆盖以下关键性质：

1. 将阈值设为 180° 时，重力后端与环境中原始官方 PoseLib 的输出逐元素一致；覆盖多个随机 seed、噪声、误匹配比例及重复/退化对应点。
2. 严格门限下，输出 pose 的独立计算重力夹角不超过门限。
3. 返回的 inlier mask 与该输出 pose 的独立重投影计算完全一致。
4. 构造略有偏差的先验时，门限内的 pose 被接受、门限外的 pose 被拒绝。
5. 构造会使优化越界的样例时，严格模式会回退，而仅候选筛选模式可漂移到门限外。
6. 反向重力先验且极紧门限时，所有候选均拒绝，求解器返回失败而不是单位 pose。
7. 传入不完整先验（仅 camera Up 或仅 world Up）时明确报错，不能静默退回普通标定。

180°等价性尤其重要：它说明新后端没有改变 P3P、评分、RANSAC、内参或 BA 的原有数值路径；差别只在门限真正收紧时才出现。

### 6.3 固定真实 2D--3D 对应点的回放

验证脚本为 `scripts/verify_gravity_pnp.py`。它首先独立审计已有的 900 帧重力定位输出，再选取四个代表帧：首帧、历史结果中重力夹角最大的帧、发生 LO 回退的帧及最终 BA 回退代表帧。

为避免把 RoMa 匹配随机性或序列裁剪传播误判为 PnP 效果，脚本对每个选中帧只运行一次匹配，冻结其实际进入 PnP 的以下输入：

- 2D--3D 对应点；
- 相机内参和像素坐标约定；
- `u_cam`、`u_world` 和 ECEF 原点；
- 重投影阈值、RANSAC 迭代次数、成功概率和 seed。

然后以同一份冻结输入对照四种模式：原始 PoseLib、180°门限、2°严格模式、2°仅候选筛选模式；另加反转先验的极紧门限失败控制。脚本还重新生成选中帧的地图 crop，并要求其 RGB 与原运行保存的 crop PNG 完全相同。

本次报告位于 `/media/amax/PS2000/rebuttal/ortholoc/gravity_check_v2/report.json`，并写有 `checks_passed=true`。结果如下。

#### 全序列审计（已有 900 帧结果）

| 检查 | 结果 |
|---|---:|
| 审计帧数 | 900 |
| 最终 pose 的最大独立重力夹角 | 1.976212° |
| 与日志中求解器夹角的最大差异 | 0.000002185° |
| 阈值 | 2° |

所以该已有重力运行的 900 个最终 pose 都满足 2°硬约束；日志与脚本独立计算的差异只有浮点精度量级。

#### 冻结输入的四帧对照

| 图像 | 180°与原 PnP 的最大 pose 元素差 | 2°严格模式最终夹角 | 2°严格模式内点数 |
|---|---:|---:|---:|
| `0_0.png` | 0 | 1.792619° | 327 |
| `899_0.png` | 0 | 1.611191° | 216 |
| `5_0.png` | 0 | 1.934282° | 366 |
| `869_0.png` | 0 | 0.858557° | 361 |

所有四帧均通过“原 PnP == 180°控制”“严格模式不越界”“最终 inlier mask 一致”和“反向先验明确失败”的断言。

`899_0.png` 是最直观的优化后检查例子：

| 同一冻结输入的设置 | 最终重力夹角 | 内点数 | 最终 BA |
|---|---:|---:|---|
| 原 PnP / 180° | 3.616763° | 285 | 接受 |
| 2°仅候选筛选 | 3.422162° | 282 | 接受 |
| 2°严格模式 | 1.611191° | 216 | 拒绝越界 BA；5 次优化回退 |

这同时证明两件事：先验并非没有被使用；以及仅筛选 P3P 候选不足以保证最终 BA 的输出仍符合重力先验，当前严格实现确实发挥了作用。

## 7. 为什么启用重力约束后整体结果可能差别不大

结果接近不意味着约束未生效。主要原因是：

1. **大多数视觉解本来已经接近 RP 先验。** 例如上述 `0_0.png`、`5_0.png` 和 `869_0.png` 的无约束解已经在 2°以内；筛掉错误 P3P 候选后，最佳重投影解仍是同一个 pose。
2. **先验只约束两个旋转自由度。** 它不直接给位置、高度和 yaw，因此不能期望它像完整 6-DoF pose 先验一样总是显著降低平移误差。
3. **正确门控可能减少内点。** `899_0.png` 中严格模式从 285 个无约束内点变为 216 个内点，因为它放弃了与 RP 不一致、但纯重投影误差较小的解。这是硬约束的正常代价，不是程序故障。
4. **全流程指标还受 PnP 以外步骤影响。** 匹配、地图 crop、下一帧使用前一帧预测姿态，以及现有 GT reset 都影响完整序列指标。不同完整运行之间的 RoMa 随机采样和递推 crop 也会放大微小差异。

因此，判断“重力 PnP 是否正确”的主证据应是本文第 6 节的受控验证，而不是要求端到端中位误差必须大幅改善。端到端实验仍有价值，但回答的是“该传感器先验对系统整体收益有多大”，不是“该 PnP 约束有没有正确接入”。

## 8. 用于 rebuttal 的准确表述

可采用如下表述：

> We incorporate roll--pitch measurements as a gravity-direction prior in the PnP stage. The prior gates each P3P pose hypothesis by the angular consistency between the world Up direction and the camera-frame Up direction derived from the measured roll and pitch. We do not use prior yaw, position, or altitude. Unlike candidate-only gating, we also validate the pose after local optimization and final bundle adjustment, preventing an optimization step from returning a pose outside the stated angular bound. Controlled replay on identical 2D--3D correspondences confirms that disabling the gate (180 degrees) reproduces the original PoseLib result exactly, while the 2-degree setting enforces the intended bound.

对于公平性，应同时明确：这是带外部姿态传感器信息的变体，而不是纯视觉定位；与其比较的设置若无法获得等价的 roll/pitch 测量，应单独报告无先验基线与带先验变体。当前工作中，先验只进入 PnP 候选几何一致性检查，视觉匹配、地图和评价协议不因先验而改变。

## 9. 可复现命令

```bash
conda activate ortholoc
bash scripts/build_gravity_pnp.sh
python -B -m unittest discover -s tests -v

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -B scripts/verify_gravity_pnp.py \
  --reference_run /media/amax/PS2000/rebuttal/ortholoc/RoMa_gravity/DJI_20250612193930_0012_V \
  --output /media/amax/PS2000/rebuttal/ortholoc/gravity_check_new
```

验证脚本要求 `--output` 指向不存在的新目录，不覆盖已有定位结果。冻结后的 `.npz` 可不读取地图、不重新匹配而仅重放 PnP：

```bash
python -B scripts/verify_gravity_pnp.py \
  --inputs /media/amax/PS2000/rebuttal/ortholoc/gravity_check_new \
  --output /media/amax/PS2000/rebuttal/ortholoc/gravity_replay_new
```

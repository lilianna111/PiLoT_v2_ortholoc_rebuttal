# OrthoLoC RoMa：中心像素深度与 DSM 的 PnP 硬门控

本文记录旧 hard 模式。Google RoMa 启动脚本现默认使用 visual-first `balanced`
优化，见[软评分与 balanced 说明](ortholoc_soft_pnp.md)。重现本文门控实验必须
显式指定 `--ortholoc_prior_fusion hard`。

## 1. 本次实现

在普通 P3P 产生候选后检查深度先验，不替换最小求解器，不向 BA 添加软残差，也不直接修改候选的高度或姿态。

```text
P3P 产生候选（当前 crop 的平移 ECEF，world-to-camera）
    ↓
pose 有限性检查
    ↓
若启用 gravity：重力夹角 ≤ 2°
    ↓
若启用 depth：恢复绝对 ECEF 相机位置，转换为经纬高与参考姿态
             → 用中心像素 Z 深度反投影
             → 查完整 DSM，要求 |反投影点高程 - DSM 高程| ≤ 10 m
    ↓
原有 MSAC / LO-RANSAC / 最终 BA
    ↓
再次检查所有启用的约束，越界优化回退到优化前的合法 pose
    ↓
输出最终 pose，并重新计算最终 pose 的 inlier mask
```

全部候选被拒绝时返回 PnP 失败，不输出单位位姿，也不自动退回无先验 PnP。LO 优化越界则撤销该次优化；最终 BA 越界则保留 BA 前的合法解。

重力模式、无先验模式仍可独立运行；深度默认不启用。`run_google_roma.sh` 原有默认 `gravity` 保持不变。

## 2. 输入与深度定义

用户已确认 `depth_1` 是**深度图中心像素的相机 Z 深度**，单位米。

```text
/.../renders/<sequence>/0_0.npy 238.65972900390625
```

文本中已经保存深度值，因此不再打开行中提到的 npy 文件。以其 stem `0_0` 对齐实际查询图像 `0_0.png`，不按行号对齐。重复、缺失、不匹配、NaN 或非正深度均报错。启动时在现有结果目录清理之前校验输入和 DSM 头信息。

中心像素使用 `[w//2,h//2]`，与已有提取脚本 `depth[h//2,w//2]` 一致，**不使用 K 的主点替代数组中心**。当前 3840×2160 图像取 `(1920,1080)`，而主点约为 `(1915.7,1075.1)`。

以原始 OpenCV 内参计算：

```text
p_cam = depth * inverse(K) * [u_center, v_center, 1]^T
```

这里深度是 `p_cam.z`，不是将视线单位化后的欧氏测距。PoseLib 使用的 COLMAP `+0.5` 同时作用于匹配点和主点；深度检查使用原始 OpenCV K 和像素，不额外加 `0.5`。

## 3. 坐标恢复与参考姿态适配

现有 `cgcs2000_grid_to_local_ecef` 生成：

```text
X_local = X_absolute_ecef - O
X_cam = R_w2c * X_local + t_local
```

因此每一个候选的绝对相机中心和相机到 ECEF 的旋转为：

```text
C_absolute_ecef = -R_w2c.T * t_local + O
R_ecef_from_camera = R_w2c.T
```

只有深度检查临时恢复绝对坐标；MSAC 仍用原来的局部 pose 和局部 3D 点。最终输出仍由原来的 `add_ecef_origin_to_pose` 恢复位置，不重复加原点。

先用复制的 `ECEF_to_WGS84(C_absolute_ecef)` 得到 `(lon,lat,h)`。EPSG:4326 用于经纬度，代码保留传入/返回的 WGS84 椭球高，不将其误当成另一个“旋转坐标系”。

参考 `costs.py` 的姿态重建约定为：

```text
R_ecef_from_camera = Q_enu_to_ecef(lon,lat)
                    * EulerXYZ(pitch_ref, roll_ref, yaw_ref) * F
F = diag(1,-1,-1)
```

故本次适配器从候选旋转恢复：

```text
A = Q_enu_to_ecef.T * R_w2c.T * F
(pitch_ref, roll_ref, yaw_ref) = EulerXYZ(A)
pose_ref = [lon,lat,h,roll_ref,pitch_ref,yaw_ref]
```

不能直接把 OrthoLoC 输出的 `pitch-180` 欧拉角原封不动传给参考反投影函数，因为参考函数还乘了 F。适配后的旋转重建与原候选旋转一致，测试同时对照了直接 ECEF 反投影结果。

## 4. 复用了哪些代码

参考工程 `/home/amax/Documents/code/pilot_v2/pilot_v2_0702_60ms/Pilot_0528` **未修改**。

- `pixloc/utils/transform.py`：复制 ECEF/WGS84 转换和 ENU→ECEF 基向量构造；缓存相同的 pyproj 转换器以供 RANSAC 重复调用。
- `pixloc/pixlib/geometry/costs.py`：复制 `_center_points_from_wgs84_poses` 的 CPU 旋转、反投影部分，以及 `_query_dsm_heights` 的 CPU 仿射坐标和 `map_coordinates(order=1)` 双线性查表部分。
- 必要适配：使用候选自身的位置和旋转，而非渲染初值；中心像素显式传入；DSM 只读加载；越界、非有限值和 NoData 邻域严格拒绝。
- 不复制参考工程的 LM、Jacobian、深度软损失、Torch 或 CUDA 分支。

DSM 使用 GeoTIFF 自带 CRS，当前为 EPSG:4547。反投影点的经纬度转成 DSM 平面坐标后查高程。沿用工程现有的栅格索引约定：`col=(E-gt[0])/gt[1]`、`row=(N-gt[3])/gt[5]`，不额外加栅格半像素偏移。不把 DSM nodata `-9999` 或查表失败替换成 0。

残差为**垂直高程差**，不是 ECEF Z 分量，也不是沿视线的距离差；DSM 高程和输入/输出高度需要沿用现有地图流水线的一致高程基准。

实现主要位于 `ortholoc/depth.py` 和 `native/gravity_pnp/gravity_pnp.cc`。native 后端通过持有 GIL 的 Python CPU 回调复用几何函数，候选仍由原始 PoseLib P3P 生成。不开深度时不调用回调；未进行 GPU/CUDA 加速改造，原 RoMa 匹配设备配置不变。

## 5. 运行

在当前 ortholoc 环境中重新编译后端（更换 Python 环境也需重新编译）：

```bash
cd /home/amax/Documents/code/lxy/PiLoT_ortholoc
bash scripts/build_gravity_pnp.sh
python -B -m unittest discover -s tests -v

# 仅深度先验
bash run_google_roma.sh --ortholoc_pnp_prior depth --ortholoc_prior_fusion hard --ortholoc_pnp_seed 0

# 重力 + 深度
bash run_google_roma.sh --ortholoc_pnp_prior gravity_depth --ortholoc_prior_fusion hard --ortholoc_pnp_seed 0
```

默认按序列读取 `datasets/angle/<sequence>.txt` 和 `datasets/depth_1/<sequence>.txt`。可用 `--ortholoc_depth_prior_file` 指定文件，`--ortholoc_depth_threshold_m` 指定高程差门限。

输出路径分别为：

```text
/media/amax/PS2000/rebuttal/ortholoc/RoMa_depth/<sequence>/
/media/amax/PS2000/rebuttal/ortholoc/RoMa_gravity_depth/<sequence>/
```

目录中的 `poses.txt` 供现有精度评价使用；`poses.json` 保存最终 pose 和每帧 `pnp_stats`。再次运行相同模式、相同序列时，**原有流程会清理该模式的既有结果目录**；如需保留多次实验，设置新的 `--ortholoc_output_root`。

保留用户的 `target_names` 选择，不替换序列选择或 GT 初始化/reset 协议。2026-09-18 检查时，用户已将活动序列设为 0012_V。`evaluate.py` 已补充两种深度模式，但保留原有 `sequence=0012_V`，评价其他序列时需设置匹配的 sequence。

## 6. 验证和检查结果

实现时后端编译成功，46 项 CPU 测试通过，其中包含原有重力回归和新增深度测试：

- 绝对 ECEF 原点恢复、平移原点改变后的不变性；漏加原点的反例。
- 适配姿态的旋转重建及直接 ECEF 反投影对照。
- 从参考源文件 AST 提取原始函数，对照转换、反投影和 DSM CPU 查表结果；不导入参考工程、不修改它。
- 重力检查先于深度回调；深度单独启用和联合启用。
- `±10 m` 接受，`±10.01 m` 拒绝，NaN 拒绝；全部拒绝不返回单位 pose。
- 优化越界回退；返回 pose 的约束和 inlier 一致性。
- DSM 边界、斜坡双线性插值、NoData 和非有限值拒绝；真实 DSM 在求解器相关库加载后可只读加载。
- CLI/环境变量优先级、中心像素与 OpenCV/COLMAP 像素约定、调用链先验转发。

当前六个活动序列的深度文件与图像名称已只读检查：每个序列均有 900 条有效深度，覆盖全部查询图像。本次**未运行完整序列、未生成新精度指标**。

可先在已有冻结的四帧 PnP 输入上进行 CPU 检验，不运行匹配器或重新 crop：

```bash
python -B scripts/verify_depth_pnp.py \
  --frozen-dir /media/amax/PS2000/rebuttal/ortholoc/gravity_check_v2 \
  --depth-file /media/amax/AE0E2AFD0E2ABE69/datasets/depth_1/DJI_20250612193930_0012_V.txt \
  --dsm /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif \
  --output-dir /media/amax/PS2000/rebuttal/ortholoc/depth_check_v1
```

输出目录必须尚不存在。脚本检查放行控制与原 PoseLib 一致、全拒绝控制返回失败；真实深度/联合模式若成功，独立用直接 ECEF 公式核验反投影、约束和最终 inlier。若打印 `NO FEASIBLE POSE`，表示这帧没找到可行解，不表示验证证明其定位成功。已有四份真实冻结输入已完成 CPU 回放与独立核验；完整序列的运行和精度评测由用户执行。

主要诊断字段：`depth_rejected_hypotheses`、`invalid_depth_hypotheses`、`gravity_rejected_hypotheses`、`rejected_refinements`、`final_refinement_accepted`、`depth_camera_wgs84`、`depth_point_wgs84`、`depth_dsm_height_m`、`depth_residual_m`、`depth_error_m`。

最终误差变化很小本身不代表门控无效：若原视觉解已满足门限，最佳解可能不变；门控通常排除错误候选，而不一定提高已正确定位帧的精度。应结合候选拒绝数、最终约束残差、失败率和固定输入对照判断。

论文公平性仍需注明深度数据实际来源。代码不读取 GT 作为深度 fallback，并不自动证明输入深度来自独立传感器；如果深度由 GT/渲染生成，需明确其性质及噪声设定，不能称为已验证的真实独立传感器测量。

## 7. 失败帧冻结与独立重放（2026-09-18）

此修改只增加诊断，不改变门控、MSAC、优化、随机种子或 DSM 半像元约定。重新编译 native 后端：

```bash
bash scripts/build_gravity_pnp.sh
```

启用深度时，若 native PnP 返回失败，会在该序列结果目录的 `pnp_failures/` 保存：

- `<frame>.npz`：所有筛选/采样后实际传入 native 的 2D–3D 对应点，COLMAP `+0.5` 像素约定、原始 OpenCV K、绝对 ECEF 原点、重力向量、阈值、RANSAC options 和失败统计。
- `<frame>.json`：上述配置，以及通过重力检查的 P3P 候选中**绝对深度残差最小的有限 DSM 候选**。记录局部/绝对 w2c pose、绝对 ECEF 相机位置、参考代码的经纬高姿态、反投影点、DSM 投影坐标、浮点行列、四邻域高程和有符号残差。该候选在 MSAC 之前记录，未评分或优化，不作为输出 pose。

若无有限 DSM 残差，`closest_depth_candidate` 为 null；不是用单位 pose 冒充候选。快照独占创建，已有同名文件不覆盖；I/O 失败会报告 `failure_diagnostic_error`，不替代原始 PnP 失败。成功帧不写快照。失败帧调试表和 `poses.json` 也保留该帧 ECEF 原点。

入口程序原本会清空同名序列输出目录。为保留上一轮结果，使用**尚未使用过的输出根目录**运行：

```bash
ORTHOLOC_OUTPUT_ROOT=/media/amax/PS2000/rebuttal/ortholoc/depth_diagnostic_20260918 \
bash run_google_roma.sh \
  --ortholoc_pnp_prior gravity_depth \
  --ortholoc_prior_fusion hard \
  --ortholoc_gravity_threshold_deg 10 \
  --ortholoc_depth_threshold_m 10 \
  --ortholoc_pnp_seed 0
```

例如再次失败于 373 帧后，独立重放：

```bash
python -B scripts/replay_depth_failure.py \
  --snapshot /media/amax/PS2000/rebuttal/ortholoc/depth_diagnostic_20260918/RoMa_gravity_depth/DJI_20250612193930_0012_V/pnp_failures/373_0.npz
```

脚本不匹配、不 crop、不运行 GPU、不写结果。先直接使用 `C_abs=-R.T@t+O` 和 `P_abs=C_abs+R.T@(depth*K^-1*[u,v,1])` 复算，不经过欧拉角；独立用 rasterio 仿射逆变换与手写四邻域插值核验高程残差。同时打印扣除半像元后的查表结果作对照，不用于修改门控。然后按快照中的原始输入、种子、重力/深度门限重跑 native PnP，检查失败状态、迭代/拒绝数量、最小残差候选和残差均一致。

DSM 路径、CRS 和 geotransform 被记录，但没有复制整个 DSM。重放需要保留原 DSM 文件；若 DSM 内容或求解实现发生变化，复现断言可能失败。重放一致证明这次失败可以复现、这次候选几何计算一致，并不单独证明输入深度的测量射线与实际传感器标定一致。

新增诊断、精确输入冻结、最小候选记录与只读重放测试后，50 项 CPU 测试通过。未运行完整定位序列；373 帧的实际失败快照仍需本次用户运行才能获得。

还在临时目录编译了移除本次诊断记录的对照后端（未修改仓库中的后端源码），同进程回放 `0_0`、`5_0`、`869_0`、`899_0` 四份真实输入，使用重力 10°、深度 10 m、seed 0：新旧后端 pose 数组、inlier、全部原有求解统计和残差完全一致，pose 最大差为 0。无约束控制与原始 PoseLib 的 pose 最大差也为 0。这验证诊断记录未改变所检输入的求解结果，不替代实际失败帧的检查。

# RoMa / OrthoLoC 重力先验 PnP

`run_google_roma.sh` 当前调用 `run_feicuiwan.sh → main.py`，不是 `main_google_0401.py`。
此改动只针对这条调用链。`main.py` 和共享脚本默认 `none`，仍使用原环境的官方 PoseLib；
`run_google_roma.sh` 默认开启 `gravity`，按目标序列名读取下面的 angle 目录。
匹配、裁剪、首帧初始化和 GT reset 协议不变；新增先验不读取 GT 或上一帧姿态。

## 编译（一次）

在运行定位的同一个 Python 环境中执行：

```bash
conda activate ortholoc
bash scripts/build_gravity_pnp.sh
python -B -m unittest discover -s tests -v
```

需要 CMake >= 3.16、C++17 编译器、Eigen3，以及首次编译时的网络。
源依赖固定为 PoseLib v2.0.5 (`7e9f5f5`) 和 pybind11 v2.13.6 (`a2e59f0`)。
依赖下载到 `native/gravity_pnp/build`；扩展输出为 `ortholoc/_gravity_pnp*.so`，均不提交 Git。
不安装/覆盖环境中的 `poselib`，换 Python 环境时需重新编译。

## 先验文件

默认 `roll_pitch` 格式：

```text
# 完整图像 basename，单位：度
1_0.png 0.0 40.9
2_0.png 0.2 41.1
```

已配置目录 `/media/amax/AE0E2AFD0E2ABE69/datasets/angle`，
每个序列读取 `<序列名>.txt`，每行严格为 `图像名 roll pitch`，不读取 yaw 或位置。
当前目标 `DJI_20250612193930_0012_V` 的 900 张图像都有对应先验。

必须是本工程的**相机/云台姿态约定**：
`R_enu_from_cam = Rotation.from_euler('xyz', [pitch - 180, roll, yaw], degrees=True)`。
转换为 `R_enu_from_cam.T @ [0,0,1]`，不使用 yaw。
不能直接输入没有做机体—相机外参转换的机体 IMU roll/pitch。

也可显式选择 `--ortholoc_gravity_prior_format camera_up`：

```text
# 世界 Up 在 OpenCV 相机坐标系中的有符号方向 ux uy uz
1_0.png 0.017 0.753 -0.658
```

OpenCV camera 轴为右、下、前。若输入是向下重力，应先取反变为 Up。
向量读取后归一化，不对夹角取绝对值。按 basename 严格对齐，拒绝重名、NaN、
零向量、缺失查询；不按行号猜测，不回退到 GT，按序列名确定文件。

## 运行

无重力先验（原路径）：

```bash
bash run_google_roma.sh --ortholoc_pnp_prior none --ortholoc_pnp_seed 0
```

有重力先验：

```bash
bash run_google_roma.sh --ortholoc_pnp_seed 0
```

覆盖默认先验文件或阈值：

```bash
bash run_google_roma.sh \
  --ortholoc_pnp_prior gravity \
  --ortholoc_gravity_prior_file /absolute/path/to/angles.txt \
  --ortholoc_gravity_threshold_deg 2 \
  --ortholoc_pnp_seed 0
```

也支持环境变量 `ORTHOLOC_PNP_PRIOR`、`ORTHOLOC_GRAVITY_PRIOR_DIR`、`ORTHOLOC_GRAVITY_PRIOR_FILE`、
`ORTHOLOC_GRAVITY_PRIOR_FORMAT`、`ORTHOLOC_GRAVITY_THRESHOLD_DEG`、`ORTHOLOC_PNP_SEED`、
`ORTHOLOC_OUTPUT_ROOT`。显式先验文件优先于目录拼接。
命令行参数优先。脚本仍使用它原来的目标序列和地图；文件名中的 google 不代表自动切换数据。

`run_google_roma.sh` 结果根目录为 `/media/amax/PS2000/rebuttal/ortholoc`，
无先验和重力先验分别保存到 `RoMa/<序列名>/` 和 `RoMa_gravity/<序列名>/`。
位姿 TXT、JSON、调试日志、reset 统计、匹配图和 crop 可视化均在该根目录下；
共享脚本单独运行仍沿用旧的默认结果根目录。临时裁剪缓存路径不变。
同一模式再次运行仍沿用原程序的覆盖/清理行为，请先备份已有该模式结果。

## 算法与日志

复用 PoseLib 2.0.5 的三点 P3P、采样、MSAC 重投影评分、LO-RANSAC 和最终 BA。
候选合法性：`angle(R_w2c @ up_world, up_camera) <= threshold`。
3D 点使用平移后的 ECEF 坐标，因此 `up_world` 是由地图 ECEF 原点的经纬度计算的
局部 ENU Up，不能用 ECEF `[0,0,1]`。

区别于 UAVD4L 原实现：LO/最终 BA 后仍检查角度，若越界保留优化前合法解；
全部候选拒绝时返回失败，不优化/返回单位位姿，不返回传感器位姿。
无合法解时沿用当前定位失败即停止的流程，不做隐式视觉 fallback。

`poses.json` 保存 prior 模式、文件、格式、阈值、seed，以及逐帧 `pnp_stats`：
迭代数、inliers、生成/拒绝候选数、拒绝 LO 次数、最终 BA 是否接受、最终重力夹角。
`ortholoc_pose_debug.txt` 追加模式、夹角和拒绝候选数。

测试包含与官方 PoseLib 的 180°无筛选结果对齐、ECEF 竖直、相机 RP/yaw 约定、
噪声/误匹配、错误先验、全部拒绝、单位位姿假解、原 PnP 路径及采样 mask 映射。

注意：两次顺序运行可能因上一帧输出不同而产生不同 crop；这可以比较系统整体收益，
不能宣称只量化了 PnP 的收益。纯 PnP 成对实验应使用冻结的同一批 2D–3D 对应点。
现有 GT 初始化/reset 未在本次改动中消除，因此不要把整条运行管线称为完全无 GT 的纯视觉设置。

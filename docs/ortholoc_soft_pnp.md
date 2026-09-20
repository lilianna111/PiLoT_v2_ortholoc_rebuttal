# OrthoLoC RoMa：旧版软评分与当前 visual-first balanced PnP

> `soft` 是保留的旧版候选重评分模式。Google RoMa 启动脚本当前默认
> `balanced`：视觉 P3P/MSAC/LO-RANSAC 先确定 pose 和内点，先验仅在最终
> pose-only BA 中以受限梯度辅助优化；视觉分数或内点数恶化时回退视觉 pose。

## 当前推荐：balanced

```text
RoMa 2D–3D 对应点
    → 原始 PoseLib P3P + MSAC + LO-RANSAC（不读取重力／深度）
    → 原始视觉 BA，得到 visual pose 与 visual inliers
    → 检查测量可靠性：角度残差 ≤ 25°；深度 DSM 残差有限且 ≤ 20 m
    → 在同一视觉内点上做视觉 + 可用先验的 pose-only BA
    → 重新计算全部视觉对应点的 MSAC 和内点数
    → 内点保留 ≥ 90% 且视觉 MSAC 不恶化超过 5%：接受；否则回退 visual pose
```

该模式参照 `costs.py` 的思想以梯度范数平衡先验。`gravity_weight`、
`depth_weight` 的默认 `0.1` 对应每项约 `0.15 × ||g_visual||` 的初始目标，
两个先验的合成梯度再硬限制为不超过 `0.30 × ||g_visual||`。这是最终 BA 的
梯度预算，**不是** P3P 候选总分的 30% 配额。深度只更新平移，重力只更新旋转。

运行双先验：

```bash
bash run_google_roma.sh \
  --ortholoc_pnp_prior gravity_depth \
  --ortholoc_prior_fusion balanced \
  --ortholoc_pnp_seed 0
```

结果目录为 `RoMa_gravity_depth_balanced`；输出 `poses.json` 的 `pnp_stats` 记录
`prior_gravity_used`、`prior_depth_used`、`prior_depth_skip_reason`、先验强度、
优化前后视觉分数／内点数和 `prior_refinement_accepted`。

## 旧版 soft：实现范围

保留原始 PoseLib 2.0.5 的 P3P 求解器、采样器、视觉内点定义、LO-RANSAC 和视觉 BA。
不新增联合 LM，不把先验加入视觉 BA 的 Jacobian，不直接改写输出高度或 roll/pitch。
软模式只改变假设评分及精化结果的保留标准；旧硬模式仍可运行。

```text
相同的 RoMa 2D–3D 对应点
    → 普通 P3P
    → 有限性检查（深度启用时还需有效 DSM 查询）
    → 视觉 MSAC + 重力鲁棒惩罚 + 深度鲁棒惩罚
    → 原来的视觉 LO-RANSAC / BA
    → 用相同联合评分比较精化前后 pose
    → 输出最终 pose，重算视觉内点
```

软模式不会因有限重力夹角超过 10°或有限深度残差超过 10 m 删除候选，最终也不做这种超限剔除。
NaN、无穷或无效 DSM 不是“大残差”：深度权重大于零时，这类候选仍不可评分。
它们不会被改成零残差，否则候选可以通过离开 DSM 覆盖范围逃避惩罚。
没有自动停用大残差先验、GT 回退或新增失败重试。原主程序定位失败后停止的行为不变。

## 评分公式与参数

设 N 为当前 PnP 实际使用的对应点数，tau 为该 estimator 原有的视觉内点阈值。

```text
V(T) = original_MSAC(T) / (N * tau²)
rho(s) = 2 * log(1 + s / 2)
G(T) = gravity_weight * rho((gravity_error_deg(T) / gravity_scale_deg)²)
D(T) = depth_weight   * rho((depth_residual_m(T) / depth_scale_m)²)
S(T) = V(T) + G(T) + D(T)
```

V 在 PoseLib 已有的归一化相机坐标中计算，tau 同步按焦距归一化；不混合像素和角度单位。
原 MSAC 将视觉离群点及相机后方点计为 tau²，故 V 是对应点数量归一化的截断视觉代价。
重力沿用已有 roll/pitch → signed camera Up 和世界 Up 的夹角；不新增 yaw 测量。
深度沿用已核查的绝对 ECEF 恢复、原始 OpenCV K、数组中心相机 Z 深度、DSM 双线性查表。  

rho 复用只读参考项目 costs.py 的 loss_fn1 在 alpha=0、truncate=1 时的核形式。
代码以对数空间计算，避免极大的有限残差平方溢出后重新变成硬拒绝。
它是单调增长的有限软惩罚，不是直接截断损失。核导数随大残差下降，但此处**只做候选评分**，
不把该导数加入视觉优化器。没有照搬参考代码的自适应梯度范数缩放。
参考的核宽度、角度残差表示及深度选择权重与这里不完全相同，不能宣称逐项数值等价。

| CLI 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--ortholoc_prior_fusion` | Google RoMa 启动脚本为 balanced；直接 main.py 为 hard | 选择 balanced、旧 soft 评分或旧硬门控 |
| `--ortholoc_gravity_scale_deg` | 10 | 软评分尺度，不是最大允许角度 |
| `--ortholoc_depth_scale_m` | 10 | 软评分尺度，不是最大允许高程差 |
| `--ortholoc_gravity_weight` | 0.1 | 重力惩罚强度；0 不参与评分 |
| `--ortholoc_depth_weight` | 0.1 | 深度惩罚强度；0 不参与评分 |

0.1 是初始调试权重，不是验证集已选定的论文参数，不能据此保证精度提升。
旧的 `gravity_threshold_deg` / `depth_threshold_m` 只在 hard 模式生效。
Google RoMa 仍默认仅启用 gravity，depth 需显式选择 `depth` 或 `gravity_depth`。

原程序要求最终 pose 有超过三个视觉内点。软评分也不把只有三点支持的假设作为有效最优解；
这没有改变视觉内点定义。LO 仍可精化这种样本，并在获得足够视觉支持后参与评分。
否则极小先验惩罚可能把一个无法作为定位结果的三点模型排在可用视觉模型之前。

LO 调用原视觉 refine_model 后，若数值无效或联合分数变大，则保留调用前的 pose。
最终 BA 同理；这里比较的是同一 S，而非先验是否小于门限。
所有实际启用的软权重为零时，后端使用原始 MSAC、原精化接受逻辑，并跳过深度回调，
避免仅仅添加一个常数或归一化改变原视觉路径的数值行为。
无先验 `--ortholoc_pnp_prior none` 仍直接调用原始 poselib.estimate_absolute_pose。

## 运行与保存

```bash
cd /home/amax/Documents/code/lxy/PiLoT_ortholoc
bash scripts/build_gravity_pnp.sh

# 两个先验的软评分（不新增 LM）
bash run_google_roma.sh \
  --ortholoc_pnp_prior gravity_depth \
  --ortholoc_prior_fusion soft \
  --ortholoc_gravity_scale_deg 10 \
  --ortholoc_depth_scale_m 10 \
  --ortholoc_gravity_weight 0.1 \
  --ortholoc_depth_weight 0.1 \
  --ortholoc_pnp_seed 0

# 原始无新增先验 PnP
bash run_google_roma.sh --ortholoc_pnp_prior none --ortholoc_pnp_seed 0

# 旧硬门控对照
bash run_google_roma.sh --ortholoc_pnp_prior gravity_depth \
  --ortholoc_prior_fusion hard --ortholoc_gravity_threshold_deg 10 \
  --ortholoc_depth_threshold_m 10 --ortholoc_pnp_seed 0
```

新软模式结果位于 `/media/amax/PS2000/rebuttal/ortholoc/` 下的：

```text
RoMa_gravity_soft/<sequence>/poses.txt
RoMa_depth_soft/<sequence>/poses.txt
RoMa_gravity_depth_soft/<sequence>/poses.txt
```

不会使用旧硬模式的目录。同一软模式、同一序列再次运行，原主流程仍会清理该结果目录。
不同权重/尺度的实验需指定不同的 `--ortholoc_output_root`，否则会覆盖同模式旧结果。
图像匹配设备、用户当前序列选择、crop 生成、GT 初始化与 GT 触发下一帧重置均未修改。

`poses.json` 保存融合方式、尺度、权重，以及每帧 pnp_stats 中的：

- `prior_fusion`、`soft_priors_active`；
- `gravity_error_deg`、`depth_residual_m`；
- `visual_score_normalized`、`gravity_prior_cost`、`depth_prior_cost`、`joint_score`；
- `final_refinement_score_before/after`、`final_refinement_accepted`；
- 原有视觉内点数、候选数和拒绝统计。

启用有效软先验且成功时，`model_score` 与最终 `joint_score` 一致。
零权重或旧硬模式的 model_score 仍是原始 MSAC，不能跨融合模式直接比较这个字段。
软模式的 gravity_rejected_hypotheses 为零；depth_rejected_hypotheses 仅统计无效 DSM 残差，
不是超过尺度的有限残差。成功帧也可能保留很大的先验残差，这是软评分的允许行为，不代表数据正确。

## 验证和公平性边界

CPU 测试入口：`python -B -m unittest discover -s tests -p 'test_*pnp.py' -v`。
新增测试覆盖大残差不超限拒绝、10 m 两侧连续评分、极大有限残差、零权重恢复原 PnP、
两个先验分别改变视觉模型选择、独立 MSAC/惩罚分解、实际深度的直接 ECEF 检查、
LO/BA 接受评分、无效 DSM 无法逃避约束，以及 main/对应点/启动脚本参数转发。

2026-09-18：后端在 ortholoc Python 3.10 环境编译成功，66 项 CPU 测试通过。
373 帧旧硬模式的冻结输入按原参数重放，失败状态、计数和候选残差一致。
同一输入使用默认软权重 0.1 时虽返回 pose，但只有 5/1959 个视觉内点，重力误差约 38.245°，
深度残差约 3.346 m。独立评分复算一致：V=0.997491、G=0.423575、D=0.010891、S=1.431957。
这说明异常深度能够把候选选择拉偏，**不是定位正确的证明**，默认权重不能未经验证用于论文。
同一输入的两个软权重归零后，与原始 PoseLib 返回 pose 的最大数组差为零，得到 312 个视觉内点。
这些检查没有运行完整序列或生成 GT 定位精度指标，也没有修复深度来源或 crop 边界插值问题。

已有四帧真实冻结输入在相同默认软配置下完成直接 ECEF 几何、独立 MSAC、惩罚分解及返回内点检查：

| 帧 | 最终视觉内点 | 重力夹角（度） | 深度残差（米） | 零权重与原 PoseLib pose 的最大数组差 |
| --- | ---: | ---: | ---: | ---: |
| 0_0.png | 179 | 0.995206 | -1.545269 | 0 |
| 5_0.png | 330 | 2.049923 | -1.288785 | 0 |
| 869_0.png | 363 | 0.837303 | -0.382199 | 0 |
| 899_0.png | 291 | 2.653041 | 2.580833 | 0 |

软模式需要对更多候选查 DSM，而不是先用角度门限删掉它们，Python CPU 回调也会增加耗时。
实现未进行 GPU/CUDA 加速；正式比较需要报告这一额外开销。

已有四帧冻结输入的只读 CPU 回放可使用验证脚本，软模式验证几何、评分分解和最终内点，
而不是要求先验误差小于硬阈值。保存报告的目录必须尚不存在：

```bash
python -B scripts/verify_depth_pnp.py \
  --frozen-dir /media/amax/PS2000/rebuttal/ortholoc/gravity_check_v2 \
  --depth-file /media/amax/AE0E2AFD0E2ABE69/datasets/depth_1/DJI_20250612193930_0012_V.txt \
  --dsm /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif \
  --prior-fusion soft \
  --output-dir /media/amax/PS2000/rebuttal/ortholoc/soft_check_v1
```

这是 sensor-assisted OrthoLoC with prior-aware hypothesis scoring，而非完整联合先验 BA。
公平性实验应比较原 OrthoLoC、仅角度、仅深度及联合四组，使用相同先验文件和地图，
验证集选权重并披露搜索范围，报告精度、成功率和运行时间，不只统计成功帧。
同一测量输入不等于相同融合能力；剩余差距不能全部归因于视觉方法。
需如实说明深度是实测还是 GT/渲染生成，以及各组的初始化、GT 重置和失败处理规则。

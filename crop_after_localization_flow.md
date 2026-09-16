# Crop 之后的数据如何用于后续定位

## 结论先说

`crop/test.py` 对应的 crop 结果，进入后续定位时真正核心有两类：

1. crop 出来的参考图像
   - `dom_warp`
2. crop 出来的 3D 点云
   - `xyz_ecef`

但在进入定位器之前，这两类数据会在 `main.py` 里进一步加工成一组“参考观测”：

- `render_frame`
- `Points_3D_ECEF`
- `p2d_r`
- `visible_r`
- `T_render`
- `T_init`
- `dd`
- `render_T_ecef`

其中：

- `render_frame` 用来提参考图特征
- `Points_3D_ECEF` 用来做几何投影
- `p2d_r` 用来在参考图上取参考特征
- `fp_q - fp_r` 构成主特征对齐残差
- `render_T_ecef + gt_depth / gt_roll / gt_pitch` 用来施加地理先验约束

---

## 1. crop 本身产出什么

`main.py` 调用的不是 `crop/test.py` 这个脚本文件本身，而是同一套核心函数：

- `crop/crop/proj2map.py` 中的 `crop_dsm_dom_point(...)`

它返回的主要数据有：

- `dom_warp`
- `xyz_ecef`
- `dsm_indices`
- `world_points_ecef`
- `hit_lonlat`
- `used_uvs`
- `crop_bbox`
- `homography`

在当前主链里，真正被继续使用的主要是：

- `dom_warp`
- `xyz_ecef`

其他字段大多是调试或分析信息。

---

## 2. crop 输出先在 `main.py` 里被二次处理

对应位置：

- `main.py` 中 `rendering_worker(...)`

### 2.1 取出 crop 结果

从 `crop_result` 里直接拿：

- `color = crop_result["dom_warp"]`
- `points3d = crop_result["xyz_ecef"]`

这里的含义是：

- `color` 是参考图
- `points3d` 是和参考图像素对齐的 3D 点云

### 2.2 先做有效值筛选

代码当前用的是：

```python
valid_mask = np.isfinite(points3d).all(axis=-1)
```

这一步的作用：

- 剔除 `NaN`
- 剔除 `Inf`

它得到一张二维 mask，表示哪些像素位置有可用 3D 点。

### 2.3 resize 到渲染相机分辨率

为了让 crop 图和 query 图在后续网络里尺寸一致，代码会把 crop 结果 resize 到渲染相机分辨率：

- 图像 `color` 用 `cv2.INTER_LINEAR`
- 点云 `points3d` 每个通道分别 resize，用 `cv2.INTER_NEAREST`
- `mask` 也 resize，用 `cv2.INTER_NEAREST`

作用：

- 保证参考图和 query 图在特征提取尺度上对齐
- 保证 3D 点和图像像素依然一一对应

### 2.4 padding 到 16 的倍数

如果配置里打开 `padding`，会继续补边：

- 图像 `color`：`BORDER_REPLICATE`
- 点云 `points3d`：`BORDER_REPLICATE`
- mask：`BORDER_CONSTANT, value=0`

作用：

- 让图像尺寸满足网络和特征提取器的尺寸要求
- 图像补边不引入黑边
- 点云也同步补边，维持像素对齐
- mask 在 padding 区设为 0，保证 padding 出来的区域不会被当成有效观测

### 2.5 从有效区域采样参考点

后面不会把整张 3D 图都送进优化，而是：

1. 用 `mask` 取出有效 3D 点
2. 随机采样最多 500 个
3. 同时记录这些点在参考图上的 2D 像素位置

于是得到：

- `points_sampled`
- `coords_sampled`

进一步构造成：

- `p2d_r`
- `visible_r`

其中：

- `p2d_r` 是这些参考点的 2D 像素坐标
- `visible_r` 初始全为 `True`

这一步非常关键，因为后面真正参与优化的是“参考图上若干采样点”，而不是整幅图每个像素。

---

## 3. crop 后的数据如何进入定位器

`main.py` 中把下面这些量送入：

- `localizer.run_query(...)`

传进去的主要变量有：

- `color`
- `points_sampled`
- `p2d_r`
- `visible_r`
- `T_init`
- `T_render`
- `dd`
- `render_T_ecef`
- `image_query`

这条链是：

1. `main.py`
2. `pixloc/localization/localizer.py`
3. `pixloc/localization/base_refiner.py`
4. `pixloc/pixlib/models/learned_optimizer.py`
5. `pixloc/pixlib/geometry/costs.py`

当前真正使用的 cost 实现是：

- `pixloc/pixlib/geometry/costs.py` 中的 `DirectAbsoluteCost2`

不是别的备份版本或旁支版本。

---

## 4. 每个 crop 后数据在后续定位里的用途

下面按“数据项 -> 在哪用 -> 作用”说明。

---

## 4.1 `render_frame`

### 来源

- 来自 crop 的 `dom_warp`
- 在 `main.py` 中 resize / padding 后得到

### 在哪用

在 `BaseRefiner.refine_query_pose(...)` 里：

- 对 `render_frame` 做 dense feature extraction

对应作用：

- 生成参考图的 dense feature map

也就是说，crop 图在后续确实是被当作“参考图像”来做特征提取和特征对齐的。

### 它的本质角色

它是：

- reference image

后面 query 图和它做特征匹配。

---

## 4.2 `Points_3D_ECEF`

### 来源

- 来自 crop 的 `xyz_ecef`
- 在 `main.py` 里经过 mask 筛选、resize、padding、采样
- 然后又在 `get_3D_samples_v3(...)` 里做了去中心化和坐标系整理

### 在哪用

在 `BaseRefiner.refine_pose_using_features(...)` 中作为：

- `p3d`

再进入 `costs.py` 中：

- `transform_p3d(...)`
- `project_p3d(...)`

### 作用

它是后续定位里的几何骨架。

具体作用是：

1. 这些 3D 点代表参考图上被选中的观测点
2. 对每个候选查询位姿，把这些 3D 点投到 query 图里
3. 得到它们在 query 图上的预测位置
4. 在这些位置上采 query feature
5. 和参考 feature 做差，形成优化残差

### 它的本质角色

它是：

- 几何对齐中的 3D 支撑点

没有它，系统只能做纯图像特征匹配；有了它，才能做“带位姿和相机模型约束的投影式特征对齐”。

---

## 4.3 `p2d_r`

### 来源

- 来自 `coords_sampled`
- 是采样出来的参考点在 crop 图上的二维像素坐标

### 在哪用

在 `costs.py` 中，先用它在参考图特征图上插值：

- `fp_r, valid_r, _ = self.interpolate_feature_map(f_r, p2d_r)`

### 作用

它定义了：

- 参考图上“哪些位置”的特征被拿来当作 reference descriptor

也就是说，`p2d_r` 不是优化变量，而是参考点采样位置。

### 它的本质角色

它是：

- 参考帧的 2D 采样坐标

后面所有 query 投影点都要和它们对应。

---

## 4.4 `visible_r`

### 来源

- 在 `main.py` 中初始设成全 `True`
- 后面可能在不确定度筛点时被进一步过滤

### 在哪用

在 `costs.py` 中：

- `valid_ref_mask = (valid_r & visible_r).to(torch.float32)`

### 作用

它和参考图插值得到的 `valid_r` 一起，构成参考点有效掩码。

这样做可以避免：

- 超出边界的点
- padding 区无效点
- 被筛掉的不可靠点

进入参考残差。

### 它的本质角色

它是：

- 参考采样点的有效性开关

---

## 4.5 `fp_r`

### 来源

- 参考图特征图 `f_r`
- 加上 `p2d_r`

### 在哪用

在 `costs.py` 中：

- `fp_r = ...`
- 后面和 `fp_q` 相减

### 作用

它就是：

- 参考帧上被采样位置的特征描述子

是后续特征对齐的目标值。

---

## 4.6 `fp_q`

### 来源

- query 图特征图 `f_q`
- 加上当前候选位姿投影得到的 `p2d_q`

### 在哪用

在 `costs.py` 中：

- `fp_q, valid_q, J_f = self.interpolate_feature_map(...)`

### 作用

它表示：

- 当前候选位姿下，这些参考 3D 点投影到 query 图后对应的 query feature

---

## 4.7 主特征对齐残差 `res = fp_q - fp_r`

### 在哪用

在 `costs.py` 中直接构造：

- `res = fp_q - fp_r`

### 作用

这是整个定位优化的主残差。

直观理解就是：

- 如果当前候选位姿是对的
- 那么参考点投到 query 上的位置，抽出来的 query 特征应该和参考特征相似

所以优化会不断调整位姿，使 `fp_q` 靠近 `fp_r`。

### 它的本质角色

它是：

- crop 后参考图参与后续定位的核心方式

不是做传统 keypoint matching，而是做 dense feature alignment under geometry。

---

## 4.8 `T_render`

### 来源

- 在 `get_3D_samples_v3(...)` 中，根据当前 render pose 构造出的参考帧位姿

### 在哪用

在优化器里作为：

- 参考帧的 pose

### 作用

它用于定义：

- 这些 `Points_3D_ECEF` 在参考系和优化坐标系里的几何位置关系

也用于后续和 refined pose 进行相对变化比较。

---

## 4.9 `T_init`

### 来源

- `get_3D_samples_v3(...)` 生成的初始候选查询位姿

### 在哪用

在优化器里作为：

- 初始候选位姿集合

### 作用

后续优化不是从单个位姿开始，而是从多个初始候选位姿开始跑，最后选最优候选。

因此它是：

- pose refinement 的起点

---

## 4.10 `dd`

### 来源

- `get_3D_samples_v3(...)` 中对 3D 点做中心化时得到的偏移量

### 在哪用

在：

- `T_render`
- `T_init`
- 深度先验的坐标恢复

都会用到

### 作用

它是数值稳定性相关量。

由于 ECEF 坐标非常大，直接优化容易数值不稳定，所以先减去 `origin` 和 `dd` 做局部化，再通过 `dd` 恢复。

---

## 4.11 `render_T_ecef`

### 来源

- `get_3D_samples_v3(...)` 返回的原始参考帧 ECEF 平移

### 在哪用

在 `costs.py` 中主要用于：

- 深度 prior
- 角度 prior

### 作用

它的作用不是特征对齐，而是把局部优化坐标恢复成真实地理坐标，用于：

1. 查询候选位姿恢复到 ECEF / WGS84
2. 用相机中心或中心光线去 DSM 查高度
3. 把候选姿态转成局部 ENU 再计算 roll / pitch

所以它是：

- 地理约束桥梁

---

## 4.12 `gt_depth / gt_roll / gt_pitch`

虽然它们不是 crop 直接产出，但和 crop 后数据是一起参与优化的，所以也说明一下。

### `gt_depth`

作用：

- 与当前候选位姿预测出来的中心点高度做比较
- 构造深度残差

前提：

- `render_T_ecef` 非空
- `dsm_path` 非空
- 开启 `enable_depth_prior`

### `gt_roll / gt_pitch`

作用：

- 与当前候选位姿的 roll / pitch 做比较
- 构造角度残差

前提：

- `render_T_ecef` 非空
- 开启 `enable_angle_prior`

它们都不是主特征残差，而是附加先验项。

---

## 5. 后续优化里真正发生了什么

可以把主流程概括成下面几步：

1. crop 产生参考图和参考 3D 点
2. 从参考图有效区域里采样一批 3D 点及其 2D 坐标
3. 对 crop 图和 query 图分别提 dense feature
4. 在参考图上用 `p2d_r` 取参考特征 `fp_r`
5. 用候选查询位姿把参考 3D 点投到 query 图，得到 `p2d_q`
6. 在 query 图上按 `p2d_q` 取 query 特征 `fp_q`
7. 用 `fp_q - fp_r` 构成主残差
8. 可选叠加深度 prior 和角度 prior
9. 高斯牛顿 / LM 迭代更新位姿

所以从本质上说：

- crop 输出的图像负责“参考特征”
- crop 输出的点云负责“几何投影”
- 两者一起构成“几何约束下的特征对齐”

---

## 6. 哪些 crop 输出没有进入后续定位主链

虽然 crop 返回了不少字段，但当前主链里没有真正参与定位优化的包括：

- `dsm_indices`
- `world_points_ecef`
- `hit_lonlat`
- `used_uvs`
- `crop_bbox`
- `homography`
- `debug_overlay_path`

这些更偏向：

- 调试
- 可视化
- 分析 crop 行为

而不是后续定位的核心输入。

---

## 7. 最终总结

### crop 后真正用于后续定位的核心数据

- `dom_warp`
- `xyz_ecef`

### 它们被加工成的后续定位输入

- `render_frame`
- `Points_3D_ECEF`
- `p2d_r`
- `visible_r`
- `T_render`
- `T_init`
- `dd`
- `render_T_ecef`

### 它们分别承担的角色

- `render_frame`：参考图像，用来提参考特征
- `Points_3D_ECEF`：几何支撑点，用来投影到 query
- `p2d_r`：参考采样点在 crop 图上的 2D 坐标
- `visible_r`：参考点有效性 mask
- `fp_q - fp_r`：主特征对齐残差
- `render_T_ecef + gt_depth / gt_roll / gt_pitch`：附加地理先验约束

### 一句话概括

crop 之后的结果不是简单拿去“做一张参考图”而已，而是同时提供了：

- 参考外观
- 参考 3D 几何
- 参考 2D-3D 对应关系

后续定位本质上是在做：

- 基于 crop 参考图和 crop 点云的几何引导特征对齐


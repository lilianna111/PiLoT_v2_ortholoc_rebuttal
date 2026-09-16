# 面向高精度实时定位的新 Crop 流程设计

## 1. 目标

目标只有一个：

- 为后续定位提供高质量、几何关系清楚的 `render_frame`、`p2d_r`、`Points_3D_ECEF`

后续优化真正需要的是：

- `Points_3D_ECEF[i]` 是一个真实 3D 点
- `p2d_r[i]` 是这个 3D 点在参考图 `render_frame` 上的位置
- 参考特征从 `render_frame` 的 `p2d_r[i]` 处提取
- 该 3D 点投影到 query 后，在 query 上提取对应特征
- 用 `fp_q - fp_r` 做 LM 优化

因此真正要保证的是：

- ref 侧自己的图像、相机、2D 坐标、3D 点彼此一致

不是：

- ref 和 query 必须同尺寸


## 2. 关键结论

### 2.1 ref 和 query 不需要相同尺寸

当前特征对齐不是逐像素对齐，而是：

1. 在 ref 上按 `p2d_r` 取特征
2. 用 `Points_3D_ECEF` 投影到 query，得到 `p2d_q`
3. 在 query 上按 `p2d_q` 取特征
4. 比较两边特征残差

相关代码：

- 参考点取特征：[costs.py:676](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/pixlib/geometry/costs.py#L676)
- 3D 投影到 query：[costs.py:688](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/pixlib/geometry/costs.py#L688)
- 残差定义：[costs.py:700](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/pixlib/geometry/costs.py#L700)

所以：

- ref 和 query 不需要同尺寸
- 不需要为了“匹配 query”把 ref 强行 resize 到 `512 x 288`


### 2.2 新方案里 crop 后不应强制 resize

如果目标是高精度实时定位，那么 crop 后更合理的做法是：

- 保留原始 crop patch 尺寸
- 不为了进入特征对齐而强制缩到 `512 x 288`

原因：

- resize 会减少有效纹理细节
- resize 会让参考特征发生额外变化
- 如果对 3D 图再 resize，会污染原始 DSM 真点

所以新流程里：

- 图像不强制 resize
- 3D 点绝不 resize


## 3. 新的 Crop 流程

### 3.1 四角点和 DSM 求交

输入：

- query 图像四个角点
- query 内参
- query 初始位姿
- DSM

处理：

- 对四个角点发射射线
- 与 DSM 求交
- 得到四个落点

输出：

- 地面四边形 footprint

这一步保留你现有方法。


### 3.2 使用 `bbox + quadrilateral mask`

不要再做透视规整。

做法：

1. 取四边形的轴对齐外接矩形 `bbox`
2. 从原始 DOM / DSM 上裁出 `dom_patch`、`dsm_patch`
3. 在 patch 内生成四边形 `polygon_mask`

这样：

- `dom_patch` 保持原始 DOM 像素
- `dsm_patch` 保持原始 DSM 栅格
- `polygon_mask == 1` 的区域是有效区域

这一步就是用规则矩形承载不规则四边形，但不改变真实纹理。


### 3.3 在原始 patch 上建立真实 3D

对 patch 中每个有效像素：

- 根据其在 DSM 中的位置取高程
- 配合地理坐标转成 ECEF

得到：

- `xyz_patch_ecef`

这一步要求：

- 不做透视 warp
- 不做 3D resize
- 保持 3D 点来自原始 DSM


### 3.4 无效值过滤

有效值规则建议是：

- `finite`
- 不是 `nodata`
- 不是明显异常哨兵值，如 `<= -9990`
- **不要因为 `< 0` 就过滤**

原因：

- 海面等区域可能出现负高程

这一点已经确认并修正。


### 3.5 在原始 patch 上采样

采样应该发生在原始 patch 上，而不是 resize 后，也不是 padding 后。

采样条件：

- `polygon_mask == 1`
- `3D 有效`

得到：

- `p2d_r`
- `Points_3D_ECEF`

这里的定义是：

- `p2d_r[i]`：原始 patch 上的像素位置
- `Points_3D_ECEF[i]`：该像素对应的原始 DSM 真点

这是新流程的核心。


### 3.6 `render_frame` 直接使用原始 patch

新流程里：

- `render_frame = dom_patch`

也就是说：

- 最终送给 ref 特征提取的参考图，就是原始 crop patch
- 不再强制缩放到 `512 x 288`

如果后续为了显存、速度确实要缩放，那应当作为单独的工程折中处理，而不是方法本身的必选步骤。


### 3.7 关于 padding

对新流程而言：

- crop 阶段不需要为了“和 query 同尺寸”去 padding
- 也不需要为了“变成 512 x 288”去 padding

但要区分两种 padding：

1. 方法层面的 crop 输出
2. 网络输入前的工程性补边

方法层面：

- 不需要 padding

工程层面：

- 如果后端网络需要方图或某个步长对齐，可以在进入网络前补边
- 但补边区域必须不参与采样

你当前 `base_refiner` 里会把图补成方图：

- [base_refiner.py:267](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/localization/base_refiner.py#L267)
- [base_refiner.py:268](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/localization/base_refiner.py#L268)

这属于后端工程处理，不应该反过来逼迫 crop 先把内容缩小。


## 4. 新流程里的三个核心输出

### 4.1 `render_frame`

定义：

- 原始 DOM 上通过 `bbox` 裁出的 patch

作用：

- 作为参考图提取特征

特点：

- 不透视变形
- 不强制 resize
- 尽量保持原始外观


### 4.2 `Points_3D_ECEF`

定义：

- 原始 patch 上采样得到的 DSM 真 3D 点

作用：

- 投影到 query，建立几何约束

特点：

- 不 resize
- 不从 resize 后 3D 图中采样
- 保持原始 DSM 真点属性


### 4.3 `p2d_r`

定义：

- 原始 patch 上采样点的 2D 坐标

作用：

- 在参考特征图上提取参考特征

特点：

- 与 `render_frame` 同一坐标系
- 与 `Points_3D_ECEF` 一一对应


## 5. 这个新流程是否满足你的目标

### 5.1 是否满足定位优化需求

满足。

因为它直接输出：

- `render_frame`
- `p2d_r`
- `Points_3D_ECEF`

并且三者关系更清楚：

- `Points_3D_ECEF[i]` 是真实 DSM 点
- `p2d_r[i]` 是该点在参考图上的位置
- `fp_r[i]` 从该位置提取


### 5.2 是否更符合高精度实时定位

更符合。

原因：

- 不再透视规整
- 不再强制 resize 降采样
- 保留更多原始纹理信息
- 保留原始 DSM 真点
- 计算量仍远小于逐像素真实重渲染


## 6. 之前问题是否被解决

### 6.1 “DSM < 0 应不应该过滤”

已解决。

结论：

- 不应因为 `< 0` 直接过滤


### 6.2 “四个角点围成的不是规则矩形怎么办”

已解决。

新方案答案是：

- 用 `bbox + quadrilateral mask`
- 不再透视规整


### 6.3 “透视变换会不会让参考图不真实”

已解决。

因为新流程不再使用透视规整。


### 6.4 “resize 会不会改变 `Points_3D_ECEF`”

已解决。

因为新流程不对 3D 点做 resize。


### 6.5 “采样点是真实 DSM 点还是后面虚构出来的”

已解决。

新流程中采样点来自：

- 原始 patch
- 原始 DSM
- 原始有效区域

因此它们是原始 DSM 真点，不是 resize 或 warp 后虚构出来的。


### 6.6 “resize 后 `p2d_r`、3D 点、特征还能不能对应”

这个问题在新主方案里被回避了。

因为主方案就是：

- 不强制 resize

所以：

- `p2d_r`
- `render_frame`
- `Points_3D_ECEF`

天然处于同一个原始 patch 坐标系。


### 6.7 “特征向量能不能和原始 DOM 完全一致”

在新主方案里，已经尽量接近这个目标。

因为：

- 不做透视规整
- 不做强制 resize

所以参考特征就是从原始 patch 上提的，已经尽量保持和原始 DOM 一致。

当然，如果后端网络输入前仍做了补边或其他预处理，特征提取管线内部仍可能有轻微变化，但比旧方案已经小很多。


## 7. 需要额外注意的一点

虽然方法上 crop 后不需要 resize，但**当前代码实现**里，`ref_camera` 和部分缩放逻辑还是按现有固定尺寸在写：

- [main.py:238](/home/amax/Documents/code/lxy/PiLoT_v55/main.py#L238)
- [base_refiner.py:170](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/localization/base_refiner.py#L170)
- [base_refiner.py:469](/home/amax/Documents/code/lxy/PiLoT_v55/pixloc/localization/base_refiner.py#L469)

所以如果按这个新流程落地代码，需要同步改：

- `ref_camera.size`
- `render_frame` 实际尺寸
- `p2d_r` 坐标系

原则是：

- ref 图可以不是 `512 x 288`
- 但 ref 侧自己的图像、相机、坐标系统必须一致


## 8. 最终结论

按现在讨论后的新流程，推荐方案是：

- 四角点求交
- 原始 DOM/DSM 上做 `bbox + quadrilateral mask`
- 在原始 patch 上采样 `p2d_r`
- 直接从原始 DSM 取 `Points_3D_ECEF`
- `render_frame` 直接使用原始 patch
- 不再为了进入特征对齐而强制 resize 到 `512 x 288`
- 只在后端网络确实有工程需要时，再单独处理补边

这套方案：

- 更符合你的高精度实时定位目标
- 解决了之前讨论的大部分核心问题
- 不再因为透视规整和强制 resize 损失额外真实性


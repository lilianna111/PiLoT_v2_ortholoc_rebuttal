`crop_wgs84` 是按 WGS84 地图栅格编写的一套独立裁剪代码。

坐标约定：
- 输入地图: WGS84 经纬度栅格，通常是 `EPSG:4326`
- 输入位姿: `[lon, lat, alt, roll, pitch, yaw]`
- 内部世界坐标: `ECEF (EPSG:4978)`
- 输出点云: `(H, W, 3)` 的 ECEF XYZ

目录说明：
- `crop/transform_colmap.py`: 把 WGS84 位姿转成 ECEF 下的 `pose_w2c`
- `crop/ray_casting.py`: 在 WGS84 DSM 上做射线求交
- `crop/proj2map.py`: 按 `中心点 + 短边构建斜向长方形` 生成第一阶段 bbox / mask / ECEF 点云
- `crop/pipeline.py`: 兼容 `pixloc/crop/test_0310.py` 的第二阶段旋转与裁边
- `test.py`: 命令行入口
- `test_0310_wgs84.py`: 直接改路径即可运行的脚本

输出文件：
- `{name}_bbox_dom.jpg`
- `{name}_bbox_dom.npy`
- `{name}_bbox_dom_mask.png`
- `{name}_crop_points_on_dom.jpg`
- `{name}_final_img.jpg`
- `{name}_final_points.npy`
- `{name}_final_mask.png`

说明：
- 这版沿用 `pixloc/crop/test_0310.py` 的 pipeline：
  第一阶段先用中心点投影和短边方向构造斜向长方形，保存 `bbox_*`。
  第二阶段再按 yaw 对 `bbox_*` 做同步旋转和裁边，保存 `final_*`。
- 输出点云是 ECEF XYZ，mask 同步保留了长方形外区域以及 DSM 空洞区域。
- 现在默认以内存返回结果，不强制保存中间文件。
- 主返回值里直接包含：
  `bbox_img`、`bbox_points`、`bbox_mask`、`final_img`、`final_points`、`final_mask`
- 可选开关：
  `save_outputs=True` 时保存完整文件；
  `save_debug_vis=True` 时只保存调试可视化图片。

命令行示例：

```bash
python /home/amax/Documents/code/PiLoT/PiLoT/crop_wgs84_0329/test.py \
  --dsm_path /path/to/dsm_wgs84.tif \
  --dom_path /path/to/dom_wgs84.tif \
  --name 1 \
  --lon 114.04270934472524 \
  --lat 22.416065873493764 \
  --alt 141.49985356855902 \
  --roll 4.551786987694334 \
  --pitch 1.3641392882272305 \
  --yaw 103.15503822838967 \
  --K_json /home/amax/Documents/code/PiLoT/PiLoT/crop_wgs84_0329/K.json
```

如果要保存：

```bash
python /home/amax/Documents/code/PiLoT/PiLoT/crop_wgs84_0329/test.py \
  --dsm_path /path/to/dsm_wgs84.tif \
  --dom_path /path/to/dom_wgs84.tif \
  --output_dir /path/to/output \
  --save_outputs \
  --save_debug_vis \
  --lon ... --lat ... --alt ... --roll ... --pitch ... --yaw ...
```



python /home/amax/Documents/code/PiLoT/PiLoT/crop_wgs84/test.py   --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKisland_GNSS/dsm_wgs84.tif   --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKisland_GNSS/dom_wgs84.tif   --output_dir /home/amax/Documents/code/PiLoT/PiLoT/crop_wgs84/test   --name sample   --lon 114.25834288619072   --lat 22.205296022324383   --alt 100.75537325828935   --roll 1.4894249861234055   --pitch 2.1523655622721525   --yaw 67.25778027171904   --K_json /home/amax/Documents/code/PiLoT_v2/crop/K.json



python /home/amax/Documents/code/PiLoT_v2/crop_wgs84_0329/test.py   --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM_WGS84.tif   --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DOM_WGS84.tif   --output_dir /home/amax/Documents/code/PiLoT_v2/crop_wgs84_0329/test   --name sample   --lon 112.999167   --lat 28.2933  --alt 53.674    --roll 0.0   --pitch 35.6   --yaw 177.7   --K_json /home/amax/Documents/code/PiLoT_v2/crop_wgs84_0329/K.json --save_outputs --save_debug_vis

python crop_wgs84_0329/test.py \
  --dsm_path /media/amax/AE0E2AFD0E2ABE69/datasets/mapscape/model/usa_2/dsm.tif \
  --dom_path /media/amax/AE0E2AFD0E2ABE69/datasets/mapscape/model/usa_2/dom.tif \
  --output_dir /home/amax/Documents/code/PiLoT_v2/tmp_crop_out \
  --name test1 \
  --lon -87.621232 \
  --lat 41.860991999999996 \
  --alt 500.0 \
  --roll 0.0 \
  --pitch 10.0 \
  --yaw 281.3096633649805 \
  --K_json /home/amax/Documents/code/PiLoT_v2/crop_wgs84_0329/K_mapscape.json \
  --print_corner_intrinsics \
  --save_outputs \
  --save_debug_vis \
  --save_full_dom_points
-87.621232 41.860991999999996 800.0 0.0 10.0 281.3096633649805
-87.621232 41.860991999999996 500.0 0.0 10.0 281.3096633649805
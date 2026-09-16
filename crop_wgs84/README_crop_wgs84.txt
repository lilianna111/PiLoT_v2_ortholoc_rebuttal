crop_wgs84.zip 内容：
- proj2map_test_wgs84.py
- transform_colmap_wgs84.py
- utils.py

本版修改点：
1. 世界坐标统一为 ECEF (EPSG:4978)。
2. WGS84 DSM/DOM 栅格仍以 lon/lat 索引，但所有输出 3D 点云改为 ECEF。
3. pixel_to_world 不再与全局 z=const 平面求交，而是与相机当前位置的局部切平面(指定 geodetic height)求交。
4. 短边方向、旋转矩形构造改为在局部 ENU 平面中完成，再转回 ECEF。
5. geo_coords_to_dsm_index 支持 ECEF 点 -> lon/lat -> 栅格索引。
6. transform_colmap_pose_intrinsic 输出的 pose 世界坐标改为 ECEF。

建议替换方式：
- 把原来 import proj2map_test / transform_colmap 的位置改成导入 *_wgs84 版本。
- 你的地图文件应当是 WGS84 经纬度栅格（通常 EPSG:4326），DSM 高程单位仍为米。

说明：
- 这版保持了你现在“中心点投影 + 短边方向 + 斜向长方形 + mask”的主裁剪方法。
- 代码已通过 Python 语法检查，但没有在你的真实数据上跑端到端验证。

import pyproj
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import math # 需要导入 math 模块

# ==================== 新增：自动获取UTM区域EPSG代码的函数 ====================
def get_utm_epsg_code(lon, lat):
    """
    根据经纬度计算并返回对应的UTM区域的EPSG代码。
    
    Args:
        lon (float): 经度
        lat (float): 纬度
        
    Returns:
        str: UTM投影坐标系的EPSG代码 (例如 'EPSG:32638')
    """
    # 计算UTM区域号
    utm_zone = math.floor((lon + 180) / 6) + 1
    
    # 确定是南半球还是北半球
    # EPSG代码：北半球以 '326' 开头，南半球以 '327' 开头
    epsg_prefix = 32600 if lat >= 0 else 32700
    
    # 构造完整的EPSG代码
    epsg_code = f"EPSG:{epsg_prefix + utm_zone}"
    
    return epsg_code

# ==================== 修改：将 WGS84 转换为本地UTM平面坐标 ====================
def wgs84_to_local_utm(wgs84_coords):
    """
    将WGS84经纬度坐标转换为其所在地的UTM平面坐标（米）。
    
    Args:
        wgs84_coords (list): [lon, lat, height] 格式的WGS84坐标
    """
    lon, lat, height = wgs84_coords
    
    # 动态获取当前位置的UTM EPSG代码
    utm_epsg_code = get_utm_epsg_code(lon, lat)
    print(f"坐标 [{lon:.2f}, {lat:.2f}] 位于 {utm_epsg_code} 区域。")
    
    # 定义源坐标系和目标坐标系
    wgs84_crs = pyproj.CRS('EPSG:4326')
    local_utm_crs = pyproj.CRS(utm_epsg_code)
    
    # 创建转换器
    transformer = pyproj.Transformer.from_crs(wgs84_crs, local_utm_crs, always_xy=True)
    
    # 执行转换
    x, y = transformer.transform(lon, lat)
    
    return [x, y, height]

# ==================== 修改：将本地UTM平面坐标转换回WGS84 ====================
def local_utm_to_wgs84_batch(utm_coords, origin_lon, origin_lat):
    """
    将UTM平面坐标批量转换回WGS84经纬度坐标。
    
    Args:
        utm_coords (np.array): Nx3的UTM坐标数组 [x, y, z]
        origin_lon (float): 用于确定UTM区域的原始经度
        origin_lat (float): 用于确定UTM区域的原始纬度
    """
    x, y = utm_coords[:, 0], utm_coords[:, 1]
    
    # 动态获取UTM EPSG代码
    utm_epsg_code = get_utm_epsg_code(origin_lon, origin_lat)
    
    # 定义源坐标系和目标坐标系
    wgs84_crs = pyproj.CRS('EPSG:4326')
    local_utm_crs = pyproj.CRS(utm_epsg_code)
    
    # 创建转换器
    transformer = pyproj.Transformer.from_crs(local_utm_crs, wgs84_crs, always_xy=True)
    
    # 执行转换
    lons, lats = transformer.transform(x, y)
    
    heights = utm_coords[:, 2]
    
    return np.column_stack((lons, lats, heights))

# 你的原始函数（无需修改）
def load_poses_from_json(file_path):
    coord_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    poses_dict, names_list, origin_pose_dict = {}, [], {}
    with open(file_path, 'r') as f:
        camera_poses = json.load(f)
    for pose_data in camera_poses:
        name = pose_data["OriginalImageName"]
        T4x4_original = np.array(pose_data["T4x4"])
        T4x4_transformed = T4x4_original @ coord_transform
        R_c2w = T4x4_transformed[:3, :3]
        t_c2w = T4x4_transformed[:3, 3]
        euler_angles = R.from_matrix(R_c2w).as_euler('xyz', degrees=True)
        poses_dict[name] = (R_c2w, t_c2w, euler_angles)
        names_list.append(name)
        origin_pose_dict[name] = T4x4_original
    return poses_dict, names_list, origin_pose_dict


# ==================== 主逻辑部分（使用新函数） ====================
target_name = "interval1_AMvalley01"

json_path = f"/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/images/{target_name}/sampleinfos_interpolated.json"
# 原点坐标 [经度, 纬度, 高度]
origin_wgs84 = [44.898231974206659, 39.946784145786722, 1368.9232888183451]  #valley
# origin_wgs84 = [114.04385752853096, 22.416275912068869, 179.65800664446786]  #airport
# origin_wgs84 = [114.04387569569714, 22.415056378573908, 179.66438658301294] 
# origin_wgs84 = [114.26043089616468, 22.207854978330825, 38.890188836725919]  #island
# 22.207854978330825,114.26043089616468

# 1. 将WGS84原点转换为本地UTM平面坐标
origin_utm = wgs84_to_local_utm(origin_wgs84)

with open(json_path, 'r', encoding='utf-8') as f:
    camera_poses_data = json.load(f)

# 坐标系转换矩阵，用于处理旋转
coord_transform = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

c2w_t_list = []
euler_list = []
name_list = []

for pose_entry in camera_poses_data:
    name_list.append(pose_entry["OriginalImageName"])
    
    T4x4 = np.array(pose_entry["T4x4"])
    
    # 2. 将局部相机位移（米）与UTM原点坐标（米）相加
    # 这一步现在是物理意义正确的，因为两者单位和坐标系都一致
    c2w_t_global_utm = T4x4[:3, 3] + origin_utm
    c2w_t_list.append(c2w_t_global_utm)
    
    # 计算旋转（这部分逻辑不变）
    T4x4_transformed = T4x4 @ coord_transform
    R_c2w = T4x4_transformed[:3, :3]
    euler_angles = R.from_matrix(R_c2w).as_euler('xyz', degrees=True)
    euler_list.append(euler_angles)

# 3. 将全局UTM坐标批量转换回WGS84经纬度
# 需要传入原始经纬度来确定正确的UTM转换区域
trans_wgs84 = local_utm_to_wgs84_batch(np.array(c2w_t_list), origin_lon=origin_wgs84[0], origin_lat=origin_wgs84[1])
 
save_pose_path = f"/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/poses/{target_name}.txt"
with open(save_pose_path, 'w') as f:
    for i in range(len(trans_wgs84)):
        # 输出：图像名 经度 纬度 高度 欧拉角...
        f.write(f"{name_list[i]} {trans_wgs84[i][0]} {trans_wgs84[i][1]} {trans_wgs84[i][2]}  {euler_list[i][1]} {euler_list[i][0]} {euler_list[i][2]}\n")

print(f"处理完成，位姿已保存到 {save_pose_path}")



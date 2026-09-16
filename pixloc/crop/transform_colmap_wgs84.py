import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj
import re
import os
import glob


def convert_euler_to_matrix(euler_xyz):
    ret = R.from_euler('xyz', [euler_xyz[1]-90, float(euler_xyz[2]), -euler_xyz[0]], degrees=True)
    rotation_matrix = ret.as_matrix()
    return rotation_matrix


def wgs84_to_ecef(trans):
    """Convert coordinates from WGS84 [lon, lat, h] to ECEF [X, Y, Z]."""
    lon, lat, height = trans

    wgs84 = pyproj.CRS("EPSG:4326")
    ecef = pyproj.CRS("EPSG:4978")

    transformer = pyproj.Transformer.from_crs(wgs84, ecef, always_xy=True)
    x, y, z = transformer.transform(lon, lat, height)

    return np.array([x, y, z], dtype=np.float64)


def transform_colmap_pose_intrinsic(pose_data):
    """
    将 WGS84 坐标的相机位姿和相机内参转换为 ECEF 坐标系，并生成对应字典。
    """
    lon, lat, alt, roll, pitch, yaw = pose_data
    yaw = -yaw
    pitch = pitch - 90

    euler_xyz = [yaw, pitch, roll]
    r_c2w = convert_euler_to_matrix(euler_xyz)

    trans = [lon, lat, alt]
    t_c2w = wgs84_to_ecef(trans)

    T = np.identity(4)
    T[:3, :3] = r_c2w.T
    T[:3, 3] = -r_c2w.T @ t_c2w
    poses_dict = T

    f_mm = 24.00
    intrinsics_dict = np.array([
        [1350.0, 0.0, 957.85],
        [0.0, 1350.0, 537.55],
        [0.0, 0.0, 1.0]
    ])

    osg_dict = euler_xyz
    q_intrinsics_info = [1920, 1080, 1920, 1080, f_mm]

    return poses_dict, intrinsics_dict, q_intrinsics_info, osg_dict
    intrinsics_dict = np.array([
        [735.5326, 0,       586.1788],
        [0,        735.5326, 523.1838],
        [0,        0,       1]
    ], dtype=np.float64)

    osg_dict = euler_xyz
    q_intrinsics_info = [1920, 1080, 1920, 1080, 24.00]

    return poses_dict, intrinsics_dict, q_intrinsics_info, osg_dict
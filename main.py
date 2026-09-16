import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)     # Linux 默认 fork → 改成 spawn
import threading
from pixloc.utils.data import Paths
import os
import sys
import queue
import glob
import cv2
import shutil
import argparse
import ast
import json
import numpy as np
from tqdm import tqdm
from pprint import pformat
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate
from pixloc.utils.get_depth import get_3D_samples_v3, pad_to_multiple, generate_render_camera, get_3D_samples_v2
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg, WGS84_to_ECEF
from pixloc.utils import video_generation
from pixloc.pixlib.datasets.view import read_image_list
import time
import yaml
import copy
import logging
import torch
from multiprocessing import Process, Queue, Event
from crop.crop.transform_colmap import (
    normalize_K,
    ecef_to_wgs84 as crop_ecef_to_wgs84,
    enu_to_ecef_rotation as crop_enu_to_ecef_rotation,
    wgs84_to_ecef as crop_wgs84_to_ecef,
)
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S")
'''
1. config init rot, init translation, datapath
'''
def get_init(pose_file):
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # if '14000' in parts[0]:
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                # pitch, roll, yaw, lon, lat, alt,  = map(float, parts[1: ])
                
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                origin = WGS84_to_ECEF(translation)
                break
    return euler_angles, translation, origin
def load_poses(pose_file, origin = None):
    """Load poses from the pose file."""
    pose_dict = {}
    translation_list = []
    euler_angles_list = []
    name_list = []
    init_euler = None
    init_trans = None
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                translation_list.append([lon, lat, alt])
                euler_angles_list.append([pitch, roll, yaw])
                if '_' not in parts[0]:
                    name = parts[0][:-4] +'_0.png'
                else:
                    name = parts[0]
                name_list.append(name)
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                T_in_ECEF_c2w = euler_angles_to_matrix_ECEF(euler_angles, translation)
                pose_dict[name] = {}
                pose_dict[name]['T_w2c_4x4'] = copy.deepcopy(T_in_ECEF_c2w)
                T_in_ECEF_c2w[:3, 1] = -T_in_ECEF_c2w[:3, 1]  # Y轴取反，投影后二维原点在左上角
                T_in_ECEF_c2w[:3, 2] = -T_in_ECEF_c2w[:3, 2]  # Z轴取反
                T_in_ECEF_c2w[:3, 3] -= origin  # t_c2w - origin
                render_T_w2c = np.eye(4)
                render_T_w2c[:3, :3] = T_in_ECEF_c2w[:3, :3].T
                render_T_w2c[:3, 3] = -T_in_ECEF_c2w[:3, :3].T @ T_in_ECEF_c2w[:3, 3]
                
                
                pose_dict[name]['euler'] = [pitch, roll, yaw]
                pose_dict[name]['trans'] = [lon, lat, alt]
                
                
                render_T_w2c = Pose.from_Rt(render_T_w2c[:3, :3], render_T_w2c[:3, 3])
                pose_dict[name]['T_w2c'] = render_T_w2c.to_flat()
    return pose_dict
def load_k_and_size(k_json_path=None):
    import json
    if k_json_path is None:
        k_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop", "K_uavscene.json")
    with open(k_json_path, "r") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float64)
    image_width = int(data["image_width"])
    image_height = int(data["image_height"])
    K = normalize_K(K)
    return K, image_width, image_height


def ensure_ortholoc_import_path():
    ortholoc_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OrthoLoC", "OrthoLoC")
    if ortholoc_root not in sys.path:
        sys.path.insert(0, ortholoc_root)
    return ortholoc_root


def load_ortholoc_components():
    ensure_ortholoc_import_path()
    from ortholoc import utils as ortholoc_utils
    from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI, MATCHER_ZOO
    return ortholoc_utils, MatcherIMCUI, MATCHER_ZOO


_CGCS2000_TO_WGS84_TRANSFORMER = None


def cgcs2000_to_wgs84(x, y):
    global _CGCS2000_TO_WGS84_TRANSFORMER
    if _CGCS2000_TO_WGS84_TRANSFORMER is None:
        import pyproj

        _CGCS2000_TO_WGS84_TRANSFORMER = pyproj.Transformer.from_crs(
            "EPSG:4547", "EPSG:4326", always_xy=True
        )
    return _CGCS2000_TO_WGS84_TRANSFORMER.transform(x, y)


def cgcs2000_grid_to_local_ecef(points3d_cgcs, valid_mask):
    points3d_cgcs = np.asarray(points3d_cgcs, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    lon, lat = cgcs2000_to_wgs84(points3d_cgcs[..., 0], points3d_cgcs[..., 1])
    x_ecef, y_ecef, z_ecef = crop_wgs84_to_ecef(lon, lat, points3d_cgcs[..., 2])
    points3d_ecef = np.stack([x_ecef, y_ecef, z_ecef], axis=-1).astype(np.float64)

    valid_mask = valid_mask & np.isfinite(points3d_ecef).all(axis=-1)
    if not np.any(valid_mask):
        raise ValueError("crop has no valid ECEF points")

    ecef_origin = points3d_ecef[valid_mask].mean(axis=0).astype(np.float64)
    points3d_local_ecef = points3d_ecef - ecef_origin
    points3d_local_ecef[~valid_mask] = np.nan
    return points3d_local_ecef.astype(np.float32), ecef_origin, valid_mask


def add_ecef_origin_to_pose(pose_c2w, ecef_origin):
    pose_c2w = np.asarray(pose_c2w, dtype=np.float64).copy()
    pose_c2w[:3, 3] += np.asarray(ecef_origin, dtype=np.float64)
    return pose_c2w


def ortholoc_pose_c2w_to_crop_pose(pose_c2w):
    from scipy.spatial.transform import Rotation as Rotation

    pose_c2w = np.asarray(pose_c2w, dtype=np.float64)
    if pose_c2w.shape == (3, 4):
        pose_c2w_4x4 = np.eye(4, dtype=np.float64)
        pose_c2w_4x4[:3, :] = pose_c2w
        pose_c2w = pose_c2w_4x4
    if pose_c2w.shape != (4, 4):
        raise ValueError(f"pose_c2w must be 3x4 or 4x4, got {pose_c2w.shape}")

    r_ecef_from_cam = pose_c2w[:3, :3]
    t_c2w_ecef = pose_c2w[:3, 3]
    lon, lat, alt = crop_ecef_to_wgs84(*t_c2w_ecef)

    r_ecef_from_enu = crop_enu_to_ecef_rotation(float(lon), float(lat))
    r_local_c2w = r_ecef_from_enu.T @ r_ecef_from_cam
    euler_xyz = Rotation.from_matrix(r_local_c2w).as_euler("xyz", degrees=True)

    # Inverse of crop.transform_colmap.convert_euler_to_matrix():
    # r_local_c2w = R.from_euler("xyz", [pitch - 180, roll, yaw]).
    pitch = float(euler_xyz[0] + 180.0)
    if pitch > 180.0:
        pitch -= 360.0
    roll = float(euler_xyz[1])
    yaw = float(euler_xyz[2])

    return np.array([pitch, roll, yaw], dtype=np.float64), np.array([lon, lat, alt], dtype=np.float64)


def angle_diff_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def max_euler_diff_deg(euler_a, euler_b):
    euler_a = np.asarray(euler_a, dtype=np.float64)
    euler_b = np.asarray(euler_b, dtype=np.float64)
    return max(angle_diff_deg(a, b) for a, b in zip(euler_a, euler_b))


def translation_diff_m(trans_a, trans_b):
    xyz_a = crop_wgs84_to_ecef(*np.asarray(trans_a, dtype=np.float64))
    xyz_b = crop_wgs84_to_ecef(*np.asarray(trans_b, dtype=np.float64))
    return float(np.linalg.norm(np.asarray(xyz_a) - np.asarray(xyz_b)))


def pose_values_for_txt(euler_pitch_roll_yaw, trans_lon_lat_alt):
    euler = np.asarray(euler_pitch_roll_yaw, dtype=np.float64)
    trans = np.asarray(trans_lon_lat_alt, dtype=np.float64)
    return [
        float(trans[0]),
        float(trans[1]),
        float(trans[2]),
        float(euler[1]),
        float(euler[0]),
        float(euler[2]),
    ]


def format_float(value):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.10f}"


def format_pose_values(values):
    return ",".join(format_float(value) for value in values)


def format_matrix_flat(matrix):
    if matrix is None:
        return ""
    arr = np.asarray(matrix, dtype=np.float64).reshape(-1)
    return ",".join(format_float(value) for value in arr)


def run_ortholoc_localization_on_crop(
    img_path,
    image_dop,
    dsm_grid,
    intrinsics_matrix,
    matcher,
    angles,
    min_conf=0.5,
    reprojection_error=5.0,
    pnp_mode="poselib",
    num_points=None,
):
    ortholoc_utils, _, _ = load_ortholoc_components()

    image_query = ortholoc_utils.io.load_image(img_path)
    height, width = image_query.shape[:2]
    h_dop, w_dop = image_dop.shape[:2]
    logging.info(
        "OrthoLoC matcher input before preprocessing: query=%dx%d crop=%dx%d image=%s",
        width,
        height,
        w_dop,
        h_dop,
        os.path.basename(img_path),
    )

    all_correspondences_2d2d = matcher.run(image_query, image_dop, angles=angles, normalized=True)
    if not all_correspondences_2d2d:
        raise RuntimeError("OrthoLoC matcher returned no correspondence candidates.")

    best_angle_idx = max(
        range(len(all_correspondences_2d2d)),
        key=lambda i: len(all_correspondences_2d2d[i].take_min_conf(min_conf)),
    )
    correspondences_2d2d = all_correspondences_2d2d[best_angle_idx]
    best_angle = matcher.angles[best_angle_idx]
    num_matches_raw = len(correspondences_2d2d)

    correspondences_2d2d = correspondences_2d2d.take_min_conf(
        min_conf=min_conf, keep_at_least=num_points, inclusive=False
    )
    correspondences_2d2d = correspondences_2d2d.take_covisible(
        h0=height, w0=width, h1=h_dop, w1=w_dop, is_normalized=True
    )
    if not correspondences_2d2d.is_valid:
        raise RuntimeError("No valid correspondences after confidence/covisibility filtering.")

    correspondences_2d3d = correspondences_2d2d.to_2d3d(grid3d_1=dsm_grid)
    success, pose_c2w_pred, intrinsics_matrix_pred, inliers_mask, reprojection_errors = correspondences_2d3d.calibrate(
        num_points=num_points,
        intrinsics_matrix=intrinsics_matrix,
        width=width,
        height=height,
        reprojection_error=reprojection_error,
        pnp_mode=pnp_mode,
    )
    if not success or pose_c2w_pred is None:
        raise RuntimeError("OrthoLoC PnP/calibration failed.")

    pose_c2w_4x4 = np.eye(4, dtype=np.float64)
    pose_c2w_4x4[:3, :] = np.asarray(pose_c2w_pred, dtype=np.float64)
    pose_w2c_4x4 = np.asarray(ortholoc_utils.pose.inv_pose(pose_c2w_4x4), dtype=np.float64)
    num_inliers = int(np.count_nonzero(inliers_mask)) if inliers_mask is not None else None
    median_reproj = float(np.nanmedian(reprojection_errors)) if reprojection_errors is not None else None

    return {
        "pose_c2w": pose_c2w_4x4,
        "pose_w2c": pose_w2c_4x4,
        "intrinsics": np.asarray(intrinsics_matrix_pred, dtype=np.float64) if intrinsics_matrix_pred is not None else None,
        "best_angle": float(best_angle),
        "num_matches_raw": int(num_matches_raw),
        "num_matches_filtered": int(len(correspondences_2d2d)),
        "num_inliers": num_inliers,
        "median_reprojection_error": median_reproj,
    }
# def process_map_crop(ref_DSM_path, pose_data, ref_npy_path, name, map_data_pack, paths_pack, ray_area, ray_area_minZ):        # 包含输出路径
#     """
#     单帧处理函数：只负责计算和裁剪，不负责加载大地图
#     """

#     # 1. 解包地图数据 (这些数据在内存里，很快)
#     geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = map_data_pack
#     ref_rgb_path, ref_depth_path = paths_pack

#     # 2. 坐标转换 (Read prior pose and intrinsic)
#     query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
    
#     # 3. 执行裁剪 (Crop map)
#     # 这一步会把图片保存到硬盘
#     data = generate_ref_map(
#         ref_DSM_path, pose_data,
#         ref_npy_path, geotransform,
#         query_intrinsics_dict, 
#         query_prior_poses_dict, 
#         name, 
#         area_minZ, 
#         dsm_data, 
#         dsm_trans, 
#         dom_data, 
#         ref_rgb_path, 
#         ref_depth_path,
#         ray_area,
#         ray_area_minZ,
#         crop_padding=0, 
#         debug=False
#     )
#     # print("✅ process_map_crop Ended! data:", data)
def process_map_crop(ref_DSM_path, pose_data, ref_npy_path, name, map_data_pack, paths_pack,
                     ray_area, ray_area_minZ, locator=None):
    """
    单帧处理函数：只负责计算和裁剪，不负责加载大地图
    """
    from pixloc.crop.transform_colmap import transform_colmap_pose_intrinsic as crop_transform_colmap_pose_intrinsic
    from pixloc.crop.proj2map import generate_ref_map

    geotransform, area, area_minZ, dsm_data, dsm_trans, dom_data = map_data_pack
    ref_rgb_path, ref_depth_path = paths_pack

    t_pose0 = time.perf_counter()
    query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = (
        crop_transform_colmap_pose_intrinsic(pose_data)
    )
    t_pose1 = time.perf_counter()

    data = generate_ref_map(
        ref_DSM_path, pose_data,
        ref_npy_path, geotransform,
        query_intrinsics_dict,
        query_prior_poses_dict,
        name,
        area_minZ,
        dsm_data,
        dsm_trans,
        dom_data,
        ref_rgb_path,
        ref_depth_path,
        ray_area,
        ray_area_minZ,
        crop_padding=0,
        debug=False,
        save_to_disk=False,
        locator=locator,
    )
    timings = data.get('timings', {}).copy()
    timings['pose_transform_s'] = t_pose1 - t_pose0
    timings['crop_total_s'] = timings.get('total_s', 0.0) + timings['pose_transform_s']
    data['timings'] = timings
    return data


def postprocess_crop_to_render_grid(color, points3d, valid_mask, target_w, target_h, enable_padding):
    color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    points3d = cv2.resize(points3d.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    mask = valid_mask.astype(np.uint8)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    if enable_padding:
        pad_h = (16 - target_h % 16) % 16
        pad_w = (16 - target_w % 16) % 16
        if pad_h or pad_w:
            color = cv2.copyMakeBorder(color, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            points3d = cv2.copyMakeBorder(points3d, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    return color, points3d, mask.astype(bool)

class DualProcessTask:        
    def __init__(self, config, init_euler = None, init_trans = None, name = None, args=None):
        # 用 multiprocessing 队列/事件
        self.task_q   = Queue(maxsize=2)     # 渲染 → 定位
        self.pose_q   = Queue(maxsize=3)     # 定位 → 渲染
        self.stop_evt = Event()
        self.args = args
        self.render_config = config["render_config"]
        default_confs = config["default_confs"] 
        self.conf = default_confs['from_render_test'] # from_render_test
        # conf初始化
        folder_path = default_confs['dataset_path']
        dataset_name = default_confs['dataset_name']
        output_name = default_confs['dataset_name']
        # self.euler_angles, self.translation = self.render_config['init_rot'], self.render_config['init_trans']
        
        if name is not None:
            dataset_name = name
            output_name = name
        # if init_euler is not None:
        #     self.render_config['init_rot'], self.render_config['init_trans'] = init_euler, init_trans
        #     self.euler_angles = init_euler
        #     self.translation = init_trans
        self.refine_conf = default_confs['refine']
        self.mul = self.refine_conf['mul']
        self.ortholoc_matcher = getattr(args, "ortholoc_matcher", "Mast3R")
        # self.estimated_pose = os.path.join(folder_path, 'estimation', 'FPVLoc@'+output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        # output_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth"
        # output_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/uavscene"
        # output_folder = "/home/ps/Documents/liuxy24/PiLoT_v55/usa_2/none"
        matcher_dir = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in self.ortholoc_matcher
        ).strip("_") or "matcher"
        output_folder = os.path.join(
            "/media/amax/PS2000/ortholoc",
            matcher_dir,
        )
        # output_folder = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth_weixing"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        pre_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/mapscape/test"
        # output_folder = "outputs"
        self.outputs = os.path.join(output_folder, output_name)
        self.pre_path = os.path.join(pre_path, output_name)
        if not os.path.exists(self.outputs):
            os.makedirs(self.outputs)
        else:
            shutil.rmtree(self.outputs)
            os.makedirs(self.outputs)
        if not os.path.exists(self.pre_path):
            os.makedirs(self.pre_path)
        else:
            shutil.rmtree(self.pre_path)
            os.makedirs(self.pre_path)
            
        self.estimated_pose = os.path.join(output_folder, output_name +'.txt') #'FPVLoc@'+ dataset_name +'.txt'
        self.estimated_pose_in_dir = os.path.join(self.outputs, "poses.txt")
        self.ortholoc_pose_json = os.path.join(self.outputs, "poses.json")
        self.ortholoc_pose_debug_txt = os.path.join(self.outputs, "ortholoc_pose_debug.txt")
        self.reset_count_txt = os.path.join(self.outputs, "reset_counts.txt")
        self.ortholoc_match_figures = os.path.join(self.outputs, "figures")
        self.crop_vis_dir = os.path.join(self.outputs, "crop_dom")
        os.makedirs(self.crop_vis_dir, exist_ok=True)
        
        self.gt_pose = os.path.join(folder_path, 'poses', dataset_name +'.txt') #

        self.crop_k_json = getattr(args, "crop_k_json", None)
        self.dom_path = os.path.expanduser(getattr(args, "dom_path", ""))
        self.dsm_path = os.path.expanduser(getattr(args, "dsm_path", ""))
        self.ortholoc_intrinsics = os.path.expanduser(getattr(args, "ortholoc_intrinsics", ""))
        self.ortholoc_device = getattr(args, "ortholoc_device", "cuda")
        self.ortholoc_angles = [float(angle) for angle in getattr(args, "ortholoc_angles", [0])]
        self.ortholoc_min_conf = getattr(args, "ortholoc_min_conf", 0.5)
        self.ortholoc_reprojection_error = getattr(args, "ortholoc_reprojection_error", 5.0)
        self.ortholoc_pnp_mode = getattr(args, "ortholoc_pnp_mode", "poselib")
        self.ortholoc_num_points = getattr(args, "ortholoc_num_points", None)
        self.min_ortholoc_crop_size = getattr(args, "min_ortholoc_crop_size", 32)
        self.max_pose_jump_m = getattr(args, "max_pose_jump_m", 100.0)
        self.max_euler_jump_deg = getattr(args, "max_euler_jump_deg", 100.0)
        self.max_gt_recrops = getattr(args, "max_gt_recrops", 1)
        self.gt_reset_translation_thresh_m = getattr(args, "gt_reset_translation_thresh_m", 50.0)
        self.gt_reset_rotation_thresh_deg = getattr(args, "gt_reset_rotation_thresh_deg", 50.0)
        self.continue_on_error = getattr(args, "continue_on_error", False)
        self.run_evaluate = getattr(args, "evaluate", False)
        
        self.last_frame_info = {}
        self.last_frame_info['observations'] = []
        self.last_frame_info['refine_conf'] = self.refine_conf
        
        print(f'conf:\n{pformat(self.conf)}')
        # 初始化先验位姿和内参
        self.name_q = None
        # camera
        self.query_resize_ratio = default_confs['cam_query']['width'] / default_confs['cam_query']['max_size']
        fx, fy, cx, cy = default_confs['cam_query']['params'] 
        w, h = default_confs['cam_query']['width'], default_confs['cam_query']['height']
        
        raw_query_camera = np.array([w, h, cx, cy, fx, fy])
        self.render_camera_osg = raw_query_camera / self.query_resize_ratio
        
        default_confs['cam_query']['params']  = np.array(default_confs['cam_query']['params']) / self.query_resize_ratio
        default_confs['cam_query']['width'], default_confs['cam_query']['height'] = default_confs['cam_query']['width'] / self.query_resize_ratio, default_confs['cam_query']['height']/self.query_resize_ratio
        cam_query = default_confs["cam_query"]
        self.query_camera = Camera.from_colmap(cam_query) #! 2.
        
        img_path = os.path.join(folder_path, 'images', dataset_name)
        self.img_list = glob.glob(img_path + "/*.png") + glob.glob(img_path + "/*.jpg") + glob.glob(img_path + "/*.JPG")
        # 支持 1671606440.199954033.jpg 这类带小数的时间戳：按数值排序，避免同秒多图顺序错乱
        def _img_sort_key(path):
            name = os.path.basename(path).rsplit(".", 1)[0]
            if "_" in name:
                prefix, suffix = name.split("_", 1)
                if prefix.isdigit():
                    # 兼容 1_0.png 这类命名，先按下划线前的帧号排序。
                    return (0, int(prefix), suffix)
            try:
                return (1, float(name))
            except ValueError:
                return (2, name)
        self.img_list = sorted(self.img_list, key=_img_sort_key)
        # self.img_list = self.img_list[:1]
        
        self.dd = None
        self.name_r = None

        self.render_camera = generate_render_camera(self.render_camera_osg).float()
        self.render_config['render_camera'] = self.render_camera_osg 

        # 是否padding, num init  pose
        self.num_init_pose = default_confs['num_init_pose']
        self.padding = default_confs['padding']
        self.euler_angles, self.translation, self.origin = get_init(self.gt_pose)
        self.render_config['init_rot'], self.render_config['init_trans'] = self.euler_angles, self.translation
        default_confs['refine']['origin'] = self.origin
        self.gt_pose_dict = load_poses(self.gt_pose, origin = self.origin)
        
        self.device = 'cuda'
        self.origin = torch.tensor(self.origin, device=self.device)
        self.query_camera, self.render_camera = self.query_camera.to(self.device), self.render_camera.to(self.device)

        self.pose_q.put_nowait({
            "frame_idx": 0,
            "euler": np.asarray(self.euler_angles, dtype=np.float64),
            "trans": np.asarray(self.translation, dtype=np.float64),
            "reset_count": 0,
            "pose_source": "init",
        })

    # ---------------- 裁剪线程（取代原渲染） ----------------        
    def rendering_worker(self):
        import torch
        print("✅ Rendering Worker Started! pid:", os.getpid())
        ref_DOM_path = self.dom_path
        ref_DSM_path = self.dsm_path
        if not os.path.exists(ref_DSM_path):
            raise FileNotFoundError(f"找不到文件: {ref_DSM_path}")
        if not os.path.exists(ref_DOM_path):
            raise FileNotFoundError(f"找不到文件: {ref_DOM_path}")

        logging.info(f"Loading Map from {ref_DOM_path}...")

        ref_npy_path = os.path.splitext(ref_DSM_path)[0] + ".npy"
        ref_rgb_path = self.pre_path
        ref_depth_path = self.pre_path
        paths_pack = (ref_rgb_path, ref_depth_path)

        from pixloc.crop.utils import read_DSM_config
        from pixloc.crop.ray_casting import TargetLocation

        map_data_pack = read_DSM_config(ref_DSM_path, ref_DOM_path, ref_npy_path)
        _, ray_area, ray_area_minZ, _, _, _ = map_data_pack
        locator = TargetLocation({"ray_casting": {}}, use_dsm=False)

        logging.info("Map Loaded Successfully.")

        next_idx = 0
        fps_log_every = 0
        # === 3. 主循环：按位姿裁剪 ===
        while True:
            try:
                item = self.pose_q.get(timeout=1)
            except queue.Empty:
                if self.stop_evt.is_set():
                    break
                continue
            if item is None:
                break
            if isinstance(item, dict):
                frame_idx = int(item.get("frame_idx", next_idx))
                euler = np.asarray(item["euler"], dtype=np.float64)
                trans = np.asarray(item["trans"], dtype=np.float64)
                reset_count = int(item.get("reset_count", 0))
                pose_source = item.get("pose_source", "unknown")
            else:
                frame_idx = next_idx
                euler, trans = item  # 欧拉角 (pitch, roll, yaw) 与平移 (lon, lat, alt)
                euler = np.asarray(euler, dtype=np.float64)
                trans = np.asarray(trans, dtype=np.float64)
                reset_count = 0
                pose_source = "legacy"
            if frame_idx >= len(self.img_list):
                logging.info("已生成全部 %d 帧 crop，忽略多余 pose 并结束 crop worker", len(self.img_list))
                break
            euler_pitch, euler_roll, euler_yaw = euler[0], euler[1], euler[2]
            euler_crop = [euler_roll, euler_pitch, euler_yaw]
            pose_data = list(trans) + list(euler_crop)
            # 用读入的图片名（如 1671606440.199954033）作为保存名，与 img_list 顺序一致
            img_path = self.img_list[frame_idx]
            name = os.path.splitext(os.path.basename(img_path))[0]

            t0 = time.perf_counter()
            tt0 = time.time()
            try:
                crop_result = process_map_crop(
                    ref_DSM_path,
                    pose_data,
                    ref_npy_path,
                    name,
                    map_data_pack,
                    paths_pack,
                    ray_area,
                    ray_area_minZ,
                    locator=locator,
                )
            except Exception as exc:
                logging.exception(f"process_map_crop 失败 name={name}: {exc}")
                self.stop_evt.set()
                break
            tt1 = time.time()
            # print(f"process_map_crop 耗时: {tt1-tt0} 秒")

            # img_file = os.path.join(ref_rgb_path, f'{name}_dom.png')
            # npy_file = os.path.join(ref_depth_path, f'{name}_dsm.npy')
            # if (not os.path.exists(img_file)) or (not os.path.exists(npy_file)):
            #     logging.error(f"裁剪结果缺失: {img_file} or {npy_file}")
            #     continue
            color = crop_result["dom_crop"]
            # save_path = os.path.join(self.outputs, f'{name}_dom.png')
            # cv2.imwrite(save_path, color)
            # logging.info(f"Saved dom_warp image: {save_path}")
            points3d = crop_result["point_cloud_crop"]
            # color = cv2.imread(img_file, cv2.IMREAD_COLOR)
            # if color is None:
            if color is None or color.size == 0:
                logging.error(f"读取裁剪 RGB 失败")
                self.stop_evt.set()
                break
            # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)    #作用：将BGR格式转换为RGB格式

            valid_mask = np.isfinite(points3d).all(axis=-1) & (points3d[..., 2] > 0)
            if not np.any(valid_mask):
                logging.error(f"裁剪结果无有效3D点")
                self.stop_evt.set()
                break

            target_w = int(self.render_camera_osg[0])
            target_h = int(self.render_camera_osg[1])
            color, points3d, valid_mask = postprocess_crop_to_render_grid(
                color=color,
                points3d=points3d,
                valid_mask=valid_mask,
                target_w=target_w,
                target_h=target_h,
                enable_padding=self.padding,
            )

            if not np.any(valid_mask):
                logging.error(f"裁剪结果无有效3D点(经过resize/padding后): {name}")
                self.stop_evt.set()
                break

            try:
                points3d_ecef, ecef_origin, valid_mask = cgcs2000_grid_to_local_ecef(points3d, valid_mask)
            except Exception as exc:
                logging.exception(f"裁剪3D点转ECEF失败 name={name}: {exc}")
                self.stop_evt.set()
                break

            t_render = (time.perf_counter() - t0) * 1e3  # ms
            fps_log_every += 1
            # if fps_log_every % 30 == 0:
            #     logging.info("crop: %.2f ms", t_render)

            crop_save_path = os.path.join(self.crop_vis_dir, f'{name}.png')
            cv2.imwrite(crop_save_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            logging.info(f"Saved crop image: {crop_save_path}")

            crop_h, crop_w = color.shape[:2]
            if min(crop_h, crop_w) < self.min_ortholoc_crop_size:
                logging.warning(
                    f"crop 尺寸小于建议阈值但仍继续 OrthoLoC: name={name}, size={crop_w}x{crop_h}"
                )

            try:
                self.task_q.put(
                    {
                        "name": name,
                        "img_path": img_path,
                        "frame_idx": frame_idx,
                        "reset_count": reset_count,
                        "pose_source": pose_source,
                        "dom_warp": color,
                        "xyz_ecef": points3d_ecef,
                        "ecef_origin": ecef_origin,
                        "render_euler": np.asarray(euler, dtype=np.float64),
                        "render_trans": np.asarray(trans, dtype=np.float64),
                    },
                    timeout=1,
                )
            except queue.Full:
                break
            next_idx = max(next_idx, frame_idx + 1)

        self.stop_evt.set()
        self.task_q.put(None)
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        # logging.info('Render/Crop process done')
    # ---------------- 定位线程 ----------------
    def localization_worker(self):
        print("✅ Localization Worker Started! pid:", os.getpid())
        error_path = os.path.join(self.outputs, "localization_error.txt")
        fault_path = os.path.join(self.outputs, "localization_faulthandler.txt")
        fault_file = None
        stage = "load_ortholoc_components"
        try:
            import faulthandler

            os.makedirs(self.outputs, exist_ok=True)
            fault_file = open(fault_path, "w", encoding="utf-8", buffering=1)
            faulthandler.enable(file=fault_file, all_threads=True)
            logging.info("Localization init: loading OrthoLoC components")
            ortholoc_utils, MatcherIMCUI, MATCHER_ZOO = load_ortholoc_components()
            if self.ortholoc_matcher not in MATCHER_ZOO:
                raise ValueError(f"Unsupported OrthoLoC matcher: {self.ortholoc_matcher}")

            stage = "load_camera_params"
            logging.info("Localization init: loading intrinsics from %s", self.ortholoc_intrinsics)
            _, intrinsics_matrix = ortholoc_utils.io.load_camera_params(self.ortholoc_intrinsics)
            if intrinsics_matrix is None:
                raise ValueError(f"Failed to load intrinsics from {self.ortholoc_intrinsics}")

            stage = "init_matcher"
            logging.info("Localization init: creating matcher %s on %s", self.ortholoc_matcher, self.ortholoc_device)
            matcher = MatcherIMCUI(
                name=self.ortholoc_matcher,
                device=self.ortholoc_device,
                angles=self.ortholoc_angles,
            )
            logging.info("Localization init finished")
        except Exception as exc:
            import traceback

            os.makedirs(self.outputs, exist_ok=True)
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"stage: {stage}\n")
                f.write(traceback.format_exc())
            logging.exception("Localization init failed at %s: %s", stage, exc)
            self.stop_evt.set()
            try:
                self.pose_q.put(None, timeout=1)
            except Exception:
                pass
            return

        aggregated_poses = []
        failed_images = []
        reset_records = []
        gt_reset_requests = 0

        debug_header = [
            "idx",
            "image",
            "status",
            "reset_count",
            "pose_source",
            "reason",
            "crop_size",
            "prior_lon_lat_alt_roll_pitch_yaw",
            "ortholoc_lon_lat_alt_roll_pitch_yaw",
            "next_crop_lon_lat_alt_roll_pitch_yaw",
            "gt_error_m",
            "gt_error_deg",
            "best_angle",
            "matches_raw",
            "matches_filtered",
            "inliers",
            "median_reproj",
            "ecef_origin",
            "pose_c2w_flat",
            "pose_w2c_flat",
        ]
        with open(self.ortholoc_pose_debug_txt, "w", encoding="utf-8") as f_debug:
            f_debug.write("\t".join(debug_header) + "\n")

        def write_pose_debug(
            frame_idx,
            image_name,
            status,
            reason,
            crop_size,
            reset_count,
            pose_source,
            prior_euler,
            prior_trans,
            next_euler,
            next_trans,
            ortho_ret=None,
            ortho_euler=None,
            ortho_trans=None,
            jump_m=None,
            jump_deg=None,
        ):
            row = [
                str(frame_idx),
                image_name,
                status,
                str(reset_count),
                pose_source,
                "" if reason is None else str(reason).replace("\t", " "),
                crop_size,
                format_pose_values(pose_values_for_txt(prior_euler, prior_trans)),
                format_pose_values(pose_values_for_txt(ortho_euler, ortho_trans)) if ortho_euler is not None and ortho_trans is not None else "",
                format_pose_values(pose_values_for_txt(next_euler, next_trans)),
                format_float(jump_m),
                format_float(jump_deg),
                format_float(ortho_ret.get("best_angle") if ortho_ret else None),
                format_float(ortho_ret.get("num_matches_raw") if ortho_ret else None),
                format_float(ortho_ret.get("num_matches_filtered") if ortho_ret else None),
                format_float(ortho_ret.get("num_inliers") if ortho_ret else None),
                format_float(ortho_ret.get("median_reprojection_error") if ortho_ret else None),
                format_matrix_flat(ortho_ret.get("ecef_origin") if ortho_ret else None),
                format_matrix_flat(ortho_ret.get("pose_c2w") if ortho_ret else None),
                format_matrix_flat(ortho_ret.get("pose_w2c") if ortho_ret else None),
            ]
            line = "\t".join(row)
            with open(self.ortholoc_pose_debug_txt, "a", encoding="utf-8") as f_debug:
                f_debug.write(line + "\n")
            logging.info(
                "POSE_DEBUG %s frame=%s prior=%s ortho=%s next_crop=%s reason=%s",
                status,
                image_name,
                row[7],
                row[8] or "None",
                row[9],
                row[5],
            )

        def enqueue_pose(frame_idx, euler, trans, reset_count=0, pose_source="ortholoc"):
            self.pose_q.put(
                {
                    "frame_idx": int(frame_idx),
                    "euler": np.asarray(euler, dtype=np.float64),
                    "trans": np.asarray(trans, dtype=np.float64),
                    "reset_count": int(reset_count),
                    "pose_source": pose_source,
                }
            )

        def pose_result_line(image_name, euler, trans):
            return (
                f"{image_name} {' '.join(map(str, np.asarray(trans, dtype=np.float64).tolist()))} "
                f"{' '.join(map(str, [float(euler[1]), float(euler[0]), float(euler[2])]))}"
            )

        pose_output_paths = [self.estimated_pose, self.estimated_pose_in_dir]
        for pose_output_path in pose_output_paths:
            with open(pose_output_path, "w", encoding="utf-8"):
                pass

        def record_pose_result(image_name, euler, trans):
            line = pose_result_line(image_name, euler, trans)
            for pose_output_path in pose_output_paths:
                with open(pose_output_path, "a", encoding="utf-8") as f_pose:
                    f_pose.write(line + "\n")
                    f_pose.flush()
            return line

        def append_reset_record(frame_idx, image_name, reset_count, pose_source, status, reason):
            reset_records.append(
                {
                    "frame_idx": int(frame_idx),
                    "image": image_name,
                    "reset_count": int(reset_count),
                    "pose_source": pose_source,
                    "status": status,
                    "reason": "" if reason is None else str(reason),
                }
            )

        next_idx = 0
        while True:
            item = self.task_q.get()
            if item is None:
                break

            crop_name = item["name"]
            frame_idx = int(item.get("frame_idx", next_idx))
            reset_count = int(item.get("reset_count", 0))
            pose_source = item.get("pose_source", "unknown")
            img_path = item.get("img_path") or (self.img_list[frame_idx] if frame_idx < len(self.img_list) else None)
            if img_path is None:
                logging.warning("收到没有 img_path 的多余 crop item，结束 localization worker: idx=%d name=%s", frame_idx, crop_name)
                break
            render_euler = np.asarray(item["render_euler"], dtype=np.float64)
            render_trans = np.asarray(item["render_trans"], dtype=np.float64)
            dom_warp = item.get("dom_warp")
            xyz_ecef = item.get("xyz_ecef")
            ecef_origin = item.get("ecef_origin")
            if ecef_origin is not None:
                ecef_origin = np.asarray(ecef_origin, dtype=np.float64)
            qname = os.path.basename(img_path)
            if dom_warp is None:
                crop_size = "0x0"
            else:
                crop_h, crop_w = dom_warp.shape[:2]
                crop_size = f"{crop_w}x{crop_h}"

            if dom_warp is None or xyz_ecef is None or ecef_origin is None:
                reason = item.get("skip_reason", "unknown")
                logging.error(f"OrthoLoC 输入无效，停止定位 frame={qname}: {reason}")
                failed_images.append(qname)
                write_pose_debug(
                    frame_idx=frame_idx,
                    image_name=qname,
                    status="invalid_crop",
                    reason=reason,
                    crop_size=crop_size,
                    reset_count=reset_count,
                    pose_source=pose_source,
                    prior_euler=render_euler,
                    prior_trans=render_trans,
                    next_euler=render_euler,
                    next_trans=render_trans,
                )
                self.stop_evt.set()
                break

            ortho_ret = None
            pred_euler = None
            pred_translation = None
            jump_m = None
            jump_deg = None
            try:
                ortho_ret = run_ortholoc_localization_on_crop(
                    img_path=img_path,
                    image_dop=dom_warp,
                    dsm_grid=xyz_ecef,
                    intrinsics_matrix=intrinsics_matrix,
                    matcher=matcher,
                    angles=self.ortholoc_angles,
                    min_conf=self.ortholoc_min_conf,
                    reprojection_error=self.ortholoc_reprojection_error,
                    pnp_mode=self.ortholoc_pnp_mode,
                    num_points=self.ortholoc_num_points,
                )

                ortho_ret["pose_c2w"] = add_ecef_origin_to_pose(ortho_ret["pose_c2w"], ecef_origin)
                ortho_ret["pose_w2c"] = np.linalg.inv(ortho_ret["pose_c2w"])
                ortho_ret["ecef_origin"] = ecef_origin
                pred_euler, pred_translation = ortholoc_pose_c2w_to_crop_pose(ortho_ret["pose_c2w"])
                reset_next = False
                gt_key = None
                status = "accepted"
                reason = None
                next_euler = pred_euler
                next_trans = pred_translation
                output_reset_count = reset_count

                gt_keys = [qname]
                stem = os.path.splitext(qname)[0]
                if "_" not in stem:
                    gt_keys.append(f"{stem}_0.png")
                gt_key = next((key for key in gt_keys if key in self.gt_pose_dict), None)
                if gt_key is None:
                    reason = "missing_gt"
                    logging.warning("无法判断阈值 reset：frame=%s 找不到 GT pose", qname)
                else:
                    from scipy.spatial.transform import Rotation

                    gt_pose = self.gt_pose_dict[gt_key]
                    gt_euler = np.asarray(gt_pose["euler"], dtype=np.float64)
                    gt_trans = np.asarray(gt_pose["trans"], dtype=np.float64)
                    gt_center = crop_wgs84_to_ecef(*gt_trans)
                    gt_rotation = crop_enu_to_ecef_rotation(gt_trans[0], gt_trans[1]) @ Rotation.from_euler(
                        "xyz", [gt_euler[0] - 180.0, gt_euler[1], gt_euler[2]], degrees=True
                    ).as_matrix()
                    jump_m = float(np.linalg.norm(ortho_ret["pose_c2w"][:3, 3] - gt_center))
                    cos = np.clip((np.trace(gt_rotation.T @ ortho_ret["pose_c2w"][:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                    jump_deg = float(np.rad2deg(np.arccos(cos)))
                    bad_t = self.gt_reset_translation_thresh_m > 0 and jump_m > self.gt_reset_translation_thresh_m
                    bad_r = self.gt_reset_rotation_thresh_deg > 0 and jump_deg > self.gt_reset_rotation_thresh_deg
                    if (bad_t or bad_r) and frame_idx < len(self.img_list) - 1:
                        reset_next = True
                        gt_reset_requests += 1
                        next_euler = gt_euler
                        next_trans = gt_trans
                        output_reset_count = reset_count + 1
                        status = "accepted_gt_reset_next"
                        reason = (
                            f"gt_error={jump_m:.2f}m,{jump_deg:.2f}deg; "
                            f"thresh={self.gt_reset_translation_thresh_m:.2f}m,{self.gt_reset_rotation_thresh_deg:.2f}deg; "
                            f"gt_key={gt_key}"
                        )
                        logging.warning("GT reset next crop after frame=%s reset_count=%d reason=%s", qname, output_reset_count, reason)

                if frame_idx < len(self.img_list) - 1:
                    enqueue_pose(
                        frame_idx + 1,
                        next_euler,
                        next_trans,
                        reset_count=output_reset_count,
                        pose_source="gt_reset" if reset_next else "ortholoc",
                    )
                write_pose_debug(
                    frame_idx=frame_idx,
                    image_name=qname,
                    status=status,
                    reason=reason,
                    crop_size=crop_size,
                    reset_count=output_reset_count,
                    pose_source=pose_source,
                    prior_euler=render_euler,
                    prior_trans=render_trans,
                    next_euler=next_euler,
                    next_trans=next_trans,
                    ortho_ret=ortho_ret,
                    ortho_euler=pred_euler,
                    ortho_trans=pred_translation,
                    jump_m=jump_m,
                    jump_deg=jump_deg,
                )

                print(
                    f"OrthoLoC localized frame {frame_idx} | "
                    f"Pitch, Roll, Yaw: {pred_euler.tolist()} | "
                    f"Longitude, Latitude, Altitude: {pred_translation.tolist()}"
                )

                record_pose_result(qname, pred_euler, pred_translation)
                append_reset_record(frame_idx, qname, output_reset_count, pose_source, status, reason)
                aggregated_poses.append(
                    {
                        "sample_id": crop_name,
                        "image_path": os.path.abspath(img_path),
                        "pose_w2c": ortho_ret["pose_w2c"].tolist(),
                        "intrinsics": ortho_ret["intrinsics"].tolist() if ortho_ret["intrinsics"] is not None else None,
                        "best_angle": ortho_ret["best_angle"],
                        "num_matches_raw": ortho_ret["num_matches_raw"],
                        "num_matches_filtered": ortho_ret["num_matches_filtered"],
                        "num_inliers": ortho_ret["num_inliers"],
                        "median_reprojection_error": ortho_ret["median_reprojection_error"],
                        "translation_wgs84": pred_translation.tolist(),
                        "euler_pitch_roll_yaw": pred_euler.tolist(),
                        "gt_error_m": jump_m,
                        "gt_error_deg": jump_deg,
                        "gt_key": gt_key,
                        "reset_next_from_gt": reset_next,
                        "reset_count": output_reset_count,
                        "pose_source": pose_source,
                    }
                )
            except Exception as exc:
                logging.exception(f"OrthoLoC 定位失败 frame={qname}: {exc}")
                ortho_debug_ret = locals().get("ortho_ret", None)
                ortho_debug_euler = locals().get("pred_euler", None)
                ortho_debug_trans = locals().get("pred_translation", None)
                ortho_debug_jump_m = locals().get("jump_m", None)
                ortho_debug_jump_deg = locals().get("jump_deg", None)

                failed_images.append(qname)
                write_pose_debug(
                    frame_idx=frame_idx,
                    image_name=qname,
                    status="failed",
                    reason=str(exc),
                    crop_size=crop_size,
                    reset_count=reset_count,
                    pose_source=pose_source,
                    prior_euler=render_euler,
                    prior_trans=render_trans,
                    next_euler=render_euler,
                    next_trans=render_trans,
                    ortho_ret=ortho_debug_ret,
                    ortho_euler=ortho_debug_euler,
                    ortho_trans=ortho_debug_trans,
                    jump_m=ortho_debug_jump_m,
                    jump_deg=ortho_debug_jump_deg,
                )
                aggregated_poses.append(
                    {
                        "sample_id": crop_name,
                        "image_path": os.path.abspath(img_path),
                        "pose_w2c": None,
                        "intrinsics": intrinsics_matrix.tolist(),
                        "error": str(exc),
                        "reset_count": reset_count,
                        "pose_source": pose_source,
                        "reset_next_from_gt": False,
                    }
                )
                self.stop_evt.set()
                break

            if self.stop_evt.is_set():
                break
            if frame_idx >= len(self.img_list) - 1:
                break
            next_idx = max(next_idx, frame_idx + 1)
        with open(self.reset_count_txt, "w", encoding="utf-8") as f:
            f.write("frame_idx\timage\treset_count\tpose_source\tstatus\treason\n")
            for record in reset_records:
                f.write(
                    "{frame_idx}\t{image}\t{reset_count}\t{pose_source}\t{status}\t{reason}\n".format(
                        frame_idx=record["frame_idx"],
                        image=record["image"],
                        reset_count=record["reset_count"],
                        pose_source=record["pose_source"],
                        status=record["status"],
                        reason=record["reason"].replace("\t", " "),
                    )
                )
        with open(self.ortholoc_pose_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dom_path": os.path.abspath(self.dom_path),
                    "dsm_path": os.path.abspath(self.dsm_path),
                    "intrinsics_path": os.path.abspath(self.ortholoc_intrinsics),
                    "matcher": self.ortholoc_matcher,
                    "device": self.ortholoc_device,
                    "angles": self.ortholoc_angles,
                    "poses": aggregated_poses,
                    "failed_images": failed_images,
                    "gt_reset_enabled": self.gt_reset_translation_thresh_m > 0 or self.gt_reset_rotation_thresh_deg > 0,
                    "gt_reset_requests": gt_reset_requests,
                    "max_gt_recrops": self.max_gt_recrops,
                    "gt_reset_translation_thresh_m": self.gt_reset_translation_thresh_m,
                    "gt_reset_rotation_thresh_deg": self.gt_reset_rotation_thresh_deg,
                },
                f,
                indent=2,
            )
        
        # self.flush_pose_and_send_sentinel()
        # 收尾：给渲染端塞哨兵 + 关队列 + 事件
        self.pose_q.put(None)
        self.stop_evt.set()
        self.task_q.close(); self.task_q.join_thread()
        self.pose_q.close(); self.pose_q.join_thread()
        logging.info('Localization process done')
    def flush_pose_and_send_sentinel(self):
        # 1) 清空 pose_q 中所有旧条目
        try:
            while True:
                self.pose_q.get_nowait()
        except queue.Empty:
            pass

        # 2) 然后往里放一个 None，作为哨兵
        self.pose_q.put(None)
    def start_threads(self):
        # 启动定位线程 
        self.localization_thread_instance = threading.Thread(target=self.localization_thread)
        self.localization_thread_instance.start()

        # 启动渲染线程
        self.rendering_thread_instance = threading.Thread(target=self.rendering_thread)
        self.rendering_thread_instance.start()

    def stop_threads(self):
        self.localization_thread_instance.join()
        self.rendering_thread_instance.join()

    def back_project(self, points_3d_input, p2d_r_input, euler_angles, translation, query_euler_angles, query_translation, num_samples = None, device = 'cuda'):
        """
        修改后的 back_project:
        1. 不再进行随机采样 (不再生成 xs, ys)。
        2. 直接接收之前选好的 3D 点 (points_3d_input)。
        3. 调用 get_3D_samples_v3 进行坐标转换(EPSG->ECEF)、去中心化和位姿生成。
        """

        # 1. 确保输入点在 GPU 上
        if not torch.is_tensor(points_3d_input):
            # 假设输入是 [N, 3]
            points_3d = torch.as_tensor(points_3d_input, device=device, dtype=torch.float32)
        else:
            points_3d = points_3d_input.to(device)
        if p2d_r_input is not None:
            if not torch.is_tensor(p2d_r_input):
                 p2d_r_input = torch.as_tensor(p2d_r_input, device=device)
            else:
                 p2d_r_input = p2d_r_input.to(device)


        # 2. 转换渲染帧的位姿 (欧拉角 -> 旋转矩阵)
        T_render_in_ECEF_c2w = torch.as_tensor(
            euler_angles_to_matrix_ECEF(euler_angles, translation),
            device=device, dtype=torch.float32
        )

        # 3. 直接调用处理函数
        # 注意：mkpts_r 传 None，因为我们不需要 2D 像素坐标来反投影了
        # depth_mat 参数这里实际上传入的是 3D 点集
        Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd, p2d_r_filtered, render_T_ecef = get_3D_samples_v3(
            mkpts_r=p2d_r_input, 
            depth_mat=points_3d, 
            T_c2w=T_render_in_ECEF_c2w, 
            camera=self.render_camera, 
            euler_angles=euler_angles, 
            translation=translation, 
            query_euler_angles=query_euler_angles, 
            query_translation=query_translation, 
            origin=self.origin, 
            num_init_pose=self.num_init_pose,
            mul=self.mul 
        )

        return Points_3D_ECEF, T_render_in_ECEF_w2c_modified, T_initial_pose_candidates, dd, p2d_r_filtered, render_T_ecef
    # 假设的计算位姿函数
    def calculate_pose(self, render_frame, points_3d):
        # 根据渲染图和3D点计算位姿
        return "Pose"

    def video_save(self):
        video_generation.create_video_from_images(self.outputs, self.outputs+'/video.mp4') 
    def eval(self):
        evaluate(self.estimated_pose, self.gt_pose)
    def run(self):
        ctx = mp.get_context("spawn")        # 保持 spawn
        p_render = ctx.Process(target=self.rendering_worker, daemon=True)
        p_loc    = ctx.Process(target=self.localization_worker, daemon=True)

        p_render.start(); p_loc.start()
        p_loc.join()                         # 先等定位结束
        p_render.join(5)                    # 最多等 30 s
        if p_render.is_alive():              # 兜底：仍卡住就强退
            p_render.terminate()
            p_render.join()
def parse_args():
    parser = argparse.ArgumentParser(description="你的程序说明")

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/feicuiwan_m4t.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="0018_V",
        help="实验名称，用于日志标记"
    )

    parser.add_argument(
        "--init_euler",
        type=str,
        default="[0.0, 37.1, -42.4]",
        help="初始欧拉角（格式如：[roll, pitch, yaw]）"
    )

    parser.add_argument(
        "--init_trans",
        type=str,
        default="[112.999054, 28.290497, 157.509]",
        help="初始平移（格式如：[x, y, z]）"
    )

    parser.add_argument(
        "--dom_path",
        type=str,
        default="/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dom_wgs84.tif",
        help="用于 crop 的 DOM GeoTIFF"
    )

    parser.add_argument(
        "--dsm_path",
        type=str,
        default="/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dsm_wgs84.tif",
        help="用于 crop 的 DSM GeoTIFF"
    )

    parser.add_argument(
        "--crop_k_json",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop", "K_uavscene.json"),
        help="crop 阶段使用的相机内参 json"
    )

    parser.add_argument(
        "--ortholoc_intrinsics",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "OrthoLoC", "OrthoLoC", "dji_intrinsics.json"),
        help="OrthoLoC 定位使用的查询图内参 json"
    )

    parser.add_argument(
        "--ortholoc_matcher",
        type=str,
        default="Mast3R",
        help="OrthoLoC matcher 名称"
    )

    parser.add_argument(
        "--ortholoc_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="OrthoLoC 运行设备"
    )

    parser.add_argument(
        "--ortholoc_angles",
        nargs="+",
        type=int,
        default=[0],
        help="OrthoLoC 匹配时的旋转候选角"
    )

    parser.add_argument(
        "--ortholoc_min_conf",
        type=float,
        default=0.5,
        help="OrthoLoC 最低匹配置信度"
    )

    parser.add_argument(
        "--ortholoc_reprojection_error",
        type=float,
        default=5.0,
        help="OrthoLoC PnP 重投影阈值"
    )

    parser.add_argument(
        "--ortholoc_pnp_mode",
        type=str,
        default="poselib",
        choices=["cv2", "poselib", "pycolmap"],
        help="OrthoLoC PnP 后端"
    )

    parser.add_argument(
        "--ortholoc_num_points",
        type=int,
        default=None,
        help="OrthoLoC PnP 最多使用多少 2D-3D 对应点，默认不截断"
    )

    parser.add_argument(
        "--min_ortholoc_crop_size",
        type=int,
        default=32,
        help="crop 结果宽或高小于该值时跳过 OrthoLoC，避免 MASt3R 处理 1x1 退化图"
    )

    parser.add_argument(
        "--max_pose_jump_m",
        type=float,
        default=100.0,
        help="保留兼容参数；当前不再用相对先验跳变拒绝 OrthoLoC"
    )

    parser.add_argument(
        "--max_euler_jump_deg",
        type=float,
        default=100.0,
        help="保留兼容参数；当前不再用相对先验跳变拒绝 OrthoLoC"
    )

    parser.add_argument(
        "--max_gt_recrops",
        type=int,
        default=1,
        help="保留兼容参数；reset 仅改变下一帧先验，不重裁当前帧"
    )

    parser.add_argument(
        "--gt_reset_translation_thresh_m",
        type=float,
        default=50.0,
        help="当前帧估计相对 GT 的三维平移误差超过该值(米)时，下一帧 crop 使用当前帧 GT；非正值关闭该阈值"
    )

    parser.add_argument(
        "--gt_reset_rotation_thresh_deg",
        type=float,
        default=50.0,
        help="当前帧估计相对 GT 的旋转误差超过该值(度)时，下一帧 crop 使用当前帧 GT；非正值关闭该阈值"
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="保留兼容参数；当前单帧失败即停止，不写替代结果"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="流程结束后运行原有 evaluate；默认不执行"
    )

    args = parser.parse_args()


    return args
# 主程序入口
if __name__ == "__main__":
    args = parse_args()
    init_euler = args.init_euler
    init_trans = args.init_trans
    name = args.name
    config_file = args.config
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_config = config["render_config"]
    default_confs = config["default_confs"]
    default_paths = config["default_paths"]
    dual_task = DualProcessTask(config,  name=name, args=args)
    dual_task.run()
    
    if args.evaluate:
        dual_task.eval()
    

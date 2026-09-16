import os
import numpy as np
from osgeo import gdal
from pyproj import Transformer, enums


# =========================================================
# 1. 改这里：ENU 原点
# =========================================================
# AMvalley
lat0 = 39.946784145786722
lon0 = 44.898231974206659
h0 = 0.0

# # Hkairport
# lat0 = 22.415056378573908
# lon0 = 114.04387569569714
# h0 = 0.0

# # HKisland_GNSS
# lat0 = 22.207854978330825
# lon0 = 114.26043089616468
# h0 = 0.0


# =========================================================
# 2. 改这里：输入文件
# =========================================================
# 局部坐标系 DSM（单波段高程）
local_dsm_tif = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/DSM.tif"

# 你的 WGS84 DSM（如果有，就填；没有可设为 None）
wgs84_dsm_tif = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dsm_wgs84.tif"   # 没有就写 None

# 你的 ECEF tif
# 支持两种情况：
#   - 3 波段：band1=X, band2=Y, band3=Z
#   - 如果不是 3 波段，脚本只做 round-trip 检查，不做 XYZ 对比
ecef_tif = None # 没有就写 None


# =========================================================
# 3. 参数
# =========================================================
sample_step = 50         # 每隔多少像素采样一次
use_all_valid = False    # True=所有有效点都检查；False=按 sample_step 抽样
height_tol = 1e-6        # 高度是否相同的阈值（米）
ll_tol_deg = 1e-10       # 经纬度 round-trip 阈值（度）
ecef_tol = 1e-4          # ECEF XYZ 比较阈值（米）


# =========================================================
# 4. 坐标转换器
# =========================================================
pipeline = f"""
+proj=pipeline
+step +proj=cart +ellps=WGS84
+step +proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0} +h_0={h0}
""".strip()

enu_transformer = Transformer.from_pipeline(pipeline)

# 经纬度 -> ECEF
geo_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

# ECEF -> 经纬度
ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)


# =========================================================
# 5. 基础函数
# =========================================================
def pixel_to_xy(gt, col, row):
    """
    栅格中心点坐标
    """
    x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
    y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
    return x, y


def world_to_pixel(gt, x, y):
    """
    只适用于 north-up 且无旋转的 geotransform
    """
    col = (x - gt[0]) / gt[1]
    row = (y - gt[3]) / gt[5]
    return col, row


def enu_to_wgs84(x, y, z):
    """
    ENU -> WGS84
    """
    lon, lat, h = enu_transformer.transform(
        x, y, z,
        direction=enums.TransformDirection.INVERSE
    )
    return lon, lat, h


def wgs84_to_enu(lon, lat, h):
    """
    WGS84 -> ENU
    """
    x, y, z = enu_transformer.transform(
        lon, lat, h,
        direction=enums.TransformDirection.FORWARD
    )
    return x, y, z


def make_valid_mask(arr, nodata):
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= (arr != nodata)
    return mask


def sample_indices(mask, step=50, use_all=False):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if use_all:
        return rows, cols

    keep = np.arange(0, len(rows), step)
    return rows[keep], cols[keep]


# =========================================================
# 6. 主检查逻辑
# =========================================================
def check_conversion(local_dsm_tif, wgs84_dsm_tif=None, ecef_tif=None):
    gdal.UseExceptions()

    # ---------- 读 local DSM ----------
    local_ds = gdal.Open(local_dsm_tif, gdal.GA_ReadOnly)
    if local_ds is None:
        raise FileNotFoundError(local_dsm_tif)

    if local_ds.RasterCount < 1:
        raise RuntimeError("local DSM 至少要有 1 个波段")

    local_band = local_ds.GetRasterBand(1)
    local_arr = local_band.ReadAsArray().astype(np.float64)
    local_nodata = local_band.GetNoDataValue()
    local_gt = local_ds.GetGeoTransform()

    local_mask = make_valid_mask(local_arr, local_nodata)
    rows, cols = sample_indices(local_mask, step=sample_step, use_all=use_all_valid)

    print("=" * 80)
    print("Local DSM:")
    print("  path:", local_dsm_tif)
    print("  size:", local_ds.RasterXSize, local_ds.RasterYSize)
    print("  nodata:", local_nodata)
    print("  valid pixels:", int(local_mask.sum()))
    print("  sampled pixels:", len(rows))

    if len(rows) == 0:
        raise RuntimeError("没有有效像素可检查")

    # ---------- 读 WGS84 DSM（可选） ----------
    wgs84_ds = None
    wgs84_arr = None
    wgs84_nodata = None
    wgs84_gt = None
    if wgs84_dsm_tif is not None:
        wgs84_ds = gdal.Open(wgs84_dsm_tif, gdal.GA_ReadOnly)
        if wgs84_ds is None:
            raise FileNotFoundError(wgs84_dsm_tif)
        wgs84_band = wgs84_ds.GetRasterBand(1)
        wgs84_arr = wgs84_band.ReadAsArray().astype(np.float64)
        wgs84_nodata = wgs84_band.GetNoDataValue()
        wgs84_gt = wgs84_ds.GetGeoTransform()

        print("-" * 80)
        print("WGS84 DSM:")
        print("  path:", wgs84_dsm_tif)
        print("  size:", wgs84_ds.RasterXSize, wgs84_ds.RasterYSize)
        print("  nodata:", wgs84_nodata)

    # ---------- 读 ECEF tif（可选） ----------
    ecef_ds = None
    ecef_xyz = None
    if ecef_tif is not None:
        ecef_ds = gdal.Open(ecef_tif, gdal.GA_ReadOnly)
        if ecef_ds is None:
            raise FileNotFoundError(ecef_tif)

        print("-" * 80)
        print("ECEF tif:")
        print("  path:", ecef_tif)
        print("  size:", ecef_ds.RasterXSize, ecef_ds.RasterYSize)
        print("  bands:", ecef_ds.RasterCount)

        if ecef_ds.RasterCount >= 3:
            ex = ecef_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
            ey = ecef_ds.GetRasterBand(2).ReadAsArray().astype(np.float64)
            ez = ecef_ds.GetRasterBand(3).ReadAsArray().astype(np.float64)

            nod1 = ecef_ds.GetRasterBand(1).GetNoDataValue()
            nod2 = ecef_ds.GetRasterBand(2).GetNoDataValue()
            nod3 = ecef_ds.GetRasterBand(3).GetNoDataValue()

            ecef_mask = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez)
            if nod1 is not None:
                ecef_mask &= (ex != nod1)
            if nod2 is not None:
                ecef_mask &= (ey != nod2)
            if nod3 is not None:
                ecef_mask &= (ez != nod3)

            ecef_xyz = (ex, ey, ez, ecef_mask)
        else:
            print("  [WARN] ECEF tif 不是 3 波段，跳过 XYZ 一致性检查")

    # =====================================================
    # A. 检查 ENU -> WGS84 -> ENU round-trip
    #    看经纬度转换逻辑本身有没有问题
    # =====================================================
    lon_errs = []
    lat_errs = []
    h_roundtrip_errs = []
    x_roundtrip_errs = []
    y_roundtrip_errs = []
    z_roundtrip_errs = []

    # =====================================================
    # B. 检查高度 h 是否与 local DSM 一致
    #    对于你的变换，理论上 h 应该就是原始 DSM 高程
    # =====================================================
    h_minus_local = []

    # =====================================================
    # C. 如果有 WGS84 DSM，检查投影后高程值是否没变
    # =====================================================
    wgs84_value_diffs = []
    wgs84_hit_count = 0

    # =====================================================
    # D. 如果有 ECEF 3-band tif，检查 XYZ 是否一致
    # =====================================================
    ecef_dx = []
    ecef_dy = []
    ecef_dz = []
    ecef_dist = []
    ecef_hit_count = 0

    for row, col in zip(rows, cols):
        z_local = local_arr[row, col]
        x_enu, y_enu = pixel_to_xy(local_gt, col, row)

        # local ENU + height -> WGS84
        lon, lat, h = enu_to_wgs84(x_enu, y_enu, z_local)

        # WGS84 -> ENU 回去
        x2, y2, z2 = wgs84_to_enu(lon, lat, h)

        # 误差统计
        x_roundtrip_errs.append(abs(x2 - x_enu))
        y_roundtrip_errs.append(abs(y2 - y_enu))
        z_roundtrip_errs.append(abs(z2 - z_local))

        # h 是否与 local DSM 一致
        h_minus_local.append(abs(h - z_local))

        # 再做一次 WGS84 -> ECEF -> WGS84
        X, Y, Z = geo_to_ecef.transform(lon, lat, h)
        lon2, lat2, h2 = ecef_to_geo.transform(X, Y, Z)

        lon_errs.append(abs(lon2 - lon))
        lat_errs.append(abs(lat2 - lat))
        h_roundtrip_errs.append(abs(h2 - h))

        # ---------- 和 WGS84 DSM 对比高程 ----------
        if wgs84_ds is not None:
            c2f, r2f = world_to_pixel(wgs84_gt, lon, lat)
            c2 = int(round(c2f))
            r2 = int(round(r2f))

            if 0 <= r2 < wgs84_arr.shape[0] and 0 <= c2 < wgs84_arr.shape[1]:
                val = wgs84_arr[r2, c2]
                valid = np.isfinite(val)
                if wgs84_nodata is not None:
                    valid = valid and (val != wgs84_nodata)

                if valid:
                    wgs84_value_diffs.append(abs(val - z_local))
                    wgs84_hit_count += 1

        # ---------- 和 ECEF tif 的 XYZ 对比 ----------
        if ecef_xyz is not None:
            ex, ey, ez, emask = ecef_xyz

            # 默认先用相同行列对应
            if row < ex.shape[0] and col < ex.shape[1] and emask[row, col]:
                dx = ex[row, col] - X
                dy = ey[row, col] - Y
                dz = ez[row, col] - Z
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)

                ecef_dx.append(dx)
                ecef_dy.append(dy)
                ecef_dz.append(dz)
                ecef_dist.append(dist)
                ecef_hit_count += 1

    # =====================================================
    # 输出统计
    # =====================================================
    def print_stats(name, vals):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0:
            print(f"{name}: no data")
            return
        print(f"{name}:")
        print(f"  mean   = {vals.mean():.12f}")
        print(f"  max    = {vals.max():.12f}")
        print(f"  min    = {vals.min():.12f}")
        print(f"  median = {np.median(vals):.12f}")

    print("\n" + "=" * 80)
    print("A. ENU <-> WGS84 <-> ECEF round-trip 检查")
    print_stats("lon round-trip error (deg)", lon_errs)
    print_stats("lat round-trip error (deg)", lat_errs)
    print_stats("h round-trip error (m)", h_roundtrip_errs)
    print_stats("ENU x round-trip error (m)", x_roundtrip_errs)
    print_stats("ENU y round-trip error (m)", y_roundtrip_errs)
    print_stats("ENU z round-trip error (m)", z_roundtrip_errs)

    print("\n" + "=" * 80)
    print("B. 高度 h 是否等于原 local DSM 高程")
    print_stats("|h - z_local| (m)", h_minus_local)

    all_height_same = np.all(np.asarray(h_minus_local) <= height_tol)
    print(f"Height unchanged within tol={height_tol}: {all_height_same}")

    ll_ok = (
        np.max(lon_errs) <= ll_tol_deg and
        np.max(lat_errs) <= ll_tol_deg and
        np.max(h_roundtrip_errs) <= height_tol
    )
    print(f"Lon/Lat/H round-trip within tol: {ll_ok}")

    if wgs84_ds is not None:
        print("\n" + "=" * 80)
        print("C. local DSM 与 WGS84 DSM 的高程值对比")
        print("matched samples:", wgs84_hit_count)
        print_stats("|value_wgs84 - value_local| (m)", wgs84_value_diffs)
        if len(wgs84_value_diffs) > 0:
            same_wgs84_height = np.all(np.asarray(wgs84_value_diffs) <= height_tol)
            print(f"WGS84 DSM height unchanged within tol={height_tol}: {same_wgs84_height}")

    if ecef_xyz is not None:
        print("\n" + "=" * 80)
        print("D. 计算得到的 ECEF 与 ECEF tif 中 XYZ 对比")
        print("matched samples:", ecef_hit_count)
        print_stats("dx = X_tif - X_calc (m)", ecef_dx)
        print_stats("dy = Y_tif - Y_calc (m)", ecef_dy)
        print_stats("dz = Z_tif - Z_calc (m)", ecef_dz)
        print_stats("3D distance error (m)", ecef_dist)
        if len(ecef_dist) > 0:
            ecef_ok = np.all(np.asarray(ecef_dist) <= ecef_tol)
            print(f"ECEF XYZ matched within tol={ecef_tol} m: {ecef_ok}")

    print("\n" + "=" * 80)
    print("结论建议：")
    print("1. 如果 B 里 |h - z_local| 全接近 0，说明高度没变。")
    print("2. 如果 C 里 |value_wgs84 - value_local| 全接近 0，说明重投影后 DSM 值没变。")
    print("3. 如果 A 里 lon/lat/h round-trip 误差极小，说明经纬度转换链路本身是对的。")
    print("4. 如果 D 里 ECEF XYZ 误差极小，说明你生成的 ECEF tif 也对。")


if __name__ == "__main__":
    check_conversion(
        local_dsm_tif=local_dsm_tif,
        wgs84_dsm_tif=wgs84_dsm_tif,
        ecef_tif=ecef_tif
    )
import os
import numpy as np
from pyproj import Transformer, enums
from osgeo import gdal, osr


# =========================================================
# 1. ENU 原点（改成你的）
# =========================================================
# Hkairport
# lat0 = 22.415056378573908
# lon0 = 114.04387569569714
# h0 = 0.0

# #HKisland_GNSS
# lat0 = 22.207854978330825
# lon0 = 114.26043089616468
# h0 = 0.0

#AMvalley
lat0 = 39.946784145786722
lon0 = 44.898231974206659
h0 = 0.0



pipeline = f"""
+proj=pipeline
+step +proj=cart +ellps=WGS84
+step +proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0} +h_0={h0}
""".strip()

enu_transformer = Transformer.from_pipeline(pipeline)


def enu_to_wgs84(x, y, z=0.0):
    lon, lat, h = enu_transformer.transform(
        x, y, z,
        direction=enums.TransformDirection.INVERSE
    )
    return lon, lat, h


def pixel_to_enu(gt, col, row):
    x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
    y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
    return x, y


def compute_lonlat_bounds(src_ds, sample_edge_points=200):
    """
    通过采样四条边，估计原 ENU 图投到 WGS84 后的经纬度包围盒
    """
    gt = src_ds.GetGeoTransform()
    w = src_ds.RasterXSize
    h = src_ds.RasterYSize

    lons = []
    lats = []

    # 上下边
    cols = np.linspace(0, w - 1, sample_edge_points).astype(int)
    for col in cols:
        for row in [0, h - 1]:
            x, y = pixel_to_enu(gt, col, row)
            lon, lat, _ = enu_to_wgs84(x, y, 0.0)
            lons.append(lon)
            lats.append(lat)

    # 左右边
    rows = np.linspace(0, h - 1, sample_edge_points).astype(int)
    for row in rows:
        for col in [0, w - 1]:
            x, y = pixel_to_enu(gt, col, row)
            lon, lat, _ = enu_to_wgs84(x, y, 0.0)
            lons.append(lon)
            lats.append(lat)

    return min(lons), max(lons), min(lats), max(lats)


def estimate_output_resolution(src_ds):
    """
    估计输出 WGS84 栅格的经纬度分辨率（度/像素）
    用源图中心附近的一个像素步长来估算
    """
    gt = src_ds.GetGeoTransform()
    w = src_ds.RasterXSize
    h = src_ds.RasterYSize

    c = w // 2
    r = h // 2

    x0, y0 = pixel_to_enu(gt, c, r)
    x1, y1 = pixel_to_enu(gt, c + 1, r)
    x2, y2 = pixel_to_enu(gt, c, r + 1)

    lon0, lat0_, _ = enu_to_wgs84(x0, y0, 0.0)
    lon1, lat1_, _ = enu_to_wgs84(x1, y1, 0.0)
    lon2, lat2_, _ = enu_to_wgs84(x2, y2, 0.0)

    res_lon = abs(lon1 - lon0)
    res_lat = abs(lat2_ - lat0_)

    if res_lon == 0 or res_lat == 0:
        raise RuntimeError("估计输出分辨率失败")

    return res_lon, res_lat


def build_output_grid(bounds, res_lon, res_lat):
    min_lon, max_lon, min_lat, max_lat = bounds

    out_w = int(np.ceil((max_lon - min_lon) / res_lon))
    out_h = int(np.ceil((max_lat - min_lat) / res_lat))

    # north-up geotransform
    out_gt = (
        min_lon,      # top-left x
        res_lon,      # pixel width
        0.0,
        max_lat,      # top-left y
        0.0,
        -res_lat      # pixel height
    )

    return out_w, out_h, out_gt


def lonlat_to_pixel(gt, lon, lat):
    col = (lon - gt[0]) / gt[1]
    row = (lat - gt[3]) / gt[5]
    return col, row


def convert_enu_to_wgs84_manual(src_path, dst_path, is_dsm=False):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(src_path)

    src_gt = src_ds.GetGeoTransform()
    w = src_ds.RasterXSize
    h = src_ds.RasterYSize
    bands = src_ds.RasterCount

    print(f"\n处理: {src_path}")
    print(f"size = {w} x {h}, bands = {bands}")

    # 1) 估计输出经纬度范围
    bounds = compute_lonlat_bounds(src_ds, sample_edge_points=300)
    min_lon, max_lon, min_lat, max_lat = bounds
    print("lon/lat bounds:")
    print(f"  min_lon={min_lon}, max_lon={max_lon}")
    print(f"  min_lat={min_lat}, max_lat={max_lat}")

    # 2) 估计输出分辨率
    res_lon, res_lat = estimate_output_resolution(src_ds)
    print(f"estimated resolution: res_lon={res_lon}, res_lat={res_lat}")

    # 3) 建输出网格
    out_w, out_h, out_gt = build_output_grid(bounds, res_lon, res_lat)
    print(f"output size = {out_w} x {out_h}")
    print(f"output geotransform = {out_gt}")

    driver = gdal.GetDriverByName("GTiff")
    dtype = src_ds.GetRasterBand(1).DataType

    dst_ds = driver.Create(
        dst_path,
        out_w,
        out_h,
        bands,
        dtype,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    )
    if dst_ds is None:
        raise RuntimeError(f"无法创建输出: {dst_path}")

    dst_ds.SetGeoTransform(out_gt)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())

    # 读取源数据
    src_arrays = []
    for b in range(1, bands + 1):
        src_arrays.append(src_ds.GetRasterBand(b).ReadAsArray())

    # 源 nodata
    src_nodatas = []
    for b in range(1, bands + 1):
        src_nodatas.append(src_ds.GetRasterBand(b).GetNoDataValue())

    # 输出初始化
    out_arrays = []
    for b in range(bands):
        arr = np.zeros((out_h, out_w), dtype=src_arrays[b].dtype)
        out_arrays.append(arr)

    # 4) 反向映射：对每个输出像素，算它对应源图哪个位置
    #    输出网格是 lon/lat，所以先 lonlat -> ENU 不方便
    #    这里改成 forward splat：遍历源图，把值投到输出图
    #
    #    对大图来说不是最优，但先保证正确。
    #
    print("开始逐像素投影...")

    for row in range(h):
        if row % 500 == 0:
            print(f"  row {row}/{h}")

        for col in range(w):
            x, y = pixel_to_enu(src_gt, col, row)
            lon, lat, _ = enu_to_wgs84(x, y, 0.0)

            out_col_f, out_row_f = lonlat_to_pixel(out_gt, lon, lat)
            out_col = int(round(out_col_f))
            out_row = int(round(out_row_f))

            if 0 <= out_col < out_w and 0 <= out_row < out_h:
                for b in range(bands):
                    out_arrays[b][out_row, out_col] = src_arrays[b][row, col]

    # 5) 写出
    for b in range(bands):
        band = dst_ds.GetRasterBand(b + 1)
        band.WriteArray(out_arrays[b])
        if src_nodatas[b] is not None:
            band.SetNoDataValue(src_nodatas[b])

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    print(f"完成: {dst_path}")


if __name__ == "__main__":
    gdal.UseExceptions()


    dom_in = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/DOM.tif"
    dsm_in = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/DSM.tif"

    dom_out = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dom_wgs84.tif"
    dsm_out = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/AMvalley/dsm_wgs84.tif"
    # DOM：影像建议 bilinear 或 cubic
    convert_enu_to_wgs84_manual(dom_in, dom_out, is_dsm=False)
    convert_enu_to_wgs84_manual(dsm_in, dsm_out, is_dsm=True)






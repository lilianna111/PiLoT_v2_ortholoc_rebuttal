# import random
# import numpy as np
# from pyproj import Transformer, enums
# from osgeo import gdal


# lat0 = 22.415056378573908
# lon0 = 114.04387569569714
# h0 = 0.0

# pipeline = f"""
# +proj=pipeline
# +step +proj=cart +ellps=WGS84
# +step +proj=topocentric +ellps=WGS84 +lat_0={lat0} +lon_0={lon0} +h_0={h0}
# """.strip()

# enu_transformer = Transformer.from_pipeline(pipeline)


# def enu_to_wgs84(x, y, z=0.0):
#     lon, lat, h = enu_transformer.transform(
#         x, y, z,
#         direction=enums.TransformDirection.INVERSE
#     )
#     return lon, lat, h


# def pixel_to_geo(gt, col, row):
#     x = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
#     y = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
#     return x, y


# def geo_to_pixel(gt, x, y):
#     A = np.array([[gt[1], gt[2]],
#                   [gt[4], gt[5]]], dtype=float)
#     b = np.array([x - gt[0], y - gt[3]], dtype=float)
#     col_row = np.linalg.solve(A, b)
#     col = col_row[0] - 0.5
#     row = col_row[1] - 0.5
#     return col, row


# def read_pixel(ds, col, row):
#     col_i = int(round(col))
#     row_i = int(round(row))
#     if col_i < 0 or row_i < 0 or col_i >= ds.RasterXSize or row_i >= ds.RasterYSize:
#         return None

#     vals = []
#     for b in range(1, ds.RasterCount + 1):
#         arr = ds.GetRasterBand(b).ReadAsArray(col_i, row_i, 1, 1)
#         vals.append(arr[0, 0].item())
#     return vals


# def is_valid_dom_pixel(vals):
#     if vals is None:
#         return False
#     if len(vals) >= 4:
#         # alpha>0 认为有效
#         return vals[3] > 0
#     # 没 alpha 时，至少不是全 0
#     return any(v != 0 for v in vals)


# def find_valid_source_pixel(ds, max_trials=10000, margin=100):
#     w = ds.RasterXSize
#     h = ds.RasterYSize

#     for _ in range(max_trials):
#         col = random.randint(margin, w - 1 - margin)
#         row = random.randint(margin, h - 1 - margin)
#         vals = read_pixel(ds, col, row)
#         if is_valid_dom_pixel(vals):
#             return col, row, vals

#     raise RuntimeError("没有找到有效源像素，请检查图像或增大 max_trials")


# def check_valid_point(src_enu_tif, dst_wgs84_tif):
#     src_ds = gdal.Open(src_enu_tif, gdal.GA_ReadOnly)
#     dst_ds = gdal.Open(dst_wgs84_tif, gdal.GA_ReadOnly)

#     if src_ds is None or dst_ds is None:
#         raise RuntimeError("图像打开失败")

#     src_gt = src_ds.GetGeoTransform()
#     dst_gt = dst_ds.GetGeoTransform()

#     src_col, src_row, src_vals = find_valid_source_pixel(src_ds)

#     enu_x, enu_y = pixel_to_geo(src_gt, src_col, src_row)
#     lon, lat, _ = enu_to_wgs84(enu_x, enu_y, 0.0)
#     dst_col, dst_row = geo_to_pixel(dst_gt, lon, lat)
#     dst_vals = read_pixel(dst_ds, dst_col, dst_row)

#     print("\n================ 有效点检查 ================")
#     print(f"源图有效像素: col={src_col}, row={src_row}")
#     print(f"源图像素值: {src_vals}")
#     print(f"ENU: x={enu_x:.6f}, y={enu_y:.6f}")
#     print(f"WGS84: lon={lon:.10f}, lat={lat:.10f}")
#     print(f"目标图对应像素(浮点): col={dst_col:.3f}, row={dst_row:.3f}")
#     print(f"目标图像素值: {dst_vals}")

#     if dst_vals is None:
#         print("结果：目标点落在图外，说明转换可能有问题。")
#     else:
#         print("结果：至少空间上落到了目标图范围内。")
#         if len(src_vals) == len(dst_vals):
#             print("逐波段差值:")
#             for i, (sv, dv) in enumerate(zip(src_vals, dst_vals), start=1):
#                 print(f"  band {i}: dst-src = {float(dv) - float(sv)}")
#     print("===========================================\n")



# # =========================================================
# # 4. main
# # =========================================================
# if __name__ == "__main__":
#     gdal.UseExceptions()

#     # ===== 改路径 =====
#     dom_in = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/DOM.tif"
#     dsm_in = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/DSM.tif"

#     dom_out = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dom_wgs84.tif"
#     dsm_out = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dsm_wgs84.tif"


#     check_valid_point(dom_in, dom_out)

from osgeo import gdal, osr

def inspect_raster(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"打不开: {path}")
        return

    print("\n================ Raster Info ================")
    print("path:", path)
    print("size:", ds.RasterXSize, ds.RasterYSize)
    print("bands:", ds.RasterCount)

    gt = ds.GetGeoTransform()
    print("GeoTransform:", gt)

    proj = ds.GetProjection()
    print("Projection WKT:")
    print(proj if proj else "[empty]")

    srs = osr.SpatialReference()
    if proj:
        srs.ImportFromWkt(proj)
        print("EPSG guess:", srs.GetAttrValue("AUTHORITY", 1))
        print("IsGeographic:", bool(srs.IsGeographic()))
        print("IsProjected:", bool(srs.IsProjected()))

    print("Corner coordinates:")
    w = ds.RasterXSize
    h = ds.RasterYSize

    def pix2geo(col, row):
        x = gt[0] + col * gt[1] + row * gt[2]
        y = gt[3] + col * gt[4] + row * gt[5]
        return x, y

    corners = {
        "UL": pix2geo(0, 0),
        "UR": pix2geo(w, 0),
        "LL": pix2geo(0, h),
        "LR": pix2geo(w, h),
    }
    for k, v in corners.items():
        print(f"{k}: {v}")

    print("=============================================\n")


if __name__ == "__main__":
    inspect_raster("/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dom_wgs84.tif")
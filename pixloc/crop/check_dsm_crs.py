"""
检查 DSM/DOM 的坐标系与范围，用于判断「裁剪范围无效」是地图问题还是代码问题。
用法: python check_dsm_crs.py
（需先修改下方 ref_DSM / ref_DOM 路径）
"""
import rasterio

ref_DSM = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/DSM_WGS84.tif"
ref_DOM = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOM/DOM_WGS84.tif"

def main():
    with rasterio.open(ref_DSM) as dsm:
        crs = dsm.crs
        t = dsm.transform
        h, w = dsm.height, dsm.width
        bounds = dsm.bounds  # left, bottom, right, top

    print("========== DSM 坐标系与范围 ==========")
    print("CRS (坐标系):", crs)
    if crs is None:
        print("  -> 若为 None，说明 TIF 未写入坐标系，当前代码按 WGS84 用会出错")
    elif str(crs) != "EPSG:4326":
        print("  -> 不是 WGS84！代码里用 (lon, lat) 直接乘 transform 会错，需先投影到该 CRS")
    else:
        print("  -> 是 WGS84，则 transform 应为 (lon, lat)，范围应在 [-180,180], [-90,90]")

    print("\nTransform (rasterio):", t)
    print("  含义: pixel (col, row) -> (x, y) = (c + col*a, f + row*e) 等")
    print("  栅格尺寸:", h, "x", w)
    print("  bounds (left, bottom, right, top):", bounds)
    x_min, y_min, x_max, y_max = bounds
    print("  即 x 范围: [{}, {}], y 范围: [{}, {}]".format(x_min, x_max, y_min, y_max))

    # 若为 4326，x=经度 y=纬度
    if crs and "4326" in str(crs):
        if -180 <= x_min <= 180 and -90 <= y_min <= 90:
            print("  -> 范围在正常经纬度内，数据与 WGS84 假设一致")
        else:
            print("  -> 范围不在 [-180,180]/[-90,90]，可能 TIF 标了 4326 但实际存的是投影坐标")
    else:
        print("  -> 若 x/y 在数万～数十万量级，多为投影坐标(米)，需用 pyproj 将 lon/lat 转到该 CRS 再索引")

    print("\n========== 你的视角中心 (来自 test_wgs84) ==========")
    print("  center_lon ≈ 114.0427, center_lat ≈ 22.416")
    print("  若 DSM 是 WGS84，该点应落在上述 bounds 内；若 DSM 是投影坐标，需先转换再比较")

if __name__ == "__main__":
    main()

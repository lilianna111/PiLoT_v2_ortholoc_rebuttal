import os
from PIL import Image
import piexif
import math

def deg_to_dms(deg):
    """将十进制度数转换为度分秒格式"""
    d = int(abs(deg))
    m = int((abs(deg) - d) * 60)
    s = int(((abs(deg) - d) * 60 - m) * 60 * 10000)
    return ((d, 1), (m, 1), (s, 10000))

def create_gps_dict(lat, lon, alt):
    """创建GPS EXIF字典"""
    return {
        piexif.GPSIFD.GPSLatitude: deg_to_dms(lat),
        piexif.GPSIFD.GPSLongitude: deg_to_dms(lon),
        piexif.GPSIFD.GPSAltitude: (int(alt * 100), 100),
        piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
        piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W',
        piexif.GPSIFD.GPSAltitudeRef: 0
    }

def calculate_camera_params(fovy_deg, image_width, image_height, alt):
    """
    根据OSG渲染fovy和图像尺寸、相机高度计算相机内参和GSD
    """
    fovy_rad = math.radians(fovy_deg)
    
    # 焦距（像素单位）
    f_y = image_height / (2 * math.tan(fovy_rad / 2))
    f_x = f_y  # 正方形图像
    
    # 主点
    cx = image_width / 2
    cy = image_height / 2
    
    # GSD计算
    gsd_y = (2 * alt * math.tan(fovy_rad / 2)) / image_height
    gsd_x = gsd_y  # 正方形
    
    camera_params = {
        'focal_length_pixels': f_y,
        'f_x': f_x,
        'f_y': f_y,
        'cx': cx,
        'cy': cy,
        'image_width': image_width,
        'image_height': image_height,
        'gsd_x': gsd_x,
        'gsd_y': gsd_y,
        'k1': 0.0,
        'k2': 0.0,
        'k3': 0.0,
        'p1': 0.0,
        'p2': 0.0
    }
    
    # print(f"计算相机参数:")
    # print(f"focal (pixels): fx={f_x:.2f}, fy={f_y:.2f}")
    # print(f"主点: cx={cx:.2f}, cy={cy:.2f}")
    # print(f"GSD: gx={gsd_x:.6f}m/px, gy={gsd_y:.6f}m/px")
    
    return camera_params

def process_images(pose_file, image_folder, output_folder, fovy_deg, image_width, image_height):
    """批量处理PNG图片，写入EXIF"""
    
    # 读取位姿文件
    pose_data = {}
    with open(pose_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                filename = parts[0]
                roll = float(parts[4])
                pitch = float(parts[5])
                yaw = float(parts[6])
                lon = float(parts[1])
                lat = float(parts[2])
                alt = float(parts[3]) if len(parts) > 6 else 0
                pose_data[filename] = {'roll': roll, 'pitch': pitch, 'yaw': yaw,
                                       'lon': lon, 'lat': lat, 'alt': alt}
    print(f"读取到 {len(pose_data)} 条位姿数据")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_count = 0
    for filename, pose in pose_data.items():
        base_name = os.path.splitext(filename)[0]
        input_path = os.path.join(image_folder, base_name + ".png")
        if not os.path.exists(input_path):
            input_path = os.path.join(image_folder, base_name + "-RGB.png")
            if not os.path.exists(input_path):
                print(f"未找到图片: {filename}")
                continue
        
        try:
            img = Image.open(input_path)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # 计算相机内参和GSD
            camera_params = calculate_camera_params(fovy_deg, image_width, image_height, pose['alt'])
            
            # 创建EXIF
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            # GPS
            exif_dict['GPS'] = create_gps_dict(pose['lat'], pose['lon'], pose['alt'])
            
            # 姿态信息
            pose_info = f"Roll:{pose['roll']:.6f},Pitch:{pose['pitch']:.6f},Yaw:{pose['yaw']:.6f}"
            exif_dict['0th'][piexif.ImageIFD.ImageDescription] = pose_info.encode('utf-8')
            
            # 相机内参和GSD写入UserComment
            camera_info = f"Camera_Intrinsics:cx:{camera_params['cx']:.2f},cy:{camera_params['cy']:.2f}," \
                          f"k1:{camera_params['k1']:.8f},k2:{camera_params['k2']:.8f},k3:{camera_params['k3']:.8f}," \
                          f"p1:{camera_params['p1']:.8f},p2:{camera_params['p2']:.8f}," \
                          f"GSD_x:{camera_params['gsd_x']:.6f}m/px,GSD_y:{camera_params['gsd_y']:.6f}m/px," \
                          f"fx:{camera_params['f_x']:.2f},fy:{camera_params['f_y']:.2f}"
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = camera_info.encode('utf-8')
            
            # 相机制造商信息
            exif_dict['0th'][piexif.ImageIFD.Make] = "OSG Camera".encode('utf-8')
            exif_dict['0th'][piexif.ImageIFD.Model] = "Simulation".encode('utf-8')
            
            exif_bytes = piexif.dump(exif_dict)
            output_path = os.path.join(output_folder, base_name + ".jpg")
            img.save(output_path, "JPEG", exif=exif_bytes, quality=95)
            
            # print(f"✓ 处理完成: {output_path}")
            processed_count += 1
        except Exception as e:
            print(f"✗ 图片处理失败 {filename}: {e}")
    
    print(f"\n批量处理完成！成功处理 {processed_count} 张图片")

def verify_exif(jpg_path):
    """验证JPG文件的EXIF信息"""
    try:
        exif_dict = piexif.load(jpg_path)
        print(f"\n验证文件: {jpg_path}")
        
        # GPS
        if 'GPS' in exif_dict and exif_dict['GPS']:
            gps = exif_dict['GPS']
            if piexif.GPSIFD.GPSLatitude in gps:
                lat_dms = gps[piexif.GPSIFD.GPSLatitude]
                lat = lat_dms[0][0]/lat_dms[0][1] + lat_dms[1][0]/(lat_dms[1][1]*60) + lat_dms[2][0]/(lat_dms[2][1]*3600)
                print(f"GPS纬度: {lat:.8f}")
            if piexif.GPSIFD.GPSLongitude in gps:
                lon_dms = gps[piexif.GPSIFD.GPSLongitude]
                lon = lon_dms[0][0]/lon_dms[0][1] + lon_dms[1][0]/(lon_dms[1][1]*60) + lon_dms[2][0]/(lon_dms[2][1]*3600)
                print(f"GPS经度: {lon:.8f}")
            if piexif.GPSIFD.GPSAltitude in gps:
                alt = gps[piexif.GPSIFD.GPSAltitude][0] / gps[piexif.GPSIFD.GPSAltitude][1]
                print(f"GPS高度: {alt}m")
        
        # 姿态信息
        if '0th' in exif_dict and piexif.ImageIFD.ImageDescription in exif_dict['0th']:
            pose_info = exif_dict['0th'][piexif.ImageIFD.ImageDescription].decode('utf-8')
            print(f"姿态信息: {pose_info}")
        
        # 相机内参
        if 'Exif' in exif_dict and piexif.ExifIFD.UserComment in exif_dict['Exif']:
            camera_info = exif_dict['Exif'][piexif.ExifIFD.UserComment].decode('utf-8')
            print(f"相机内参: {camera_info}")
        print("Exif信息验证通过")
    except Exception as e:
        print(f"验证失败: {e}")


# # ==============================
# if __name__ == "__main__":
#     # 配置
#     pose_file = "/media/amax/PS2000/google/zigzag_9_back_left.txt"
#     image_folder = "/media/amax/PS2000/render/zigzag_9_back_left"
#     output_folder = "/media/amax/PS2000/pose"
#     IMAGE_WIDTH = 1920
#     IMAGE_HEIGHT = 1080
#     FOVY = 26.0232  # 从OSG渲染读取
    
#     # 批量处理
#     process_images(pose_file, image_folder, output_folder, FOVY, IMAGE_WIDTH, IMAGE_HEIGHT)
    
#     # 验证第一张
#     first_jpg = os.path.join(output_folder, "0.png")
#     if os.path.exists(first_jpg):
#         verify_exif(first_jpg)
#     else:
#         print("文件不存在")


# ... [保留你之前的 deg_to_dms, create_gps_dict, calculate_camera_params 函数] ...
# 注意：建议在 calculate_camera_params 中将 cx, cy 改为传入参数，或者直接用 RENDER_PARAMS 的值

def batch_process_all_folders(base_dir, txt_dir, output_base):
    """
    base_dir: 存放图片文件夹的根目录 (例如 /media/amax/PS2000/render)
    txt_dir: 存放所有 .txt 位姿文件的目录 (例如 /media/amax/PS2000/google)
    output_base: 结果输出的根目录 (例如 /media/amax/PS2000/render_pose)
    """
    
    # 相机固定参数 (来自你的 render_camera)
    RENDER_PARAMS = [1920, 1080, 957.85, 537.55, 1920, 1080, 1350.0, 1350.0]
    WIDTH, HEIGHT = RENDER_PARAMS[0], RENDER_PARAMS[1]
    FY = RENDER_PARAMS[7]
    FOVY = math.degrees(2 * math.atan(HEIGHT / (2 * FY)))
    
    print(f"--- 开始批量任务 ---")
    print(f"推算 FOVY: {FOVY:.4f}° | 分辨率: {WIDTH}x{HEIGHT}")

    # 遍历 txt 文件夹下的所有 .txt 文件
    for file in os.listdir(txt_dir):
        if not file.endswith(".txt"):
            continue
            
        task_name = os.path.splitext(file)[0]  # 比如 "zigzag_10_back_left"
        
        current_pose_file = os.path.join(txt_dir, file)
        current_image_folder = os.path.join(base_dir, task_name)
        current_output_folder = os.path.join(output_base, task_name)
        
        # 检查图片文件夹是否存在
        if not os.path.exists(current_image_folder):
            print(f"跳过任务 [{task_name}]: 未找到图片文件夹 {current_image_folder}")
            continue
            
        print(f"\n>>> 正在处理任务: {task_name}")
        
        # 调用你原本的处理函数
        process_images(
            current_pose_file, 
            current_image_folder, 
            current_output_folder, 
            FOVY, WIDTH, HEIGHT
        )

# 修改原本的 process_images 中保存后缀的一行，建议改为 .jpg
# output_path = os.path.join(output_folder, base_name + ".jpg") 
# img.save(output_path, "JPEG", exif=exif_bytes, quality=95)

# if __name__ == "__main__":
#     # 配置根路径
#     TXT_DIRECTORY = "/media/amax/PS2000/google"           # 存放所有 txt 的地方
#     IMAGE_ROOT = "/media/amax/PS2000/render"             # 存放所有图片文件夹的地方
#     OUTPUT_ROOT = "/media/amax/PS2000/render_pose"       # 输出结果的地方
    
#     batch_process_all_folders(IMAGE_ROOT, TXT_DIRECTORY, OUTPUT_ROOT)

if __name__ == "__main__":
    # 你的 render_camera 数据
    RENDER_PARAMS = [1920, 1080, 957.85, 537.55, 1920, 1080, 1350.0, 1350.0]
    target_name = [
        "zigzag_3_back_left",
        "zigzag_3_back_right",
        "zigzag_3_front_left",
        "zigzag_3_front_right",
        "zigzag_3_center",
        "zigzag_4_back_left",
        "zigzag_4_back_right",
        "zigzag_4_front_left",
        "zigzag_4_front_right",
        "zigzag_4_center",
        # "zigzag_2_center"
    ]
    for target in target_name:
    
        # 配置
        pose_file = f"/media/amax/PS2000/google/{target}.txt"
        image_folder = f"/media/amax/PS2000/render/{target}"
        output_folder = f"/media/amax/PS2000/render_pose/{target}"
    
        # 直接提取参数
        IMAGE_WIDTH = RENDER_PARAMS[0]
        IMAGE_HEIGHT = RENDER_PARAMS[1]
        FX = RENDER_PARAMS[6]
        FY = RENDER_PARAMS[7]
        CX = RENDER_PARAMS[2]
        CY = RENDER_PARAMS[3]

        # 计算得到的 FOVY 约为 43.6028

        FOVY = math.degrees(2 * math.atan(IMAGE_HEIGHT / (2 * FY)))
        
        print(f"根据内参推算的 FOVY 为: {FOVY:.4f} 度")

        # 批量处理
        process_images(pose_file, image_folder, output_folder, FOVY, IMAGE_WIDTH, IMAGE_HEIGHT)

# # ==============================
# if __name__ == "__main__":
#     # 你的 render_camera 数据
#     RENDER_PARAMS = [1920, 1080, 957.85, 537.55, 1920, 1080, 1350.0, 1350.0]
    
#     # 配置
#     pose_file = "/media/amax/PS2000/google/zigzag_10_back_left.txt"
#     image_folder = "/media/amax/PS2000/render/zigzag_10_back_left"
#     output_folder = "/media/amax/PS2000/render_pose/zigzag_10_back_left"
    
#     # 直接提取参数
#     IMAGE_WIDTH = RENDER_PARAMS[0]
#     IMAGE_HEIGHT = RENDER_PARAMS[1]
#     FX = RENDER_PARAMS[6]
#     FY = RENDER_PARAMS[7]
#     CX = RENDER_PARAMS[2]
#     CY = RENDER_PARAMS[3]

#     # 计算得到的 FOVY 约为 43.6028
#     import math
#     FOVY = math.degrees(2 * math.atan(IMAGE_HEIGHT / (2 * FY)))
    
#     print(f"根据内参推算的 FOVY 为: {FOVY:.4f} 度")

#     # 批量处理
#     process_images(pose_file, image_folder, output_folder, FOVY, IMAGE_WIDTH, IMAGE_HEIGHT)
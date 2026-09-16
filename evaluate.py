# from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate, evaluate_all


# # gt_pose = "/home/amax/Documents/datasets/poses/DJI_20251221132259_0002_V.txt"
# # # estimated_pose = "/home/amax/Documents/datasets/outputs/FPVLoc/DJI_20251221132259_0002_V.txt"
# # estimated_pose_low = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_l/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_high = "/home/amax/Documents/datasets/outputs/FPVLoc_l/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_low = "/home/amax/Documents/datasets/outputs/FPVLoc/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_high_dom_low_dsm = "/home/amax/Documents/datasets/outputs/FPVLoc_domhigh/DJI_20251221132259_0002_V.txt"
# # estimated_pose_crop_low_dom_high_dsm = "/home/amax/Documents/datasets/outputs/FPVLoc_dsmhigh/DJI_20251221132259_0002_V.txt"
# # # print("--------------------------------ya结果--------------------------------")
# # # evaluate(estimated_pose, gt_pose)
# # print("--------------------------------高精度结果--------------------------------")
# # evaluate(estimated_pose_high, gt_pose)
# # print("--------------------------------低精度结果--------------------------------")
# # evaluate(estimated_pose_low, gt_pose)
# # print("--------------------------------crop高精度结果--------------------------------")
# # evaluate(estimated_pose_crop_high, gt_pose)
# # print("--------------------------------crop低精度结果--------------------------------")
# # evaluate(estimated_pose_crop_low, gt_pose)
# # print("--------------------------------crop高精度DOM+低精度DSM结果--------------------------------")
# # evaluate(estimated_pose_crop_high_dom_low_dsm, gt_pose)
# # print("--------------------------------crop低精度DOM+高精度DSM结果--------------------------------")
# # evaluate(estimated_pose_crop_low_dom_high_dsm, gt_pose)

# gt_pose = "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_low/DJI_20251221131951_0001_V.txt"
# estimated_pose = "/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221131951_0001_V.txt"
# # evaluate_XYZ_EULER(estimated_pose, gt_pose)
# print("crop")
# evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20250612194622_0018_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth_angle/DJI_20250612194622_0018_V.txt")
# # evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132451_0003_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth/DJI_20251221132451_0003_V.txt")
# # evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth/DJI_20251221132525_0004_V.txt")
# # # print("depth")
# # evaluate_all("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs/FPVLoc_depth/DJI_20251221132525_0004_V.txt")

# #DJI_20251221132259_0002_V



from pixloc.utils.eval import evaluate_xyz, evaluate_XYZ_EULER, evaluate
import os

# target_names = [
#     "switzerland_seq4@8@foggy@intensity1@200",
#     "switzerland_seq4@8@night@intensity1@200",
#     "switzerland_seq4@8@sunny@200",
#     "switzerland_seq4@8@rainy@200",
#     "switzerland_seq4@8@cloudy@200",
#     "switzerland_seq4@8@sunset@200",
#     "switzerland_seq4@8@foggy@intensity3@500",
#     "switzerland_seq4@8@night@intensity3@500",
#     "switzerland_seq4@8@sunny@500",
#     "switzerland_seq4@8@rainy@500",
#     "switzerland_seq4@8@cloudy@500",
#     "switzerland_seq4@8@sunset@500",
#     "switzerland_seq4@8@sunny@screen16@500",
#     "switzerland_seq4@8@sunny@screen8@500",
#     "switzerland_seq4@8@foggy@intensity2@500",
#     "switzerland_seq4@8@night@intensity2@500",
# ]
# print(f"--------------angle------------------")
# for target_name in target_names:
#     if os.path.exists(f"/home/ps/Documents/liuxy24/PiLoT_v55/55/output/{target_name}.txt"):
#         print(f"--------------{target_name}------------------")
#         evaluate(f"/home/ps/Documents/liuxy24/PiLoT_v55/55/output/{target_name}.txt", f"/mnt/data1/UserData/liuxy/Mapscape/Test/poses/{target_name}.txt")
#     else:
#         print(f"--------------{target_name} not found------------------")
# print(f"--------------none------------------")
# for target_name in target_names:
#     if os.path.exists(f"/home/ps/Documents/liuxy24/PiLoT_v55/outputs_none/{target_name}.txt"):
#         print(f"--------------{target_name}------------------")
#         evaluate(f"/home/ps/Documents/liuxy24/PiLoT_v55/outputs_none/{target_name}.txt", f"/mnt/data1/UserData/liuxy/Mapscape/Test/poses/{target_name}.txt")
# print(f"--------------angle&depth------------------")
# for target_name in target_names:
#     if os.path.exists(f"/home/ps/Documents/liuxy24/PiLoT_v55/outputs_all/{target_name}.txt"):
#         print(f"--------------{target_name}------------------")
#         evaluate(f"/home/ps/Documents/liuxy24/PiLoT_v55/outputs_all/{target_name}.txt", f"/mnt/data1/UserData/liuxy/Mapscape/Test/poses/{target_name}.txt")
# print(f"--------------angle_1.0------------------")
# for target_name in target_names:
#     if os.path.exists(f"/home/ps/Documents/liuxy24/PiLoT_v55/angle_1.0/{target_name}.txt"):
#         print(f"--------------{target_name}------------------")
#         evaluate(f"/home/ps/Documents/liuxy24/PiLoT_v55/angle_1.0/{target_name}.txt", f"/mnt/data1/UserData/liuxy/Mapscape/Test/poses/{target_name}.txt")










# print("--------------{target_name}------------------")
# evaluate_XYZ_EULER(f"/media/amax/AE0E2AFD0E2ABE69/datasets/poses/{target_name}.txt", f"/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_dem/{target_name}.txt")
#   print("--------------经纬加噪声3m，yaw加3deg------------------")
#   evaluate_XYZ_EULER(f"/media/amax/AE0E2AFD0E2ABE69/datasets/poses/{target_name}.txt", f"/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_noise/{target_name}_noise.txt")

# print("--------------estimated------------------")
# evaluate_XYZ_EULER("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_noise/DJI_20251221132525_0004_V.txt")
# print("--------------经纬加噪声3m，yaw加3deg------------------")
# evaluate_XYZ_EULER("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20251221132525_0004_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_noise/DJI_20251221132525_0004_V_noise.txt")
# print("--------------prior------------------")
# evaluate_XYZ_EULER("/media/amax/AE0E2AFD0E2ABE69/datasets/poses/DJI_20250612194622_0018_V.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/FPVLoc_high/DJI_20250612194622_0018_V_prior.txt")

# evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/outputs/interval1_AMvalley01_1.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/poses/interval1_AMvalley01_1.txt")
# evaluate("/media/amax/AE0E2AFD0E2ABE69/datasets/outputs_render/uavscene/interval1_AMvalley01_1.txt", "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/poses/interval1_AMvalley01_1.txt")

evaluate("/media/amax/AE0E2AFD0E2ABE69/outputs_ortholoc/GIM_dkm_out/DJI_20250612193930_0012_V.txt", "/media/amax/AE0E2AFD0E2ABE69/outputs_ortholoc/GIM_dkm/DJI_20250612193930_0012_V.txt")
#!/usr/bin/env bash
# run_by_names.sh

set -euo pipefail

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==== 当前仓库入口 ====
MAIN_PY="${SCRIPT_DIR}/main.py"
CONFIG_PATH="${SCRIPT_DIR}/configs/feicuiwan_m4t.yaml"
CROP_K_JSON="${CROP_K_JSON:-${SCRIPT_DIR}/crop/K_feicuiwan_m4t.json}"
ORTHOLOC_INTRINSICS="${SCRIPT_DIR}/OrthoLoC/OrthoLoC/dji_intrinsics.json"

# ==== 地图数据 ====
# 如需切换地图，优先改这里
DOM_PATH="${DOM_PATH:-/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_ortho_merge.tif}"
DSM_PATH="${DSM_PATH:-/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif}"

# ==== OrthoLoC 参数 ====
ORTHOLOC_MATCHER="${ORTHOLOC_MATCHER:-Mast3R}"
ORTHOLOC_DEVICE="${ORTHOLOC_DEVICE:-cuda}"
ORTHOLOC_ANGLES=(${ORTHOLOC_ANGLES:-0 90 180 270})
MAX_GT_RECROPS="${MAX_GT_RECROPS:-1}"
GT_RESET_TRANSLATION_THRESH_M="${GT_RESET_TRANSLATION_THRESH_M:-50}"
GT_RESET_ROTATION_THRESH_DEG="${GT_RESET_ROTATION_THRESH_DEG:-50}"

# ==== 所有配置（names 作为 key） ====
names=(
  "switzerland_seq4@8@sunny@500"
  "USA_seq5@8@sunset@300"
  "cloudy_400"
  "DJI_20250612194040_0013_V_900"
  "DJI_20250612173353_0002_V"
  "DJI_20250612174308_0001_V"
  "DJI_20250612182017_0001_V"
  "DJI_20250612182732_0001_V"
  "DJI_20250612183852_0005_V"
  "DJI_20250612193704_0010_V"
  "DJI_20250612193930_0012_V"
  "DJI_20250612194150_0014_V"
  "DJI_20250612194622_0018_V"
  "DJI_20250612194903_0021_V"
  "DJI_20250612173353_0002_V_test"
  "DJI_20250804192327_0002_V"
  "DJI_20241113180128_0042_D"
  "DJI_20250926152036_0002_V"
  "DJI_20250926152037_0002_V"
  "feicuiwan_sim_seq7"
  "test"
  "test_0018_V"
  "DJI_20250612194622_0018_V_1"
  "DJI_20251217153719_0002_V"
  "DJI_20251221132525_0004_V"
  "DJI_20251221132451_0003_V"
  "DJI_20251221132451_0003_V1"
  "DJI_20251221132451_0003_V2"
  "DJI_20251221132451_0003_V3"
  "DJI_20251221132259_0002_V"
  "DJI_20251221131951_0001_V_1"
  "DJI_20251221131951_0001_V"
  "DJI_20250612194622_0018_V"
  "DJI_20250612174308_0001_V"
  "DJI_20250612194622_0018_V"
  "DJI_20251119165945_0006_V"
  "DJI_20250612173353_0002_V"
  "0018_V"
  "DJI_20250612194622_0018_V"
  "DJI_20250612174308_0001_V"
  "DJI_20250612182017_0001_V"
  "DJI_20250612182732_0001_V"
  "DJI_20250612183852_0005_V"
  "DJI_20250612193930_0012_V"  
  "DJI_20250612194150_0014_V"
  "DJI_20250612194903_0021_V"
  "2017_0001_V"
  "DJI_20250612174308_0001_V_1"
  "DJI_20250612182017_0001_V_1"
  "DJI_20250612193930_0012_V_1"
  "DJI_20251221132525_0004_V_1"
  "DJI_20250612194622_0018_V"
  "DJI_20250612174308_0001_V"
  "DJI_20250612182017_0001_V"
  "DJI_20250612182732_0001_V"
  "DJI_20250612183852_0005_V"
  "DJI_20250612193930_0012_V"
  "DJI_20250612194150_0014_V"
  "DJI_20250612194903_0021_V"
)


# ==== 你想运行哪些 name？====
target_names=(
  "DJI_20250612194150_0014_V" 
  "DJI_20250612182732_0001_V"
  "DJI_20250612183852_0005_V"
  # "DJI_20250612193930_0012_V"
  "DJI_20250612194903_0021_V"
  "DJI_20250612194622_0018_V"
  # "DJI_20250612182017_0001_V"

)

# ==== 从 txt 中读取 init_euler 和 init_trans ====
read_pose_from_file() {
  local name="$1"
  local pose_file="/media/amax/AE0E2AFD0E2ABE69/datasets/poses/${name}.txt"

  if [[ ! -f "$pose_file" ]]; then
    echo "❌ 找不到 pose 文件: $pose_file"
    return 1
  fi

  local first_line
  first_line=$(head -n 1 "$pose_file")

  # 解析：name lon lat alt roll pitch yaw
  read -r _ lon lat alt roll pitch yaw <<< "$first_line"

  # 构造 init_euler 和 init_trans
  init_euler="[$pitch, $roll, $yaw]"
  init_trans="[$lon, $lat, $alt]"

  echo "$init_euler|$init_trans"
  return 0
}  # ✅ 这一行必须存在，否则后续 for/if 会错乱

# ==== 遍历 target_names ====
for target_name in "${target_names[@]}"; do
  index=-1
  for i in "${!names[@]}"; do
    if [[ "${names[$i]}" == "$target_name" ]]; then
      index=$i
      break
    fi
  done

  if [[ $index -ge 0 ]]; then
    result=$(read_pose_from_file "$target_name")
    if [[ $? -ne 0 ]]; then
      echo "❌ 无法读取 $target_name 的位姿，跳过"
      continue
    fi

    IFS='|' read -r euler trans <<< "$result"

    echo "==== 正在运行 $target_name ===="
    echo "euler : $euler"
    echo "trans : $trans"

    echo "--- crop + ortholoc"
    python "$MAIN_PY" \
      --config "$CONFIG_PATH" \
      --init_euler "$euler" \
      --init_trans "$trans" \
      --name "$target_name" \
      --dom_path "$DOM_PATH" \
      --dsm_path "$DSM_PATH" \
      --crop_k_json "$CROP_K_JSON" \
      --ortholoc_intrinsics "$ORTHOLOC_INTRINSICS" \
      --ortholoc_matcher "$ORTHOLOC_MATCHER" \
      --ortholoc_device "$ORTHOLOC_DEVICE" \
      --ortholoc_angles "${ORTHOLOC_ANGLES[@]}" \
      --max_gt_recrops "$MAX_GT_RECROPS" \
      --gt_reset_translation_thresh_m "$GT_RESET_TRANSLATION_THRESH_M" \
      --gt_reset_rotation_thresh_deg "$GT_RESET_ROTATION_THRESH_DEG" \
      --continue_on_error

    echo -e "==== 运行 $target_name 结束 ====\n"

    # ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    # ps aux | grep multiprocessing.resource_tracker | grep -v grep | awk '{print $2}' | xargs kill -9 || true
  else
    echo "❌ 未找到 name=$target_name 对应的配置"
  fi
done

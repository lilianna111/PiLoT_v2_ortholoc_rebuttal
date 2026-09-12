from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from .proj2map import generate_ref_map
from .transform_colmap import normalize_K


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_transform_to_npy(npy_data, matrix, new_size, x_min, y_min, target_w, target_h):
    rotated_npy = cv2.warpAffine(
        npy_data,
        matrix,
        new_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_npy[y_min:y_min + target_h, x_min:x_min + target_w]


def rotate_all_with_mask(image, npy_data, angle, mask=None, crop_padding=0):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    angle_rad = np.radians(angle)
    new_w = int(h * abs(np.sin(angle_rad)) + w * abs(np.cos(angle_rad)))
    new_h = int(w * abs(np.sin(angle_rad)) + h * abs(np.cos(angle_rad)))

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += (new_w - w) // 2
    matrix[1, 2] += (new_h - h) // 2

    rotated_img = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    if mask is not None:
        rotated_mask = cv2.warpAffine(
            mask,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        ys, xs = np.where(rotated_mask > 0)
        if len(xs) > 0:
            x_min = max(int(xs.min()) - crop_padding, 0)
            x_max = min(int(xs.max()) + crop_padding + 1, new_w)
            y_min = max(int(ys.min()) - crop_padding, 0)
            y_max = min(int(ys.max()) + crop_padding + 1, new_h)

            final_img = rotated_img[y_min:y_max, x_min:x_max]
            final_mask = rotated_mask[y_min:y_max, x_min:x_max]
            final_npy = apply_transform_to_npy(
                npy_data,
                matrix,
                (new_w, new_h),
                x_min,
                y_min,
                final_img.shape[1],
                final_img.shape[0],
            )
            final_npy[final_mask == 0] = 0.0
            final_img[final_mask == 0] = 255
            return final_img, final_npy, final_mask

    return rotated_img, None, None


def run_test_0310_style_pipeline(
    dsm_path: str,
    dom_path: str,
    pose_data,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    output_dir: str | None = None,
    name: str = "1",
    crop_padding: int = 0,
    num_samples: int = 12000,
    t_near: float = 1.0,
    t_far: float = 20000.0,
    save_outputs: bool = False,
    save_debug_vis: bool = False,
    save_full_dom_points: bool = False,
    print_corner_intrinsics: bool = False,
) -> dict[str, Any]:
    K = normalize_K(K)

    if (save_outputs or save_debug_vis or save_full_dom_points) and output_dir is None:
        raise ValueError("output_dir is required when save_outputs=True or save_debug_vis=True or save_full_dom_points=True")
    if output_dir is not None and (save_outputs or save_debug_vis or save_full_dom_points):
        _ensure_dir(output_dir)

    result = generate_ref_map(
        dsm_path=dsm_path,
        dom_path=dom_path,
        pose_data=pose_data,
        K=K,
        image_width=image_width,
        image_height=image_height,
        output_dir=output_dir,
        name=name,
        crop_padding=crop_padding,
        num_samples=num_samples,
        t_near=t_near,
        t_far=t_far,
        save_bbox=save_outputs,
        save_debug_vis=save_debug_vis,
        save_full_dom_points=save_full_dom_points,
        print_corner_intrinsics=print_corner_intrinsics,
    )

    img_aabb = result["bbox_dom_rgb"]
    mask_aabb = result["crop_mask"]
    npy_aabb = result["point_cloud_crop"]
    yaw_angle = float(result["pose_meta"]["yaw_processed"])

    final_img, final_npy, final_mask = rotate_all_with_mask(
        img_aabb,
        npy_aabb,
        yaw_angle,
        mask=mask_aabb,
        crop_padding=0,
    )

    if final_img is None or final_npy is None or final_mask is None:
        final_img = img_aabb
        final_npy = npy_aabb
        final_mask = mask_aabb

    final_img_path = None
    final_npy_path = None
    final_mask_path = None

    if output_dir is not None and save_outputs:
        final_img_path = os.path.join(output_dir, f"{name}_final_img.jpg")
        final_npy_path = os.path.join(output_dir, f"{name}_final_points.npy")
        final_mask_path = os.path.join(output_dir, f"{name}_final_mask.png")
        cv2.imwrite(final_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        np.save(final_npy_path, final_npy)
        cv2.imwrite(final_mask_path, final_mask)

    if output_dir is not None and save_debug_vis and not save_outputs:
        final_img_path = os.path.join(output_dir, f"{name}_final_img.jpg")
        final_mask_path = os.path.join(output_dir, f"{name}_final_mask.png")
        cv2.imwrite(final_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(final_mask_path, final_mask)

    saved_paths = {}
    if result.get("bbox_dom_path"):
        saved_paths["bbox_img"] = result["bbox_dom_path"]
    if result.get("bbox_npy_path"):
        saved_paths["bbox_npy"] = result["bbox_npy_path"]
    if result.get("bbox_mask_path"):
        saved_paths["bbox_mask"] = result["bbox_mask_path"]
    if result.get("debug_overlay_path"):
        saved_paths["crop_points_on_dom"] = result["debug_overlay_path"]
    if result.get("full_dom_points_path"):
        saved_paths["points_on_full_dom"] = result["full_dom_points_path"]
    if final_img_path:
        saved_paths["final_img"] = final_img_path
    if final_npy_path:
        saved_paths["final_points"] = final_npy_path
    if final_mask_path:
        saved_paths["final_mask"] = final_mask_path

    result["bbox_img"] = img_aabb
    result["bbox_points"] = npy_aabb
    result["bbox_mask"] = mask_aabb
    result["final_img"] = final_img
    result["final_points"] = final_npy
    result["final_mask"] = final_mask
    result["saved_paths"] = saved_paths
    return result

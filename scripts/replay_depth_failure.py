"""Audit a failed PnP snapshot on CPU; no matching, cropping or output writes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyproj
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "OrthoLoC" / "OrthoLoC"))
from ortholoc.depth import DepthDSM, DepthPrior
from ortholoc.gravity import require_gravity_backend


def manual_lookup(area, transform, nodata, x, y, center_index=False):
    """Independent four-neighbour interpolation, without DepthDSM/SciPy."""
    col, row = (~transform) * (x, y)
    if center_index:
        col, row = col - 0.5, row - 0.5
    result = {"row_col": [row, col], "neighbors_m": None, "height_m": None}
    rows, cols = area.shape
    if not np.isfinite([row, col]).all() or not (0 <= row <= rows - 1 and 0 <= col <= cols - 1):
        return result
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
    v = np.asarray(area[np.ix_([r0, r1], [c0, c1])], dtype=np.float64)
    result["neighbors_m"] = [[float(z) if np.isfinite(z) else None for z in line] for line in v]
    if not np.isfinite(v).all() or (nodata is not None and (v == nodata).any()):
        return result
    fr, fc = row - r0, col - c0
    result["height_m"] = float((1 - fr) * ((1 - fc) * v[0, 0] + fc * v[0, 1]) +
                               fr * ((1 - fc) * v[1, 0] + fc * v[1, 1]))
    return result


def audit_candidate(rt, prior, area, transform, crs, nodata):
    """Direct absolute-ECEF computation: no Euler pose reconstruction."""
    R, t = rt[:, :3], rt[:, 3]
    centre = -R.T @ t + prior.ecef_origin
    t_absolute = t - R @ prior.ecef_origin
    np.testing.assert_allclose(-R.T @ t_absolute, centre, atol=1e-5, rtol=0)
    cam_point = prior.depth * np.linalg.solve(prior.K, [*prior.pixel, 1.0])
    endpoint = centre + R.T @ cam_point
    to_wgs = pyproj.Transformer.from_crs(4978, 4326, always_xy=True)
    point = list(to_wgs.transform(*endpoint))
    x, y = pyproj.Transformer.from_crs(4326, crs, always_xy=True).transform(*point[:2])
    lookup = manual_lookup(area, transform, nodata, x, y)
    centered = manual_lookup(area, transform, nodata, x, y, center_index=True)
    residual = None if lookup["height_m"] is None else point[2] - lookup["height_m"]
    copied = prior.details(rt)
    copied_endpoint = pyproj.Transformer.from_crs(4326, 4978, always_xy=True).transform(
        *copied["depth_point_wgs84"])
    np.testing.assert_allclose(copied_endpoint, endpoint, atol=1e-5, rtol=0)
    if residual is None:
        assert copied["depth_residual_m"] is None
    else:
        np.testing.assert_allclose(residual, copied["depth_residual_m"], atol=1e-4, rtol=0)
    return {"camera_center_absolute_ecef_m": centre.tolist(),
            "camera_wgs84": list(to_wgs.transform(*centre)),
            "point_absolute_ecef_m": endpoint.tolist(), "point_wgs84": point,
            "projected_xy_m": [x, y], "existing_index_lookup": lookup,
            "pixel_center_index_lookup": centered, "independent_residual_m": residual,
            "callback_residual_m": copied["depth_residual_m"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        x, X = data["points2D"], data["points3D"]
        origin, K = data["ecef_origin"], data["K"]
        camera_up, world_up = data["camera_up"], data["world_up"]
    camera, recorded = meta["camera"], meta["stats"]
    fusion = recorded.get("prior_fusion", "hard")
    dsm = DepthDSM.from_file(meta["dsm_path"])
    assert dsm.proj_crs == meta["dsm_crs"], "DSM CRS differs from the captured run"
    np.testing.assert_array_equal(dsm.geotransform, meta["dsm_geotransform"])
    prior = DepthPrior(dsm, origin, meta["depth_prior_m"], K, camera["width"], camera["height"])
    np.testing.assert_array_equal(prior.pixel, meta["depth_pixel_uv"])
    closest = recorded.get("closest_depth_candidate")
    report = {"image": meta["image"], "snapshot": str(args.snapshot),
              "num_native_points": len(x), "depth_prior_m": prior.depth,
              "gravity_threshold_deg": meta["gravity_threshold_deg"],
              "depth_threshold_m": meta["depth_threshold_m"], "closest_candidate_audit": None}
    if closest is not None:
        with rasterio.open(meta["dsm_path"]) as dataset:
            audit = audit_candidate(np.asarray(closest["pose_w2c_local_ecef"]), prior,
                                    dataset.read(1), dataset.transform, dataset.crs, dataset.nodata)
        np.testing.assert_allclose(audit["independent_residual_m"], closest["depth_residual_m"],
                                   atol=1e-4, rtol=0)
        if recorded["gravity_enabled"]:
            R = np.asarray(closest["pose_w2c_local_ecef"])[:, :3]
            dot = (camera_up / np.linalg.norm(camera_up)).dot(R @ (world_up / np.linalg.norm(world_up)))
            angle = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))
            if fusion == "hard":
                assert angle <= meta["gravity_threshold_deg"] + 1e-6
            np.testing.assert_allclose(angle, closest["gravity_error_deg"], atol=1e-6, rtol=0)
            audit["independent_gravity_error_deg"] = angle
        report["closest_candidate_audit"] = audit
    backend = require_gravity_backend()
    if not getattr(backend, "supports_depth_diagnostics", False):
        raise RuntimeError("Rebuild with bash scripts/build_gravity_pnp.sh")
    rt, replay = backend.estimate_absolute_pose(
        x, X, camera, meta["options"], camera_up, world_up, meta["gravity_threshold_deg"],
        enforce_refinement_gravity=recorded["enforce_refinement_gravity"],
        depth_evaluator=prior, depth_threshold_m=meta["depth_threshold_m"],
        use_gravity=recorded["gravity_enabled"], prior_fusion=fusion,
        gravity_scale_deg=recorded.get("gravity_scale_deg", 10.),
        depth_scale_m=recorded.get("depth_scale_m", 10.),
        gravity_weight=recorded.get("gravity_weight", 0.1), depth_weight=recorded.get("depth_weight", 0.1))
    for key in ("success", "iterations", "num_inliers", "generated_hypotheses", "rejected_hypotheses",
                "nonfinite_hypotheses", "gravity_rejected_hypotheses", "depth_rejected_hypotheses",
                "invalid_depth_hypotheses", "rejected_refinements", "final_refinement_accepted"):
        assert replay[key] == recorded[key], f"Replay differs in {key}: {replay[key]} != {recorded[key]}"
    if closest is None:
        assert replay["closest_depth_candidate"] is None
    else:
        np.testing.assert_allclose(replay["closest_depth_candidate"]["pose_w2c_local_ecef"],
                                   closest["pose_w2c_local_ecef"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(replay["closest_depth_candidate"]["depth_residual_m"],
                                   closest["depth_residual_m"], atol=1e-5, rtol=0)
    assert rt is None and not replay["success"], "Snapshot was not a failed PnP run"
    report["replay_matches_failure"] = True
    report["replay_stats"] = {k: v for k, v in replay.items()
                              if k not in ("inliers", "closest_depth_candidate")}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

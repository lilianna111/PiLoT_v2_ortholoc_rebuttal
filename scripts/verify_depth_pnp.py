"""Replay frozen PnP inputs on CPU, without rematching, cropping or deleting runs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import poselib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "OrthoLoC" / "OrthoLoC"))
from ortholoc.depth import DepthDSM, DepthPrior, ECEF_to_WGS84, WGS84_to_ECEF, load_depth_priors
from ortholoc.gravity import require_gravity_backend


def verify_returned_pose(rt, info, points2D, points3D, camera, options, prior, camera_up, world_up, use_gravity):
    # Independent direct-ECEF backprojection; no Euler/WGS84 pose reconstruction.
    centre = -rt[:, :3].T @ rt[:, 3] + prior.ecef_origin
    cam_vec = prior.depth * np.linalg.solve(prior.K, [*prior.pixel, 1.])
    endpoint_ecef = centre + rt[:, :3].T @ cam_vec
    endpoint = ECEF_to_WGS84(endpoint_ecef)
    height = prior.dsm.query_heights([endpoint])[0]
    error = endpoint[2] - height
    assert np.isfinite(error)
    soft = info.get("prior_fusion", "hard") == "soft"
    if not soft:
        assert abs(error) <= info["depth_threshold_m"] + 1e-6
    details = prior.details(rt)
    np.testing.assert_allclose(WGS84_to_ECEF(details["depth_point_wgs84"]), endpoint_ecef, atol=1e-5, rtol=0)
    np.testing.assert_allclose(error, details["depth_residual_m"], atol=1e-5, rtol=0)
    if use_gravity:
        angle = np.degrees(np.arccos(np.clip(camera_up.dot(rt[:, :3] @ world_up), -1., 1.)))
        if not soft:
            assert angle <= info["gravity_threshold_deg"] + 1e-6
    fx, fy, cx, cy = camera["params"]
    cam_points = points3D @ rt[:, :3].T + rt[:, 3]
    residual = cam_points[:, :2] / cam_points[:, 2:] - (points2D - [cx, cy]) / [fx, fy]
    mask = np.sum(residual ** 2, axis=1) < (options["max_reproj_error"] / ((fx + fy) / 2)) ** 2
    mask &= cam_points[:, 2] > 0
    np.testing.assert_array_equal(mask, info["inliers"])
    assert int(mask.sum()) == info["num_inliers"]
    if soft and info["soft_priors_active"]:
        # Same normalized MSAC, including points behind the camera as outliers.
        tau2 = (options["max_reproj_error"] / ((fx + fy) / 2)) ** 2
        squared = np.sum(residual ** 2, axis=1)
        squared[cam_points[:, 2] <= 0] = tau2
        visual = np.minimum(squared, tau2).sum() / (len(points2D) * tau2)
        rho = lambda r, scale: 2 * np.log1p(0.5 * (r / scale) ** 2)
        gravity_cost = info["gravity_weight"] * rho(angle, info["gravity_scale_deg"]) if use_gravity else 0.
        depth_cost = info["depth_weight"] * rho(error, info["depth_scale_m"])
        np.testing.assert_allclose(info["visual_score_normalized"], visual, atol=1e-9, rtol=0)
        np.testing.assert_allclose(info["gravity_prior_cost"], gravity_cost, atol=1e-8, rtol=0)
        np.testing.assert_allclose(info["depth_prior_cost"], depth_cost, atol=1e-6, rtol=0)
        np.testing.assert_allclose(info["joint_score"], visual + gravity_cost + depth_cost, atol=1e-6, rtol=0)
    return error, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True, help="Existing gravity_check_v2 NPZ directory")
    parser.add_argument("--depth-file", type=Path, required=True)
    parser.add_argument("--dsm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist; creates report.json only")
    parser.add_argument("--prior-fusion", choices=("hard", "soft"), default="hard")
    parser.add_argument("--gravity-scale-deg", type=float, default=10.)
    parser.add_argument("--depth-scale-m", type=float, default=10.)
    parser.add_argument("--gravity-weight", type=float, default=.1)
    parser.add_argument("--depth-weight", type=float, default=.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Will not overwrite {args.output_dir}")
    backend = require_gravity_backend()
    if not getattr(backend, "supports_depth", False):
        raise RuntimeError("Rebuild with bash scripts/build_gravity_pnp.sh")
    if args.prior_fusion == "soft" and not getattr(backend, "supports_soft_priors", False):
        raise RuntimeError("Rebuild the soft-scoring backend with bash scripts/build_gravity_pnp.sh")
    files = sorted(args.frozen_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No frozen PnP inputs in {args.frozen_dir}")
    query_dir = args.depth_file.parent.parent / "images" / args.depth_file.stem
    names = [p.name for p in query_dir.iterdir() if p.suffix.lower() in (".png", ".jpg")]
    depths = load_depth_priors(args.depth_file, names)
    dsm = DepthDSM.from_file(args.dsm)
    results = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            camera, options = meta["camera"], meta["options"]
            if camera["model"] != "PINHOLE":
                raise ValueError("Depth replay requires PINHOLE frozen intrinsics")
            x, X = data["points2D"], data["points3D"]
            up_cam, up_world = data["camera_up"], data["world_up"]
            fx, fy, cx, cy = camera["params"]
            # Frozen inputs already use COLMAP pixels. The prior uses the original OpenCV K.
            K = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1.]])
            prior = DepthPrior(dsm, data["ecef_origin"], depths[meta["image"]], K,
                               camera["width"], camera["height"])
            original, _ = poselib.estimate_absolute_pose(x, X, camera, options, {})
            assert original is not None, f"Original PnP failed for {meta['image']}"
            records = {}
            suffix = "soft" if args.prior_fusion == "soft" else "strict"
            for label, evaluator, use_gravity, fusion in [
                ("depth_unrestricted_control", lambda rt: 0., False, "hard"),
                (f"depth_{suffix}", prior, False, args.prior_fusion),
                (f"gravity_depth_{suffix}", prior, True, args.prior_fusion),
                ("depth_all_rejected_control", lambda rt: 10.01, False, "hard"),
            ]:
                rt, info = backend.estimate_absolute_pose(
                    x, X, camera, options, up_cam, up_world, meta["threshold"],
                    depth_evaluator=evaluator, use_gravity=use_gravity, prior_fusion=fusion,
                    gravity_scale_deg=args.gravity_scale_deg, depth_scale_m=args.depth_scale_m,
                    gravity_weight=args.gravity_weight, depth_weight=args.depth_weight)
                info["gravity_threshold_deg"] = meta["threshold"]
                record = {k: v for k, v in info.items() if k != "inliers"}
                closest = record.get("closest_depth_candidate")
                if closest is not None:
                    closest["pose_w2c_local_ecef"] = np.asarray(closest["pose_w2c_local_ecef"]).tolist()
                record["pose_w2c_local_ecef"] = None if rt is None else rt.tolist()
                if label == "depth_unrestricted_control":
                    assert rt is not None
                    np.testing.assert_allclose(rt, original.Rt, atol=1e-8, rtol=1e-10)
                    record["max_pose_difference_from_original"] = float(np.max(np.abs(rt - original.Rt)))
                elif label == "depth_all_rejected_control":
                    assert rt is None and not info["success"] and not np.any(info["inliers"])
                    assert info["generated_hypotheses"] == info["rejected_hypotheses"]
                elif rt is not None:
                    error, details = verify_returned_pose(rt, info, x, X, camera, options, prior,
                                                         up_cam, up_world, use_gravity)
                    record.update(details, independent_depth_residual_m=error)
                    checks = "geometry/scores/inliers" if fusion == "soft" else "constraints/inliers"
                    print(f"PASS {meta['image']} {label}: depth={abs(error):.6f} m; {checks} checked")
                else:
                    assert not info["success"] and not np.any(info["inliers"])
                    print(f"NO FEASIBLE POSE {meta['image']} {label}: inspect rejection counters")
                records[label] = record
            results.append({"image": meta["image"], "variants": records})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = args.output_dir / "report.json"
    with report.open("x", encoding="utf-8") as file:
        json.dump({"depth_file": str(args.depth_file), "dsm_path": str(args.dsm),
                   "prior_fusion": args.prior_fusion, "results": results},
                  file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

"""Freeze selected real OrthoLoC PnP inputs and replay solver controls safely."""

import argparse
import ast
import csv
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'OrthoLoC/OrthoLoC'))
from ortholoc.gravity import ecef_up, load_gravity_priors, require_gravity_backend


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def replay(points2D, points3D, camera, options, camera_up, world_up, threshold):
    import poselib

    backend = require_gravity_backend()
    original, original_info = poselib.estimate_absolute_pose(points2D, points3D, camera, options, {})
    records = {'original': {'pose_w2c': original.Rt.tolist(),
                            'num_inliers_before_final_ba': original_info['num_inliers']}}
    for label, tau, enforce, prior in [
        ('gravity_180', 180, True, camera_up),
        ('gravity_strict', threshold, True, camera_up),
        ('gravity_candidates_only', threshold, False, camera_up),
        ('reversed_prior_tight_gate', 1e-8, True, -camera_up),
    ]:
        rt, info = backend.estimate_absolute_pose(
            points2D, points3D, camera, options, prior, world_up, tau,
            enforce_refinement_gravity=enforce)
        record = {k: v for k, v in info.items() if k != 'inliers'}
        record['pose_w2c'] = None if rt is None else rt.tolist()
        if rt is not None:
            independent_angle = angle_deg(rt[:, :3] @ world_up, prior)
            record['independent_gravity_error_deg'] = independent_angle
            np.testing.assert_allclose(independent_angle, info['gravity_error_deg'], atol=1e-5, rtol=0)
            if enforce:
                assert independent_angle <= tau + 1e-6, f'{label}: returned pose violates gravity'
            fx, fy, cx, cy = camera['params']
            camera_points = points3D @ rt[:, :3].T + rt[:, 3]
            residual = camera_points[:, :2] / camera_points[:, 2:] - (points2D - [cx, cy]) / [fx, fy]
            mask = (np.sum(residual ** 2, axis=1) < (options['max_reproj_error'] / ((fx + fy) / 2)) ** 2)
            mask &= camera_points[:, 2] > 0
            np.testing.assert_array_equal(info['inliers'], mask)
            assert info['num_inliers'] == int(mask.sum())
        else:
            assert not info['success'] and not np.any(info['inliers'])
        if label == 'gravity_180':
            assert info['success'], '180-degree control failed'
            record['max_pose_difference_from_original'] = float(np.max(np.abs(rt - original.Rt)))
            np.testing.assert_allclose(rt, original.Rt, atol=1e-8, rtol=1e-10)
            # Degenerate real samples can produce NaN P3P models even at 180°.
            assert info['rejected_hypotheses'] == info['nonfinite_hypotheses']
        if label == 'reversed_prior_tight_gate':
            assert rt is None, 'reversed, tightly gated prior unexpectedly succeeded'
            assert info['generated_hypotheses'] == info['rejected_hypotheses']
        records[label] = record
    return records


def main_functions():
    # Execute only these function definitions, never main.py's imports or constructor.
    import cv2

    wanted = {'ensure_ortholoc_import_path', 'load_ortholoc_components', 'cgcs2000_to_wgs84',
              'cgcs2000_grid_to_local_ecef', 'postprocess_crop_to_render_grid',
              'process_map_crop', 'run_ortholoc_localization_on_crop'}
    source = ast.parse((REPO / 'main.py').read_text(encoding='utf-8'))
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    from crop.crop.transform_colmap import wgs84_to_ecef, ecef_to_wgs84
    namespace = dict(np=np, os=os, sys=sys, cv2=cv2, time=time, logging=logging,
                     __file__=str(REPO / 'main.py'), _CGCS2000_TO_WGS84_TRANSFORMER=None,
                     crop_wgs84_to_ecef=wgs84_to_ecef, crop_ecef_to_wgs84=ecef_to_wgs84)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPO / 'main.py'), 'exec'), namespace)
    return namespace


def regenerate_crops(run, rows, output):
    # Isolate GIS libraries from imcui/pycolmap, as the production workers do.
    import rasterio
    from pixloc.crop.ray_casting import TargetLocation

    dsm_cache = Path(run['dsm_path']).with_suffix('.npy')
    if not dsm_cache.is_file():
        raise FileNotFoundError(f'Require existing DSM cache to avoid dataset writes: {dsm_cache}')
    with rasterio.open(run['dsm_path']) as dataset:
        dsm_data, transform = dataset.read(1), dataset.transform
    with rasterio.open(run['dom_path']) as dataset:
        dom_data = dataset.read([1, 2, 3])
    area = np.load(dsm_cache)
    np.testing.assert_array_equal(area, dsm_data)
    minimum = np.median(area[(area > 0) & (area < 10000)])
    maps = (transform.to_gdal(), area, minimum, dsm_data, transform, dom_data)
    locator = TargetLocation({'ray_casting': {}}, use_dsm=False)
    funcs = main_functions()
    crops = {}
    for name, row in rows.items():
        pose_data = row['pose_data']
        crop = funcs['process_map_crop'](run['dsm_path'], pose_data, str(dsm_cache), Path(name).stem,
                                         maps, (str(output), str(output)), maps[1], maps[2], locator)
        grid = crop['point_cloud_crop']
        valid = np.isfinite(grid).all(axis=-1) & (grid[..., 2] > 0)
        width, height = map(int, row['crop_size'].split('x'))
        color, grid, valid = funcs['postprocess_crop_to_render_grid'](
            crop['dom_crop'], grid, valid, width, height, True)
        grid, origin, _ = funcs['cgcs2000_grid_to_local_ecef'](grid, valid)
        np.testing.assert_allclose(origin, np.fromstring(row['ecef_origin'], sep=','), atol=1e-4, rtol=0)
        crops[name] = (color, grid, origin)
    return crops


def capture(reference, output, names, seed):
    import torch
    from crop.crop.transform_colmap import ecef_to_wgs84

    run = json.loads((reference / 'poses.json').read_text())
    assert run['pnp_prior_mode'] == 'gravity', 'Reference must be an existing gravity run'
    priors = load_gravity_priors(run['gravity_prior_file'], run['gravity_prior_format'])
    with (reference / 'ortholoc_pose_debug.txt').open() as file:
        debug = {r['image']: r for r in csv.DictReader(file, delimiter='\t')}
    frames = {Path(p['image_path']).name: p for p in run['poses']}
    angle_differences = []
    audit_angles = []
    for name, frame in frames.items():
        origin = np.fromstring(debug[name]['ecef_origin'], sep=',')
        up = ecef_up(*ecef_to_wgs84(*origin)[:2])
        np.testing.assert_allclose(frame['gravity_world_up'], up, atol=1e-12, rtol=0)
        np.testing.assert_allclose(frame['gravity_camera_up'], priors[name], atol=1e-12, rtol=0)
        error = angle_deg(np.asarray(frame['pose_w2c'])[:3, :3] @ up, priors[name])
        assert error <= run['gravity_threshold_deg'] + 1e-4
        audit_angles.append(error)
        angle_differences.append(abs(error - frame['pnp_stats']['gravity_error_deg']))
    audit = dict(frames=len(frames), max_gravity_error_deg=max(audit_angles),
                 max_logged_angle_difference_deg=max(angle_differences))
    if not names:
        # Keep one LO and one BA rollback representative, not every affected frame.
        selected = [next(iter(frames)), max(frames, key=lambda k: frames[k]['pnp_stats']['gravity_error_deg'])]
        for field in ['rejected_refinements', 'final_refinement_accepted']:
            representative = next((k for k, p in frames.items()
                                   if (p['pnp_stats'][field] > 0 if field == 'rejected_refinements'
                                       else not p['pnp_stats'][field])), None)
            if representative is not None:
                selected.append(representative)
        names = list(dict.fromkeys(selected))
    assert set(names) <= frames.keys(), 'Unknown image in --images'
    indexed = {int(debug[name]['idx']): frame for name, frame in frames.items()}
    rows = {}
    for name in names:
        row = dict(debug[name])
        rounded = np.fromstring(row['prior_lon_lat_alt_roll_pitch_yaw'], sep=',')
        if row['pose_source'] == 'ortholoc':
            previous = indexed[int(row['idx']) - 1]
            assert not previous['reset_next_from_gt']
            euler = np.asarray(previous['euler_pitch_roll_yaw'])
            pose_data = np.r_[previous['translation_wgs84'], euler[[1, 0, 2]]]
            # TSV rounds to ten decimals; use JSON's exact previous-frame prediction.
            np.testing.assert_allclose(pose_data, rounded, atol=5.1e-11, rtol=0)
        else:
            pose_data = rounded
        row['pose_data'] = pose_data
        rows[name] = row
    print(f'Audit passed: {len(frames)} existing poses. Regenerate {len(names)} crops.', flush=True)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        crops = pool.submit(regenerate_crops, run, rows, output).result()
    funcs = main_functions()
    utils, matcher_type, _ = funcs['load_ortholoc_components']()
    torch.manual_seed(seed)
    matcher = matcher_type(name=run['matcher'], device=run['device'], angles=run['angles'])
    _, K = utils.io.load_camera_params(run['intrinsics_path'])
    paths = []
    for name in names:
        print(f'Freeze one matching pass: {name}', flush=True)
        frame = frames[name]
        color, grid, origin = crops[name]
        saved = utils.io.load_image(str(reference / 'crop_dom' / (Path(name).stem + '.png')))
        np.testing.assert_array_equal(color, saved)
        up = ecef_up(*ecef_to_wgs84(*origin)[:2])
        frozen = {}
        original_run_pnp = utils.pose.run_pnp

        def freeze_and_solve(**kwargs):
            frozen.update({k: v.copy() if isinstance(v, np.ndarray) else v
                           for k, v in kwargs.items() if k != 'pnp_stats'})
            result = original_run_pnp(**kwargs)
            frozen['captured_pose_c2w'] = result[1]
            return result

        with patch.object(utils.pose, 'run_pnp', side_effect=freeze_and_solve):
            funcs['run_ortholoc_localization_on_crop'](
                frame['image_path'], color, grid, K, matcher, run['angles'],
                gravity_camera_up=priors[name], gravity_world_up=up,
                gravity_threshold_deg=run['gravity_threshold_deg'], pnp_seed=seed)
        assert len(frozen['pts2D']) <= 10000, 'Capture must include the exact points after PnP subsampling'
        camera_K = utils.pose.opencv_to_colmap_intrinsics(frozen['K'])
        camera = dict(model='PINHOLE', width=frozen['img_size'][0], height=frozen['img_size'][1],
                      params=[camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]])
        options = dict(max_reproj_error=frozen['reprojectionError'], max_iterations=10000,
                       success_prob=0.9999, seed=seed)
        path = output / (Path(name).stem + '.npz')
        with path.open('xb') as file:
            np.savez_compressed(file, points2D=frozen['pts2D'] + 0.5, points3D=frozen['pts3D'],
                                camera_up=priors[name], world_up=up, ecef_origin=origin,
                                captured_pose_c2w=frozen['captured_pose_c2w'],
                                metadata=json.dumps(dict(image=name, camera=camera, options=options,
                                                         threshold=run['gravity_threshold_deg'])))
        paths.append(path)
    return paths, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--reference_run', type=Path, help='Existing gravity run; regenerate selected crops safely')
    source.add_argument('--inputs', type=Path, help='Replay previously frozen .npz files without matching')
    parser.add_argument('--output', required=True, type=Path, help='NEW directory; existing directories are refused')
    parser.add_argument('--images', nargs='+', help='Exact basenames; default: first, largest tilt, LO/BA rollback')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    reference = args.reference_run.resolve() if args.reference_run else None
    inputs = args.inputs.resolve() if args.inputs else None
    output.mkdir(parents=True, exist_ok=False)
    # imcui imports truncate cwd/log.txt: keep all such side effects inside NEW output.
    os.chdir(output)
    if reference:
        paths, audit = capture(reference, output, args.images, args.seed)
    else:
        paths, audit = sorted(inputs.glob('*.npz')), None
    assert paths, 'No frozen PnP inputs'
    report = dict(existing_run_audit=audit, frames=[])
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            controls = replay(data['points2D'], data['points3D'], meta['camera'], meta['options'],
                              data['camera_up'], data['world_up'], meta['threshold'])
            expected = np.linalg.inv(data['captured_pose_c2w'])[:3]
            np.testing.assert_allclose(controls['gravity_strict']['pose_w2c'], expected, atol=1e-8, rtol=1e-10)
        report['frames'].append(dict(image=meta['image'], frozen_input=str(path), controls=controls))
        print(f"PASS {meta['image']}: original == 180°, strict gravity and inlier checks passed", flush=True)
    report['checks_passed'] = True
    with (output / 'report.json').open('x') as file:
        json.dump(report, file, indent=2)
    print(f'Report: {output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()

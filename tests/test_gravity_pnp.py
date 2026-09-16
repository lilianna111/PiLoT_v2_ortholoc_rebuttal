"""Run with the OrthoLoC environment: python -m unittest discover -s tests -v."""

import sys
import ast
import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'OrthoLoC' / 'OrthoLoC'))
from ortholoc.gravity import camera_up_from_roll_pitch, ecef_up, load_gravity_priors


def synthetic_scene(seed=13, outliers=0, noise=0.0, count=100):
    rng = np.random.default_rng(seed)
    R = Rotation.from_euler('xyz', [17, -12, 30], degrees=True).as_matrix()
    t = np.array([0.2, -0.4, 1.2])
    cam_points = rng.uniform([-2, -1.5, 5], [2, 1.5, 15], (count, 3))
    points3D = (cam_points - t) @ R
    points2D = cam_points[:, :2] / cam_points[:, 2:] * [900, 880] + [320, 240]
    points2D += rng.normal(0, noise, points2D.shape)
    if outliers:
        points2D[-outliers:] = rng.uniform([0, 0], [640, 480], (outliers, 2))
    camera = dict(model='PINHOLE', width=640, height=480, params=[900, 880, 320, 240])
    return points2D, points3D, camera, R, t


class GravityCoordinatesTest(unittest.TestCase):
    def test_camera_convention_and_yaw_invariance(self):
        for roll, pitch in [(0, 49), (10, -35), (-23, 90), (70, 179)]:
            prior = camera_up_from_roll_pitch(roll, pitch)
            for yaw in [-175, 0, 28, 130]:
                R = Rotation.from_euler('xyz', [pitch - 180, roll, yaw], degrees=True).as_matrix()
                np.testing.assert_allclose(prior, R.T @ [0, 0, 1], atol=1e-14)

    def test_ecef_vertical_is_not_ecef_z(self):
        np.testing.assert_allclose(ecef_up(0, 0), [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(ecef_up(0, 90), [0, 0, 1], atol=1e-14)
        up = ecef_up(113, 28)
        self.assertAlmostEqual(np.linalg.norm(up), 1)
        self.assertGreater(np.linalg.norm(up - [0, 0, 1]), 0.5)

    def read_prior(self, text, prior_format='roll_pitch'):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'priors.txt'
            file.write_text(text, encoding='utf-8')
            return load_gravity_priors(str(file), prior_format)

    def test_named_roll_pitch(self):
        priors = self.read_prior('# header\nfolder/2_0.png 10 40\n1_0.png -2 32 # comment\n')
        np.testing.assert_allclose(priors['2_0.png'], camera_up_from_roll_pitch(10, 40))
        self.assertEqual(set(priors), {'1_0.png', '2_0.png'})

    def test_signed_camera_up(self):
        priors = self.read_prior('a.jpg 0 -2 0\n', 'camera_up')
        np.testing.assert_array_equal(priors['a.jpg'], [0, -1, 0])

    def test_bad_priors_rejected(self):
        for text, fmt in [('a.jpg 1 2\na.jpg 1 2', 'roll_pitch'),
                          ('a.jpg nan 2', 'roll_pitch'), ('# empty', 'roll_pitch'),
                          ('a.jpg 0 0 0', 'camera_up'),
                          ('a.jpg 1 2 3 4 5 6', 'roll_pitch')]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read_prior(text, fmt)


class NativeGravityPnPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ortholoc.gravity import require_gravity_backend
        cls.backend = require_gravity_backend()
        cls.options = dict(max_reproj_error=2, max_iterations=500, min_iterations=100,
                           success_prob=0.9999, seed=7)

    def solve(self, points2D, points3D, camera, camera_up, world_up, threshold):
        return self.backend.estimate_absolute_pose(points2D, points3D, camera, self.options,
                                                   camera_up, world_up, threshold)

    def test_unrestricted_gate_matches_original_poselib(self):
        import poselib
        for outliers, noise in [(0, 0), (20, 0.2)]:
            with self.subTest(outliers=outliers):
                x, X, cam, R, _ = synthetic_scene(outliers=outliers, noise=noise)
                pose, _ = poselib.estimate_absolute_pose(x, X, cam, self.options, {})
                up = ecef_up(113, 28)
                rt, info = self.solve(x, X, cam, R @ up, up, 180)
                self.assertTrue(info['success'])
                self.assertEqual(info['rejected_hypotheses'], 0)
                np.testing.assert_allclose(rt, pose.Rt, atol=1e-9, rtol=1e-9)

    def test_ecef_pose_with_noise_and_outliers(self):
        x, X, cam, R, t = synthetic_scene(outliers=20, noise=0.2)
        world_up = ecef_up(113, 28)
        rt, info = self.solve(x, X, cam, R @ world_up, world_up, 2)
        self.assertTrue(info['success'])
        self.assertLessEqual(info['gravity_error_deg'], 2)
        self.assertGreater(info['rejected_hypotheses'], 0)
        self.assertGreaterEqual(info['num_inliers'], 78)
        self.assertLess(Rotation.from_matrix(rt[:, :3] @ R.T).magnitude(), np.deg2rad(0.1))
        np.testing.assert_allclose(rt[:, 3], t, atol=0.02)

    def test_all_hypotheses_rejected_is_failure(self):
        x, X, cam, R, _ = synthetic_scene()
        up = ecef_up(113, 28)
        rt, info = self.solve(x, X, cam, -R @ up, up, 1e-8)
        self.assertIsNone(rt)
        self.assertFalse(info['success'])
        self.assertEqual(info['rejected_hypotheses'], info['generated_hypotheses'])
        self.assertEqual(info['num_inliers'], 0)

    def test_identity_placeholder_cannot_become_a_solution(self):
        rng = np.random.default_rng(2)
        X = rng.uniform([-2, -1, 5], [2, 1, 15], (100, 3))
        R = Rotation.from_euler('x', 0.1, degrees=True).as_matrix()
        pc = X @ R.T
        x = pc[:, :2] / pc[:, 2:] * [900, 880] + [320, 240]
        cam = synthetic_scene()[2]
        rt, info = self.solve(x, X, cam, [0, 0, 1], [0, 0, 1], 1e-8)
        self.assertEqual(info['generated_hypotheses'], info['rejected_hypotheses'])
        self.assertIsNone(rt)
        self.assertFalse(info['success'])

    def test_refinement_cannot_drift_outside_the_hard_gate(self):
        x, X, cam, R, _ = synthetic_scene(noise=1.0)
        up = ecef_up(113, 28)
        camera_up = R @ up
        axis = np.cross(camera_up, [1, 0, 0])
        axis /= np.linalg.norm(axis)
        prior = Rotation.from_rotvec(axis * np.deg2rad(0.4)).apply(camera_up)
        options = dict(self.options, max_reproj_error=3)
        rt, info = self.backend.estimate_absolute_pose(x, X, cam, options, prior, up, 0.2)
        self.assertTrue(info['success'])
        self.assertIsNotNone(rt)
        self.assertLessEqual(info['gravity_error_deg'], 0.2)
        self.assertGreater(info['rejected_refinements'], 0)
        self.assertFalse(info['final_refinement_accepted'])

    def test_small_input_and_invalid_prior(self):
        x, X, cam, R, _ = synthetic_scene()
        rt, info = self.solve(x[:2], X[:2], cam, R[:, 2], [0, 0, 1], 2)
        self.assertIsNone(rt)
        self.assertFalse(info['success'])
        for vector in ([0, 0, 0], [np.nan, 1, 0]):
            with self.assertRaises(ValueError):
                self.solve(x, X, cam, vector, [0, 0, 1], 2)
        with self.assertRaises(ValueError):
            self.solve(x, X, cam, R[:, 2], [0, 0, 1], 0)


class PnPIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # imcui's import truncates cwd/log.txt; isolate that pre-existing side effect.
        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = os.getcwd()
            try:
                os.chdir(directory)
                from ortholoc.utils import pose
            finally:
                os.chdir(previous_cwd)
        cls.pose = pose

    def test_original_path_and_gravity_path(self):
        x, X, camera, R, t = synthetic_scene(outliers=20, noise=0.2)
        K = np.array([[900, 0, 319.5], [0, 880, 239.5], [0, 0, 1]])
        up = ecef_up(113, 28)
        for gravity in [False, True]:
            with self.subTest(gravity=gravity):
                stats = {}
                kwargs = dict(gravity_camera_up=R @ up, gravity_world_up=up) if gravity else {}
                success, c2w, mask = self.pose.run_pnp(x - 0.5, X, K, mode='poselib',
                                                      reprojectionError=2, ransac_seed=7,
                                                      pnp_stats=stats, **kwargs)
                self.assertTrue(success)
                np.testing.assert_allclose(c2w[:3, :3], R.T, atol=0.002)
                np.testing.assert_allclose(c2w[:3, 3], -R.T @ t, atol=0.02)
                self.assertEqual(mask.shape, (len(x),))
                self.assertEqual(stats['prior_mode'], 'gravity' if gravity else 'none')

    def test_subsampled_mask_maps_back_to_all_correspondences(self):
        x = np.zeros((10050, 2))
        X = np.zeros((10050, 3))
        fake_pose = type('Pose', (), {'Rt': np.c_[np.eye(3), np.zeros(3)]})()
        selected = np.random.default_rng(3).choice(len(x), 10000, replace=False)
        with patch.object(self.pose.poselib, 'estimate_absolute_pose', return_value=(
                fake_pose, {'inliers': np.ones(10000, dtype=bool)})):
            success, _, mask = self.pose.run_pnp(x, X, np.eye(3), mode='poselib', ransac_seed=3)
        self.assertTrue(success)
        expected = np.zeros(len(x), dtype=bool)
        expected[selected] = True
        np.testing.assert_array_equal(mask, expected)

    def test_wrong_backend_or_incomplete_prior_fails_explicitly(self):
        x, X, _, _, _ = synthetic_scene()
        for kwargs in [dict(gravity_camera_up=[0, 0, 1]),
                       dict(mode='cv2', gravity_camera_up=[0, 0, 1], gravity_world_up=[0, 0, 1])]:
            with self.assertRaises(ValueError):
                self.pose.run_pnp(x, X, np.eye(3), **kwargs)

    def test_correspondence_filtering_sampling_and_prior_forwarding(self):
        from ortholoc.correspondences import Correspondences2D3D
        x, X, _, R, t = synthetic_scene(count=120)
        x[0] = np.nan
        X[1] = np.nan
        correspondences = Correspondences2D3D(pts0=x - 0.5, pts1=X, is_normalized=False,
                                             confidences=np.ones(len(x)))
        K = np.array([[900, 0, 319.5], [0, 880, 239.5], [0, 0, 1]])
        up = ecef_up(113, 28)
        for gravity in [False, True]:
            kwargs = dict(gravity_camera_up=R @ up, gravity_world_up=up) if gravity else {}
            stats = {}
            with patch.object(self.pose, 'run_pnp', wraps=self.pose.run_pnp) as called:
                success, c2w, _, mask, _ = correspondences.calibrate(
                    num_points=70, width=640, height=480, intrinsics_matrix=K,
                    pnp_mode='poselib', ransac_seed=3, pnp_stats=stats, **kwargs)
            self.assertTrue(success)
            self.assertEqual(mask.shape, (len(x),))
            self.assertFalse(mask[:2].any())
            self.assertEqual(mask.sum(), 70)
            np.testing.assert_allclose(c2w[:, 3], -R.T @ t, atol=1e-5)
            self.assertEqual(called.call_args.kwargs['ransac_seed'], 3)
            if gravity:
                np.testing.assert_allclose(called.call_args.kwargs['gravity_world_up'], up)

    def test_main_localization_forwards_the_prior_without_reimporting_main(self):
        # Load only the function to avoid heavyweight unrelated PixLoc imports/startup.
        source = ast.parse((REPO / 'main.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and
                    n.name == 'run_ortholoc_localization_on_crop')
        from types import SimpleNamespace
        from unittest.mock import Mock
        import os
        import logging
        from ortholoc.correspondences import Correspondences2D2D

        rng = np.random.default_rng(5)
        cam3D = rng.uniform([-1, -1, 5], [1, 1, 10], (100, 3))
        K = np.array([[100, 0, 31.5], [0, 100, 31.5], [0, 0, 1]])
        x = cam3D[:, :2] / cam3D[:, 2:] * 100 + 31.5
        qnorm = x / 63 * 2 - 1
        uv = np.stack(np.unravel_index(np.arange(100), (64, 64)), axis=1)[:, ::-1]
        grid = np.zeros((64, 64, 3))
        grid[uv[:, 1], uv[:, 0]] = cam3D
        refnorm = uv / 63 * 2 - 1
        correspondences = Correspondences2D2D(pts0=qnorm, pts1=refnorm, is_normalized=True,
                                             confidences=np.ones(100))
        matcher = Mock(angles=[0])
        matcher.run.return_value = [correspondences]
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_utils = SimpleNamespace(io=SimpleNamespace(load_image=lambda _: image), pose=self.pose)
        namespace = dict(np=np, os=os, logging=logging,
                         load_ortholoc_components=lambda: (fake_utils, None, None))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(REPO / 'main.py'), 'exec'), namespace)
        fn = namespace['run_ortholoc_localization_on_crop']
        ret = fn('test.jpg', image, grid, K, matcher, [0], gravity_camera_up=np.array([0, 0, 1]),
                 gravity_world_up=np.array([0, 0, 1]), pnp_seed=3)
        self.assertEqual(ret['pnp_stats']['prior_mode'], 'gravity')
        self.assertLessEqual(ret['pnp_stats']['gravity_error_deg'], 2)


class LauncherConfigTest(unittest.TestCase):
    def launch_without_localization(self, extra_args=(), overrides=None):
        # Run both launchers with a fake Python command and isolated pose fixture.
        # Never import main.py or enter its output-directory cleanup.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'DJI_20250612193930_0012_V'
            (root / (name + '.txt')).write_text('0_0.png 113 28 200 0 37 42\n')
            for script in ('run_google_roma.sh', 'run_feicuiwan.sh'):
                text = (REPO / script).read_text().replace(
                    '/media/amax/AE0E2AFD0E2ABE69/datasets/poses/', str(root) + '/')
                (root / script).write_text(text)
            env = {k: v for k, v in os.environ.items() if not k.startswith('ORTHOLOC_')}
            env.update(CONDA_PREFIX=sys.prefix, LD_LIBRARY_PATH=env.get('LD_LIBRARY_PATH', ''))
            env.update(overrides or {})
            result = subprocess.run([
                'bash', '-c',
                'python() { printf "__ARGS_START__\\n"; printf "%s\\n" "$@"; '
                'printf "__ARGS_END__\\n"; }; export -f python; exec bash "$@"',
                'test-launcher', str(root / 'run_google_roma.sh'), *extra_args,
            ], env=env, check=True, capture_output=True, text=True)
            argv = result.stdout.split('__ARGS_START__\n', 1)[1].split('__ARGS_END__', 1)[0].splitlines()
        source = ast.parse((REPO / 'main.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
        namespace = dict(argparse=argparse, os=os, __file__=str(REPO / 'main.py'))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(REPO / 'main.py'), 'exec'), namespace)
        with patch.object(sys, 'argv', argv):
            return namespace['parse_args']()

    def test_sequence_roll_pitch_and_rebuttal_output_defaults(self):
        args = self.launch_without_localization()
        self.assertEqual(args.ortholoc_matcher, 'RoMa')
        self.assertEqual(args.ortholoc_pnp_prior, 'gravity')
        self.assertEqual(args.ortholoc_gravity_prior_format, 'roll_pitch')
        self.assertEqual(args.ortholoc_gravity_prior_file,
                         '/media/amax/AE0E2AFD0E2ABE69/datasets/angle/' + args.name + '.txt')
        self.assertEqual(args.ortholoc_output_root, '/media/amax/PS2000/rebuttal/ortholoc')

    def test_baseline_and_explicit_path_overrides(self):
        args = self.launch_without_localization(
            ['--ortholoc_pnp_prior', 'none', '--ortholoc_gravity_prior_file', '/cli/angles.txt',
             '--ortholoc_output_root', '/cli/results'],
            dict(ORTHOLOC_GRAVITY_PRIOR_FILE='/env/angles.txt'))
        self.assertEqual(args.ortholoc_pnp_prior, 'none')
        self.assertEqual(args.ortholoc_gravity_prior_file, '/cli/angles.txt')
        self.assertEqual(args.ortholoc_output_root, '/cli/results')
        args = self.launch_without_localization(overrides=dict(ORTHOLOC_GRAVITY_PRIOR_FILE='/env/angles.txt'))
        self.assertEqual(args.ortholoc_gravity_prior_file, '/env/angles.txt')


if __name__ == '__main__':
    unittest.main()

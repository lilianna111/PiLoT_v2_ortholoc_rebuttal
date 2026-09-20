"""CPU regressions for robust hypothesis scoring; no new optimizer or matcher."""

import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import poselib
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "OrthoLoC/OrthoLoC"))
from ortholoc.gravity import require_gravity_backend
import test_gravity_pnp as gravity_tests
from test_gravity_pnp import synthetic_scene
from test_depth_pnp import depth_scene


def rho(residual, scale):
    return 2 * np.log1p(0.5 * (residual / scale) ** 2)


def visual_score(rt, x, X, camera, options):
    fx, fy, cx, cy = camera["params"]
    p = X @ rt[:, :3].T + rt[:, 3]
    r = p[:, :2] / p[:, 2:] - (x - [cx, cy]) / [fx, fy]
    squared = np.sum(r ** 2, axis=1)
    tau2 = (options["max_reproj_error"] / ((fx + fy) / 2)) ** 2
    mask = (squared < tau2) & (p[:, 2] > 0)
    return (squared[mask].sum() + (~mask).sum() * tau2) / (len(x) * tau2), mask


class NativeSoftPnPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = require_gravity_backend()
        cls.options = dict(max_reproj_error=2, max_iterations=500, min_iterations=150, seed=7)

    def solve(self, x, X, camera, **kwargs):
        return self.backend.estimate_absolute_pose(
            x, X, camera, self.options, kwargs.pop("camera_up", [0, 0, 1]),
            kwargs.pop("world_up", [0, 0, 1]), 2,
            use_gravity=kwargs.pop("use_gravity", False), prior_fusion="soft", **kwargs)

    def solve_balanced(self, x, X, camera, **kwargs):
        return self.backend.estimate_absolute_pose(
            x, X, camera, self.options, kwargs.pop("camera_up", [0, 0, 1]),
            kwargs.pop("world_up", [0, 0, 1]), 2,
            use_gravity=kwargs.pop("use_gravity", False), prior_fusion="balanced", **kwargs)

    def check_score(self, rt, info, x, X, camera, gravity=True, depth=None):
        self.assertTrue(info["success"])
        visual, mask = visual_score(rt, x, X, camera, self.options)
        np.testing.assert_array_equal(mask, info["inliers"])
        self.assertEqual(int(mask.sum()), info["num_inliers"])
        gc = info["gravity_weight"] * rho(info["gravity_error_deg"], info["gravity_scale_deg"]) if gravity else 0.
        dc = info["depth_weight"] * rho(depth, info["depth_scale_m"]) if depth is not None else 0.
        np.testing.assert_allclose(info["visual_score_normalized"], visual, atol=1e-10)
        np.testing.assert_allclose(info["gravity_prior_cost"], gc, atol=1e-10)
        np.testing.assert_allclose(info["depth_prior_cost"], dc, atol=1e-10)
        np.testing.assert_allclose(info["joint_score"], visual + gc + dc, atol=1e-10)
        self.assertEqual(info["model_score"], info["joint_score"])
        before, after = info["final_refinement_score_before"], info["final_refinement_score_after"]
        if before is not None:
            self.assertEqual(info["final_refinement_accepted"], after <= before)
            np.testing.assert_allclose(info["joint_score"], min(before, after), atol=1e-10)

    def test_large_gravity_and_depth_residuals_are_not_gates(self):
        x, X, camera, R, _ = synthetic_scene()
        up = np.array([0., 0., 1.])
        rt, info = self.solve(x, X, camera, use_gravity=True, camera_up=-(R @ up),
                              world_up=up, depth_evaluator=lambda _: -1000.)
        self.assertGreater(info["gravity_error_deg"], 10)
        self.assertGreater(abs(info["depth_residual_m"]), 10)
        self.assertEqual(info["gravity_rejected_hypotheses"], 0)
        self.assertEqual(info["depth_rejected_hypotheses"], 0)
        self.assertEqual(info["rejected_hypotheses"], info["nonfinite_hypotheses"])
        self.assertFalse(info["enforce_refinement_gravity"])
        self.check_score(rt, info, x, X, camera, depth=-1000.)

    def test_finite_depth_has_no_ten_metre_discontinuity(self):
        x, X, camera, _, _ = synthetic_scene()
        costs = []
        for residual in (9.99, 10., 10.01, 1000., 1e300):
            rt, info = self.solve(x, X, camera, depth_evaluator=lambda _, r=residual: r)
            self.assertTrue(info["success"])
            self.assertEqual(info["depth_rejected_hypotheses"], 0)
            self.assertTrue(np.isfinite(info["joint_score"]))
            costs.append(info["depth_prior_cost"])
            if residual < 1e300:
                self.check_score(rt, info, x, X, camera, gravity=False, depth=residual)
        self.assertTrue(np.all(np.diff(costs) > 0))
        self.assertLess(costs[2] - costs[0], .001)

    def test_zero_weights_match_original_and_do_not_evaluate_depth(self):
        for outliers, noise in ((0, 0), (20, .2), (60, .2)):
            with self.subTest(outliers=outliers):
                x, X, camera, R, _ = synthetic_scene(outliers=outliers, noise=noise)
                original, _ = poselib.estimate_absolute_pose(x, X, camera, self.options, {})
                forbidden = Mock(side_effect=AssertionError("Zero-weight depth must not affect solving"))
                rt, info = self.solve(x, X, camera, use_gravity=True, camera_up=-(R @ [0., 0., 1.]),
                                      depth_evaluator=forbidden, gravity_weight=0., depth_weight=0.)
                forbidden.assert_not_called()
                self.assertTrue(info["success"])
                self.assertFalse(info["soft_priors_active"])
                np.testing.assert_allclose(rt, original.Rt, atol=1e-9, rtol=1e-9)
                self.assertIsNone(info["joint_score"])

    def test_zero_depth_weight_keeps_gravity_scoring(self):
        x, X, camera, R, _ = synthetic_scene()
        forbidden = Mock(side_effect=AssertionError("Disabled depth must not be evaluated"))
        rt, info = self.solve(x, X, camera, use_gravity=True, camera_up=R @ [0., 0., 1.],
                              depth_evaluator=forbidden, depth_weight=0.)
        forbidden.assert_not_called()
        self.check_score(rt, info, x, X, camera)

    def test_invalid_dsm_is_not_a_zero_cost_escape(self):
        x, X, camera, _, _ = synthetic_scene()
        for invalid in (float("nan"), float("inf")):
            rt, info = self.solve(x, X, camera, depth_evaluator=lambda _, v=invalid: v)
            self.assertIsNone(rt)
            self.assertFalse(info["success"])
            self.assertEqual(info["generated_hypotheses"], info["rejected_hypotheses"])
            self.assertGreater(info["invalid_depth_hypotheses"], 0)
            self.assertEqual(info["invalid_depth_hypotheses"], info["depth_rejected_hypotheses"])

    def test_depth_changes_selection_between_two_visual_models(self):
        x, X, camera, R, t = synthetic_scene()
        alternative = t + [.5, -.2, .4]
        p = X @ R.T + alternative
        x[60:] = (p[:, :2] / p[:, 2:] * [900, 880] + [320, 240])[60:]
        original, _ = poselib.estimate_absolute_pose(x, X, camera, self.options, {})
        # Artificial residual isolates scoring, not the physical depth formula.
        evaluator = lambda rt: 100 * (rt[2, 3] - alternative[2])
        rt, info = self.solve(x, X, camera, depth_evaluator=evaluator, depth_weight=1.)
        self.check_score(rt, info, x, X, camera, gravity=False, depth=evaluator(rt))
        self.assertLess(abs(evaluator(rt)), abs(evaluator(original.Rt)) / 2)
        self.assertGreater(np.max(np.abs(rt - original.Rt)), .01)

    def test_gravity_changes_selection_between_two_visual_models(self):
        x, X, camera, R, t = synthetic_scene()
        alternative = Rotation.from_euler('xyz', [4, -3, 2], degrees=True).as_matrix() @ R
        p = X @ alternative.T + t
        x[60:] = (p[:, :2] / p[:, 2:] * [900, 880] + [320, 240])[60:]
        up = np.array([0., 0., 1.])
        rt, info = self.solve(x, X, camera, use_gravity=True, camera_up=alternative @ up,
                              gravity_scale_deg=1., gravity_weight=1.)
        self.check_score(rt, info, x, X, camera)
        self.assertLess(info["gravity_error_deg"], 1.)
        self.assertGreater(Rotation.from_matrix(rt[:, :3] @ R.T).magnitude(), np.deg2rad(2))

    def test_physical_depth_and_direct_ecef_score_verification(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("verify_soft_depth", REPO / "scripts/verify_depth_pnp.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        x, X, camera, expected, prior, _, _ = depth_scene(noise=.2, outliers=20)
        up = np.array([0., 0., 1.])
        camera_up = expected[:, :3] @ up
        rt, info = self.solve(x, X, camera, use_gravity=True, camera_up=camera_up,
                              depth_evaluator=prior)
        info['gravity_threshold_deg'] = 2.
        self.check_score(rt, info, x, X, camera, depth=prior(rt))
        verifier.verify_returned_pose(rt, info, x, X, camera, self.options, prior, camera_up, up, True)

    def test_bad_scales_and_weights_are_rejected(self):
        x, X, camera, _, _ = synthetic_scene()
        for kwargs in (dict(gravity_scale_deg=0), dict(depth_scale_m=-1), dict(depth_scale_m=float('nan')),
                       dict(gravity_weight=-1), dict(depth_weight=float('inf'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.solve(x, X, camera, **kwargs)

    def test_balanced_keeps_visual_ransac_selection_when_depth_is_inconsistent(self):
        x, X, camera, _, _ = synthetic_scene(outliers=20, noise=.2)
        original, original_info = poselib.estimate_absolute_pose(x, X, camera, self.options, {})
        rt, info = self.solve_balanced(x, X, camera, depth_evaluator=lambda _: 1000.,
                                       gravity_weight=0., depth_weight=.1)
        self.assertTrue(info["success"])
        self.assertFalse(info["prior_depth_used"])
        self.assertEqual(info["prior_depth_skip_reason"], "inconsistent_with_visual_pose")
        self.assertEqual(info["num_inliers"], original_info["num_inliers"])
        np.testing.assert_allclose(rt, original.Rt, atol=1e-9, rtol=1e-9)

    def test_balanced_zero_weights_exactly_matches_visual_poselib(self):
        x, X, camera, R, _ = synthetic_scene(outliers=20, noise=.2)
        original, original_info = poselib.estimate_absolute_pose(x, X, camera, self.options, {})
        rt, info = self.solve_balanced(
            x, X, camera, use_gravity=True, camera_up=R @ [0., 0., 1.],
            depth_evaluator=Mock(side_effect=AssertionError("disabled depth must not be evaluated")),
            gravity_weight=0., depth_weight=0.)
        self.assertTrue(info["success"])
        self.assertFalse(info["prior_gravity_used"])
        self.assertFalse(info["prior_depth_used"])
        self.assertEqual(info["num_inliers"], original_info["num_inliers"])
        np.testing.assert_allclose(rt, original.Rt, atol=1e-9, rtol=1e-9)


class SoftIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gravity_tests.PnPIntegrationTest.setUpClass.__func__(cls)

    def test_correspondences_forward_soft_options_and_keep_input_K(self):
        from ortholoc.correspondences import Correspondences2D3D
        x, X, _, R, _ = synthetic_scene(outliers=20, noise=.2)
        K = np.array([[900., 0, 319.5], [0, 880., 239.5], [0, 0, 1.]])
        before = K.copy()
        matches = Correspondences2D3D(pts0=x - .5, pts1=X, confidences=np.ones(len(x)), is_normalized=False)
        stats = {}
        success, c2w, _, mask, _ = matches.calibrate(
            None, 640, 480, K, gravity_camera_up=R @ [0., 0., 1.], gravity_world_up=np.array([0., 0., 1.]),
            depth_prior=lambda _: 1000., prior_fusion='soft', gravity_scale_deg=12., depth_scale_m=15.,
            gravity_weight=.03, depth_weight=.04, pnp_stats=stats, ransac_seed=7)
        self.assertTrue(success)
        self.assertEqual(stats['prior_fusion'], 'soft')
        self.assertEqual(stats['gravity_scale_deg'], 12.)
        self.assertEqual(stats['depth_scale_m'], 15.)
        self.assertEqual(stats['gravity_weight'], .03)
        self.assertEqual(stats['depth_weight'], .04)
        self.assertEqual(mask.shape, (len(x),))
        self.assertEqual(c2w.dtype, np.float64)
        np.testing.assert_array_equal(K, before)
        json.dumps(stats, allow_nan=False)

    def test_main_forwards_soft_settings(self):
        import logging
        import os
        source = ast.parse((REPO / 'main.py').read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'run_ortholoc_localization_on_crop')
        image = np.zeros((64, 64, 3), np.uint8)
        matches = Mock(is_valid=True)
        matches.__len__ = Mock(return_value=100)
        matches.take_min_conf.return_value = matches
        matches.take_covisible.return_value = matches
        correspondences = matches.to_2d3d.return_value
        correspondences.calibrate.return_value = (True, np.c_[np.eye(3), np.zeros(3)], np.eye(3), np.ones(100, bool), np.zeros(100))
        matcher = Mock(angles=[0])
        matcher.run.return_value = [matches]
        utils = SimpleNamespace(io=SimpleNamespace(load_image=lambda _: image), pose=self.pose)
        namespace = dict(np=np, os=os, logging=logging, load_ortholoc_components=lambda: (utils, None, None))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'main_soft_forward', 'exec'), namespace)
        namespace[node.name]('test.png', image, np.zeros((64, 64, 3)), np.eye(3), matcher, [0],
                             prior_fusion='soft', gravity_scale_deg=11., depth_scale_m=12.,
                             gravity_weight=.02, depth_weight=.03)
        for key, value in dict(prior_fusion='soft', gravity_scale_deg=11., depth_scale_m=12.,
                               gravity_weight=.02, depth_weight=.03).items():
            self.assertEqual(correspondences.calibrate.call_args.kwargs[key], value)

    def test_zero_weights_preserve_full_calibration_pose_and_mask(self):
        from ortholoc.correspondences import Correspondences2D3D
        x, X, _, R, _ = synthetic_scene(outliers=20, noise=.2)
        K = np.array([[900., 0, 319.5], [0, 880., 239.5], [0, 0, 1.]])
        matches = Correspondences2D3D(pts0=x - .5, pts1=X, confidences=np.ones(len(x)), is_normalized=False)
        original = matches.calibrate(None, 640, 480, K, ransac_seed=7)
        prior = Mock(side_effect=AssertionError('Disabled depth must not be queried'))
        prior.details = Mock(side_effect=AssertionError('Disabled depth must not be queried for logging'))
        scored = matches.calibrate(None, 640, 480, K, ransac_seed=7, prior_fusion='soft',
                                   gravity_camera_up=-(R @ [0., 0., 1.]), gravity_world_up=np.array([0., 0., 1.]),
                                   depth_prior=prior, gravity_weight=0., depth_weight=0.)
        prior.assert_not_called()
        prior.details.assert_not_called()
        self.assertEqual(original[0], scored[0])
        for before, after in zip(original[1:], scored[1:]):
            np.testing.assert_array_equal(before, after)


class SoftLauncherTest(unittest.TestCase):
    launch_without_localization = gravity_tests.LauncherConfigTest.launch_without_localization

    def test_google_defaults_to_visual_first_balanced_with_explicit_scales_and_weights(self):
        args = self.launch_without_localization(['--ortholoc_pnp_prior', 'gravity_depth'])
        self.assertEqual(args.ortholoc_prior_fusion, 'balanced')
        self.assertEqual(args.ortholoc_gravity_scale_deg, 10.)
        self.assertEqual(args.ortholoc_depth_scale_m, 10.)
        self.assertEqual(args.ortholoc_gravity_weight, .1)
        self.assertEqual(args.ortholoc_depth_weight, .1)

    def test_hard_and_soft_cli_overrides(self):
        args = self.launch_without_localization(['--ortholoc_prior_fusion', 'hard'])
        self.assertEqual(args.ortholoc_prior_fusion, 'hard')
        args = self.launch_without_localization(['--ortholoc_depth_scale_m', '20', '--ortholoc_depth_weight', '.05'])
        self.assertEqual(args.ortholoc_depth_scale_m, 20.)
        self.assertEqual(args.ortholoc_depth_weight, .05)

    def test_output_directories_separate_soft_and_legacy_results(self):
        import os
        source = ast.parse((REPO / 'main.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DualProcessTask')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'matcher_dir' for t in n.targets))
        end = next(i for i in range(start, len(init.body)) if isinstance(init.body[i], ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'output_folder' for t in init.body[i].targets))
        for mode, fusion, expected in [('gravity_depth', 'soft', 'RoMa_gravity_depth_soft'),
                                       ('gravity_depth', 'hard', 'RoMa_gravity_depth'), ('none', 'soft', 'RoMa')]:
            owner = SimpleNamespace(ortholoc_matcher='RoMa', ortholoc_pnp_prior=mode, ortholoc_prior_fusion=fusion)
            namespace = dict(self=owner, args=SimpleNamespace(ortholoc_output_root='/test/results'), os=os)
            # Only string construction, never instantiate main or enter cleanup.
            exec(compile(ast.Module(body=init.body[start:end + 1], type_ignores=[]), 'output_folder', 'exec'), namespace)
            self.assertEqual(namespace['output_folder'], '/test/results/' + expected)

    def test_invalid_soft_parameters_are_checked_before_cleanup(self):
        source = ast.parse((REPO / 'main.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DualProcessTask')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        stop = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Attribute) and t.attr == 'task_q' for t in n.targets))
        prefix = ast.FunctionDef(name='validate', args=init.args, body=init.body[:stop], decorator_list=[])
        namespace = dict(np=np)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])), 'startup_prefix', 'exec'), namespace)
        for args in (SimpleNamespace(ortholoc_depth_scale_m=0), SimpleNamespace(ortholoc_gravity_weight=-1)):
            with self.assertRaises(ValueError):
                namespace['validate'](SimpleNamespace(), {}, args=args)


if __name__ == '__main__':
    unittest.main()

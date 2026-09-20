// Absolute-pose normalization and post-RANSAC refinement follow PoseLib 2.0.5.
// Copyright (c) 2021, Viktor Larsson. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "helpers.h"
#include <PoseLib/robust/estimators/absolute_pose.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/lm_impl.h>
#include <PoseLib/robust/ransac_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <algorithm>
#include <cmath>

namespace {
using namespace poselib;

// Keep PoseLib's P3P, visual inliers and visual refinements. Legacy hard/soft
// modes are retained; balanced mode never uses priors to rank P3P hypotheses.
class GravityEstimator : public AbsolutePoseEstimator {
  public:
    GravityEstimator(const RansacOptions &opt, const std::vector<Point2D> &x,
                     const std::vector<Point3D> &X, const Point3D &camera_up,
                     const Point3D &world_up, double threshold_deg, bool enforce_refinement_gravity,
                     const py::object &depth_evaluator, double depth_threshold_m, bool use_gravity,
                     bool soft, bool balanced, double gravity_scale_deg, double depth_scale_m,
                     double gravity_weight, double depth_weight)
        : AbsolutePoseEstimator(opt, x, X), camera_up(camera_up.normalized()),
          world_up(world_up.normalized()), threshold_deg(threshold_deg),
          enforce_refinement_gravity(enforce_refinement_gravity), depth_evaluator(depth_evaluator),
          depth_threshold_m(depth_threshold_m), use_gravity(use_gravity), soft(soft), balanced(balanced),
          gravity_scale_deg(gravity_scale_deg), depth_scale_m(depth_scale_m),
          gravity_weight(gravity_weight), depth_weight(depth_weight),
          visual_normalizer(x.size() * opt.max_reproj_error * opt.max_reproj_error) {}

    bool soft_priors() const {
        return soft && ((use_gravity && gravity_weight > 0) || depth_active());
    }

    bool depth_active() const {
        return !depth_evaluator.is_none() && !balanced && (!soft || depth_weight > 0);
    }

    // costs.py loss_fn1(alpha=0, truncate=1), on squared normalized residuals:
    // rho(s) = 2*log(1+s/2). Evaluate in log space to keep huge finite residuals
    // finite: a large mismatch must not turn back into a hard rejection.
    static double robust_penalty(double residual, double scale) {
        if (residual == 0)
            return 0;
        const double z = std::log(std::abs(residual)) - std::log(scale) - 0.5 * std::log(2.0);
        return z > 0 ? 4 * z + 2 * std::log1p(std::exp(-2 * z))
                     : 2 * std::log1p(std::exp(2 * z));
    }

    double normalized_visual_score(const CameraPose &pose, size_t *inlier_count) const {
        return AbsolutePoseEstimator::score_model(pose, inlier_count) /
               visual_normalizer;
    }

    double gravity_cost(const CameraPose &pose) const {
        return soft && use_gravity && gravity_weight > 0
                   ? gravity_weight * robust_penalty(error_deg(pose), gravity_scale_deg) : 0;
    }

    double depth_cost(const CameraPose &pose) const {
        return soft && depth_active()
                   ? depth_weight * robust_penalty(depth_residual(pose), depth_scale_m) : 0;
    }

    double error_deg(const CameraPose &pose) const {
        const double dot = camera_up.dot(pose.R() * world_up);
        return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::acos(-1.0);
    }

    bool valid(const CameraPose &pose) const {
        return pose.q.allFinite() && pose.t.allFinite() &&
               (soft || balanced || !use_gravity || error_deg(pose) <= threshold_deg);
    }

    double depth_residual(const CameraPose &pose) const {
        // RANSAC runs without the GIL; copied CPU Python geometry needs it.
        py::gil_scoped_acquire acquire;
        return depth_evaluator(pose.Rt()).cast<double>();
    }

    bool depth_valid(const CameraPose &pose) const {
        if (!depth_active())
            return true;
        const double residual = depth_residual(pose);
        return std::isfinite(residual) && (soft || std::abs(residual) <= depth_threshold_m);
    }

    bool acceptable(const CameraPose &pose) const {
        return pose.q.allFinite() && pose.t.allFinite() &&
               (!enforce_refinement_gravity || valid(pose)) && depth_valid(pose);
    }

    void generate_models(std::vector<CameraPose> *models) {
        AbsolutePoseEstimator::generate_models(models);
        generated += models->size();
        const auto end = std::remove_if(models->begin(), models->end(),
                                       [this](const CameraPose &p) {
                                           if (!p.q.allFinite() || !p.t.allFinite()) {
                                               ++nonfinite;
                                               return true;
                                           }
                                           if (!valid(p)) {
                                               ++gravity_rejected;
                                               return true;
                                           }
                                           if (depth_active()) {
                                               const double residual = depth_residual(p);
                                               // Observe only P3P hypotheses that reached the depth gate.
                                               // Do not score, refine or change their acceptance here.
                                               if (std::isfinite(residual) &&
                                                   std::abs(residual) < std::abs(closest_depth_residual)) {
                                                   closest_depth_residual = residual;
                                                   closest_depth_pose = p;
                                               }
                                               if (!std::isfinite(residual)) {
                                                   ++invalid_depth;
                                                   ++depth_rejected;
                                                   return true;
                                               }
                                               if (!soft && std::abs(residual) > depth_threshold_m) {
                                                   ++depth_rejected;
                                                   return true;
                                               }
                                           }
                                           return false;
                                       });
        rejected += std::distance(end, models->end());
        models->erase(end, models->end());
    }

    double score_model(const CameraPose &pose, size_t *inlier_count) const {
        if (generated == rejected || !acceptable(pose)) {
            *inlier_count = 0;
            return std::numeric_limits<double>::max();
        }
        if (!soft_priors())
            return AbsolutePoseEstimator::score_model(pose, inlier_count);
        const double visual = normalized_visual_score(pose, inlier_count);
        // The existing output requires >3 visual inliers. Do not let a small
        // prior penalty promote an unsupported three-point hypothesis over a
        // usable visual model. LO can still refine such a sample into a model
        // with sufficient support; the visual inlier definition is unchanged.
        if (*inlier_count <= 3)
            return std::numeric_limits<double>::max();
        return visual + gravity_cost(pose) + depth_cost(pose);
    }

    void refine_model(CameraPose *pose) const {
        // Do not refine the identity placeholder when all hypotheses were rejected.
        if (generated == rejected || !acceptable(*pose))
            return;
        const CameraPose before = *pose;
        size_t unused;
        const double before_score = soft_priors() ? score_model(before, &unused) : 0;
        AbsolutePoseEstimator::refine_model(pose);
        if (!acceptable(*pose) || (soft_priors() && score_model(*pose, &unused) > before_score)) {
            *pose = before;
            ++rejected_refinements;
        }
    }

    size_t generated = 0, rejected = 0, nonfinite = 0;
    size_t gravity_rejected = 0, depth_rejected = 0, invalid_depth = 0;
    mutable size_t rejected_refinements = 0;
    double closest_depth_residual = std::numeric_limits<double>::infinity();
    CameraPose closest_depth_pose;

  private:
    Point3D camera_up, world_up;
    double threshold_deg;
    bool enforce_refinement_gravity;
    py::object depth_evaluator;
    double depth_threshold_m;
    bool use_gravity;
    bool soft;
    bool balanced;
    double gravity_scale_deg, depth_scale_m, gravity_weight, depth_weight;
    double visual_normalizer;
};

// Only the final six-DoF refinement is prior-aware. PoseLib's own visual
// accumulator/LM and camera model are reused, so the visual term is unchanged.
class BalancedRefinement {
  public:
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Visual = CameraJacobianAccumulator<PinholeCameraModel, CauchyLoss>;
    using param_t = CameraPose;
    static constexpr size_t num_params = 6;

    BalancedRefinement(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                       const Camera &camera, double loss_scale, const Point3D &camera_up,
                       const Point3D &world_up, bool use_gravity, const py::object &depth_evaluator,
                       double gravity_scale_deg, double depth_scale_m,
                       double gravity_weight, double depth_weight)
        : loss(loss_scale), visual(x, X, camera, loss), camera_up(camera_up.normalized()),
          world_up(world_up.normalized()), use_gravity(use_gravity && gravity_weight > 0),
          depth_evaluator(depth_evaluator), use_depth(!depth_evaluator.is_none() && depth_weight > 0),
          gravity_sigma(gravity_scale_deg * std::acos(-1.0) / 180.0), depth_sigma(depth_scale_m),
          gravity_target(1.5 * gravity_weight), depth_target(1.5 * depth_weight) {
        if (use_depth)
            depth_skip_reason = "not_applied";
    }

    CameraPose step(Vec6 delta, const CameraPose &pose) const { return visual.step(delta, pose); }

    double residual(const CameraPose &pose) const {
        double cost = visual.residual(pose);
        if (gravity_used)
            cost += gravity_strength * prior_loss(gravity_residual(pose).squaredNorm());
        if (depth_used) {
            const double d = depth_residual(pose);
            if (!std::isfinite(d))
                return std::numeric_limits<double>::infinity();
            cost += depth_strength * prior_loss(std::pow(d / depth_sigma, 2));
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Mat6 &H, Vec6 &g) const {
        const size_t count = visual.accumulate(pose, H, g);
        if (gravity_used) {
            Vec6 prior_g;
            Mat6 prior_H;
            gravity_system(pose, &prior_g, &prior_H);
            g += gravity_strength * prior_g;
            H += gravity_strength * prior_H;
        }
        if (depth_used) {
            Vec6 prior_g;
            Mat6 prior_H;
            if (depth_system(pose, &prior_g, &prior_H)) {
                g += depth_strength * prior_g;
                H += depth_strength * prior_H;
            }
        }
        return count;
    }

    void configure(const CameraPose &initial, const CameraPose &visual_pose) {
        Mat6 H = Mat6::Zero();
        Vec6 gv = Vec6::Zero();
        visual.accumulate(visual_pose, H, gv);
        const double visual_norm = gv.norm();
        if (!std::isfinite(visual_norm) || visual_norm < 1e-12)
            return;

        Vec6 gg = Vec6::Zero(), gd = Vec6::Zero();
        if (use_gravity) {
            const double error_deg = gravity_angle_deg(visual_pose);
            if (error_deg <= 25.0) {
                Mat6 prior_H;
                gravity_system(visual_pose, &gg, &prior_H);
                if (gg.norm() > 1e-12) {
                    const double gate = std::clamp(1.0 - (error_deg - 8.0) / 17.0, 0.2, 1.0);
                    gravity_strength = std::clamp(
                        gravity_target * visual_norm / gg.norm(), 0.05, 20.0) * gate;
                    gravity_used = true;
                }
            }
        }
        if (use_depth) {
            const double check = depth_residual(visual_pose);
            // Large mismatch at the visual solution is an unreliable measurement,
            // not an invitation to replace a well-supported visual hypothesis.
            if (!std::isfinite(check)) {
                depth_skip_reason = "invalid_dsm";
            } else if (std::abs(check) > 20.0) {
                depth_skip_reason = "inconsistent_with_visual_pose";
            } else {
                Mat6 prior_H;
                if (depth_system(visual_pose, &gd, &prior_H) && gd.norm() > 1e-12) {
                    const double gate = std::clamp(1.0 - (std::abs(check) - 4.0) / 16.0, 0.35, 1.0);
                    const double cosine = std::clamp(gv.dot(gd) / (visual_norm * gd.norm()), -1.0, 1.0);
                    const double align = std::clamp((1.0 + cosine) / 2.0, 0.2, 1.0);
                    depth_strength = std::clamp(
                        depth_target * visual_norm / gd.norm(), 0.05, 20.0) * gate * align;
                    depth_used = true;
                    depth_skip_reason = "used";
                } else {
                    depth_skip_reason = "invalid_depth_jacobian";
                }
            }
        }
        // The two priors share one budget; this is a gradient target, not a
        // claim that either prior is 30% of a candidate's MSAC score.
        const double combined = (gravity_strength * gg + depth_strength * gd).norm();
        if (combined > 0.3 * visual_norm) {
            const double factor = 0.3 * visual_norm / combined;
            gravity_strength *= factor;
            depth_strength *= factor;
        }
    }

    bool active() const { return gravity_used || depth_used; }
    bool gravity_used = false;
    bool depth_used = false;
    double gravity_strength = 0;
    double depth_strength = 0;
    std::string depth_skip_reason = "not_requested";

  private:
    static double prior_loss(double squared) { return 2.0 * std::log1p(0.5 * squared); }
    static double prior_weight(double squared) { return 1.0 / (1.0 + 0.5 * squared); }

    Eigen::Vector3d gravity_residual(const CameraPose &pose) const {
        return (pose.R() * world_up - camera_up) / gravity_sigma;
    }

    double gravity_angle_deg(const CameraPose &pose) const {
        const double dot = std::clamp(camera_up.dot(pose.R() * world_up), -1.0, 1.0);
        return std::acos(dot) * 180.0 / std::acos(-1.0);
    }

    void gravity_system(const CameraPose &pose, Vec6 *g, Mat6 *H) const {
        const Eigen::Vector3d r = gravity_residual(pose);
        Eigen::Matrix3d skew;
        skew << 0, -world_up.z(), world_up.y(), world_up.z(), 0, -world_up.x(),
                -world_up.y(), world_up.x(), 0;
        Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
        J.block<3, 3>(0, 0) = -pose.R() * skew / gravity_sigma;
        const double w = prior_weight(r.squaredNorm());
        *g = w * J.transpose() * r;
        *H = w * J.transpose() * J;
    }

    double depth_residual(const CameraPose &pose) const {
        py::gil_scoped_acquire acquire;
        return depth_evaluator(pose.Rt()).cast<double>();
    }

    bool depth_system(const CameraPose &pose, Vec6 *g, Mat6 *H) const {
        const double raw = depth_residual(pose);
        if (!std::isfinite(raw))
            return false;
        Vec6 J = Vec6::Zero();
        for (int k = 3; k < 6; ++k) {
            Vec6 delta = Vec6::Zero();
            delta[k] = 0.2;
            const double plus = depth_residual(step(delta, pose));
            delta[k] = -0.2;
            const double minus = depth_residual(step(delta, pose));
            if (!std::isfinite(plus) || !std::isfinite(minus))
                return false;
            J[k] = (plus - minus) / (0.4 * depth_sigma);
        }
        const double r = raw / depth_sigma;
        const double w = prior_weight(r * r);
        *g = w * J * r;
        *H = w * J * J.transpose();
        return true;
    }

    CauchyLoss loss;
    Visual visual;
    Point3D camera_up, world_up;
    bool use_gravity, use_depth;
    const py::object &depth_evaluator;
    double gravity_sigma, depth_sigma, gravity_target, depth_target;
};

py::tuple estimate(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                   const py::dict &camera_dict, const py::dict &ransac_dict,
                   const Point3D &camera_up, const Point3D &world_up, double threshold_deg,
                   bool enforce_refinement_gravity, const py::object &depth_evaluator,
                   double depth_threshold_m, bool use_gravity, const std::string &prior_fusion,
                   double gravity_scale_deg, double depth_scale_m, double gravity_weight, double depth_weight) {
    if (prior_fusion != "hard" && prior_fusion != "soft" && prior_fusion != "balanced")
        throw py::value_error("prior_fusion must be hard, soft or balanced");
    const bool soft = prior_fusion == "soft";
    const bool balanced = prior_fusion == "balanced";
    if (!std::isfinite(gravity_scale_deg) || gravity_scale_deg <= 0 ||
        !std::isfinite(depth_scale_m) || depth_scale_m <= 0)
        throw py::value_error("Soft prior scales must be finite and positive");
    if (!std::isfinite(gravity_weight) || gravity_weight < 0 ||
        !std::isfinite(depth_weight) || depth_weight < 0)
        throw py::value_error("Soft prior weights must be finite and nonnegative");
    if (points2D.size() != points3D.size())
        throw py::value_error("2D and 3D correspondence counts differ");
    if (!camera_up.allFinite() || !world_up.allFinite() || camera_up.norm() < 1e-12 || world_up.norm() < 1e-12)
        throw py::value_error("camera_up and world_up must be finite nonzero vectors");
    if (!std::isfinite(threshold_deg) || threshold_deg <= 0.0 || threshold_deg > 180.0)
        throw py::value_error("gravity threshold must be in (0, 180] degrees");
    if (!std::isfinite(depth_threshold_m) || depth_threshold_m < 0.0)
        throw py::value_error("depth threshold must be finite and nonnegative");
    if (!depth_evaluator.is_none() && !PyCallable_Check(depth_evaluator.ptr()))
        throw py::value_error("depth_evaluator must be callable or None");
    for (size_t k = 0; k < points2D.size(); ++k)
        if (!points2D[k].allFinite() || !points3D[k].allFinite())
            throw py::value_error("PnP correspondences must be finite");

    const Camera camera = camera_from_dict(camera_dict);
    if (!std::isfinite(camera.focal()) || camera.focal() <= 0.0)
        throw py::value_error("Camera focal length must be positive");
    RansacOptions opt;
    update_ransac_options(ransac_dict, opt);
    if (!std::isfinite(opt.max_reproj_error) || opt.max_reproj_error <= 0)
        throw py::value_error("max_reproj_error must be finite and positive");
    BundleOptions bundle_opt;
    bundle_opt.loss_scale = opt.max_reproj_error * 0.5; // Same default as PoseLib's Python binding.
    RansacOptions scaled_opt = opt;
    scaled_opt.max_reproj_error /= camera.focal();

    std::vector<Point2D> calibrated(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k)
        camera.unproject(points2D[k], &calibrated[k]);
    GravityEstimator estimator(scaled_opt, calibrated, points3D, camera_up, world_up, threshold_deg,
                               enforce_refinement_gravity, depth_evaluator, depth_threshold_m, use_gravity,
                               soft, balanced, gravity_scale_deg, depth_scale_m, gravity_weight, depth_weight);
    CameraPose pose;
    pose.q << 1.0, 0.0, 0.0, 0.0;
    pose.t.setZero();
    RansacStats stats;
    std::vector<char> inliers(points2D.size(), false);
    bool final_refinement_accepted = false;
    bool prior_refinement_accepted = false;
    bool prior_gravity_used = false;
    bool prior_depth_used = false;
    double prior_gravity_strength = 0;
    double prior_depth_strength = 0;
    std::string prior_depth_skip_reason = "not_requested";
    double visual_score_before_prior = std::numeric_limits<double>::quiet_NaN();
    double visual_score_after_prior = std::numeric_limits<double>::quiet_NaN();
    size_t visual_inliers_before_prior = 0;
    size_t visual_inliers_after_prior = 0;
    double final_score_before = std::numeric_limits<double>::quiet_NaN();
    double final_score_after = std::numeric_limits<double>::quiet_NaN();
    {
        py::gil_scoped_release release;
        stats = ransac(estimator, scaled_opt, &pose);
        if (estimator.generated > estimator.rejected && stats.num_inliers > 3 && estimator.acceptable(pose)) {
            const double sq_threshold = scaled_opt.max_reproj_error * scaled_opt.max_reproj_error;
            get_inliers(pose, calibrated, points3D, sq_threshold, &inliers);
            const double scale = 1.0 / camera.focal();
            Camera norm_camera = camera;
            norm_camera.rescale(scale);
            bundle_opt.loss_scale *= scale;
            std::vector<Point2D> inlier2D;
            std::vector<Point3D> inlier3D;
            for (size_t k = 0; k < points2D.size(); ++k) {
                if (inliers[k]) {
                    inlier2D.push_back(points2D[k] * scale);
                    inlier3D.push_back(points3D[k]);
                }
            }
            CameraPose refined = pose;
            bundle_adjust(inlier2D, inlier3D, norm_camera, &refined, bundle_opt);
            size_t unused;
            if (estimator.soft_priors()) {
                final_score_before = estimator.score_model(pose, &unused);
                final_score_after = estimator.score_model(refined, &unused);
            }
            if (estimator.acceptable(refined) &&
                (!estimator.soft_priors() || final_score_after <= final_score_before)) {
                pose = refined;
                final_refinement_accepted = true;
            }
            if (balanced) {
                CameraPose visual_pose = pose;
                visual_score_before_prior = estimator.normalized_visual_score(visual_pose, &visual_inliers_before_prior);
                if (camera.model_id != PinholeCameraModel::model_id)
                    throw py::value_error("Balanced final refinement requires a PINHOLE camera");
                BalancedRefinement prior_refiner(inlier2D, inlier3D, norm_camera, bundle_opt.loss_scale,
                                                camera_up, world_up, use_gravity, depth_evaluator,
                                                gravity_scale_deg, depth_scale_m, gravity_weight, depth_weight);
                prior_refiner.configure(visual_pose, visual_pose);
                prior_gravity_used = prior_refiner.gravity_used;
                prior_depth_used = prior_refiner.depth_used;
                prior_gravity_strength = prior_refiner.gravity_strength;
                prior_depth_strength = prior_refiner.depth_strength;
                prior_depth_skip_reason = prior_refiner.depth_skip_reason;
                if (prior_refiner.active()) {
                    CameraPose candidate = visual_pose;
                    BundleOptions prior_opt = bundle_opt;
                    prior_opt.max_iterations = 30;
                    lm_impl(prior_refiner, &candidate, prior_opt);
                    if (candidate.q.allFinite() && candidate.t.allFinite()) {
                        visual_score_after_prior = estimator.normalized_visual_score(candidate, &visual_inliers_after_prior);
                        // Compare against the visual-only BA result on *all* input
                        // correspondences, never against the prior-optimized cost.
                        const size_t minimum_inliers = (9 * visual_inliers_before_prior + 9) / 10;
                        if (visual_inliers_after_prior >= minimum_inliers &&
                            visual_score_after_prior <= 1.05 * visual_score_before_prior) {
                            pose = candidate;
                            prior_refinement_accepted = true;
                        }
                    }
                }
            }
            // Report inliers of the returned pose, not the pre-BA pose.
            get_inliers(pose, calibrated, points3D, sq_threshold, &inliers);
        }
    }
    const size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);
    const bool success = estimator.generated > estimator.rejected && stats.num_inliers > 3 &&
                         num_inliers > 3 && estimator.acceptable(pose);
    size_t unused;
    if (success && estimator.soft_priors())
        stats.model_score = estimator.score_model(pose, &unused);
    py::dict meta;
    write_to_dict(stats, meta);
    meta["success"] = success;
    meta["prior_fusion"] = prior_fusion;
    meta["gravity_scale_deg"] = gravity_scale_deg;
    meta["depth_scale_m"] = depth_scale_m;
    meta["gravity_weight"] = gravity_weight;
    meta["depth_weight"] = depth_weight;
    meta["soft_priors_active"] = estimator.soft_priors();
    meta["balanced_refinement_active"] = balanced;
    meta["prior_gravity_used"] = prior_gravity_used;
    meta["prior_depth_used"] = prior_depth_used;
    meta["prior_gravity_strength"] = prior_gravity_strength;
    meta["prior_depth_strength"] = prior_depth_strength;
    meta["prior_depth_skip_reason"] = prior_depth_skip_reason;
    meta["prior_refinement_accepted"] = prior_refinement_accepted;
    meta["visual_score_before_prior"] = std::isfinite(visual_score_before_prior)
        ? py::cast(visual_score_before_prior) : py::none();
    meta["visual_score_after_prior"] = std::isfinite(visual_score_after_prior)
        ? py::cast(visual_score_after_prior) : py::none();
    meta["visual_inliers_before_prior"] = visual_inliers_before_prior;
    meta["visual_inliers_after_prior"] = visual_inliers_after_prior;
    meta["joint_score"] = success && estimator.soft_priors() ? py::cast(stats.model_score) : py::none();
    meta["final_refinement_score_before"] = std::isfinite(final_score_before)
        ? py::cast(final_score_before) : py::none();
    meta["final_refinement_score_after"] = std::isfinite(final_score_after)
        ? py::cast(final_score_after) : py::none();
    meta["visual_score_normalized"] = success
        ? py::cast(estimator.normalized_visual_score(pose, &unused)) : py::none();
    meta["gravity_prior_cost"] = success ? py::cast(estimator.gravity_cost(pose)) : py::none();
    meta["depth_prior_cost"] = success ? py::cast(estimator.depth_cost(pose)) : py::none();
    meta["num_inliers"] = num_inliers;
    meta["inlier_ratio"] = points2D.empty() ? 0.0 : double(num_inliers) / points2D.size();
    meta["inliers"] = convert_inlier_vector(inliers);
    meta["generated_hypotheses"] = estimator.generated;
    meta["rejected_hypotheses"] = estimator.rejected;
    meta["nonfinite_hypotheses"] = estimator.nonfinite;
    meta["gravity_rejected_hypotheses"] = estimator.gravity_rejected;
    meta["depth_rejected_hypotheses"] = estimator.depth_rejected;
    meta["invalid_depth_hypotheses"] = estimator.invalid_depth;
    meta["rejected_refinements"] = estimator.rejected_refinements;
    meta["final_refinement_accepted"] = final_refinement_accepted;
    meta["enforce_refinement_gravity"] = !soft && enforce_refinement_gravity;
    meta["gravity_enabled"] = use_gravity;
    meta["depth_enabled"] = !depth_evaluator.is_none();
    meta["depth_threshold_m"] = depth_threshold_m;
    meta["gravity_error_deg"] = success && use_gravity ? py::cast(estimator.error_deg(pose)) : py::none();
    const double final_depth = success && (estimator.depth_active() ||
                                           (balanced && !depth_evaluator.is_none() && depth_weight > 0))
                                   ? estimator.depth_residual(pose) : std::numeric_limits<double>::quiet_NaN();
    meta["depth_residual_m"] = std::isfinite(final_depth) ? py::cast(final_depth) : py::none();
    meta["closest_depth_candidate"] = py::none();
    if (std::isfinite(estimator.closest_depth_residual)) {
        py::dict closest;
        closest["stage"] = "P3P_before_MSAC";
        closest["pose_w2c_local_ecef"] = estimator.closest_depth_pose.Rt();
        closest["depth_residual_m"] = estimator.closest_depth_residual;
        closest["gravity_error_deg"] = use_gravity
            ? py::cast(estimator.error_deg(estimator.closest_depth_pose)) : py::none();
        closest["depth_passed"] = soft || std::abs(estimator.closest_depth_residual) <= depth_threshold_m;
        meta["closest_depth_candidate"] = closest;
    }
    return py::make_tuple(success ? py::cast(pose.Rt()) : py::none(), meta);
}
} // namespace

PYBIND11_MODULE(_gravity_pnp, m) {
    m.doc() = "PoseLib 2.0.5 P3P LO-RANSAC with legacy gates/scores or visual-first balanced BA.";
    m.attr("poselib_version") = "2.0.5";
    m.attr("supports_depth") = true;
    m.attr("supports_depth_diagnostics") = true;
    m.attr("supports_soft_priors") = true;
    m.attr("supports_balanced_priors") = true;
    m.def("estimate_absolute_pose", &estimate, py::arg("points2D"), py::arg("points3D"),
          py::arg("camera"), py::arg("ransac_options"), py::arg("camera_up"),
          py::arg("world_up"), py::arg("threshold_deg"), py::arg("enforce_refinement_gravity") = true,
          py::arg("depth_evaluator") = py::none(), py::arg("depth_threshold_m") = 10.0,
          py::arg("use_gravity") = true, py::arg("prior_fusion") = "hard",
          py::arg("gravity_scale_deg") = 10.0, py::arg("depth_scale_m") = 10.0,
          py::arg("gravity_weight") = 0.1, py::arg("depth_weight") = 0.1);
}

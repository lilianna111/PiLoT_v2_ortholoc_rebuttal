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
#include <PoseLib/robust/ransac_impl.h>
#include <algorithm>
#include <cmath>

namespace {
using namespace poselib;

// Reuse PoseLib's P3P sampler, MSAC score and LO-RANSAC, changing only validity.
class GravityEstimator : public AbsolutePoseEstimator {
  public:
    GravityEstimator(const RansacOptions &opt, const std::vector<Point2D> &x,
                     const std::vector<Point3D> &X, const Point3D &camera_up,
                     const Point3D &world_up, double threshold_deg, bool enforce_refinement_gravity)
        : AbsolutePoseEstimator(opt, x, X), camera_up(camera_up.normalized()),
          world_up(world_up.normalized()), threshold_deg(threshold_deg),
          enforce_refinement_gravity(enforce_refinement_gravity) {}

    double error_deg(const CameraPose &pose) const {
        const double dot = camera_up.dot(pose.R() * world_up);
        return std::acos(std::clamp(dot, -1.0, 1.0)) * 180.0 / std::acos(-1.0);
    }

    bool valid(const CameraPose &pose) const {
        return pose.q.allFinite() && pose.t.allFinite() && error_deg(pose) <= threshold_deg;
    }

    bool acceptable(const CameraPose &pose) const {
        return pose.q.allFinite() && pose.t.allFinite() &&
               (!enforce_refinement_gravity || valid(pose));
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
                                           return !valid(p);
                                       });
        rejected += std::distance(end, models->end());
        models->erase(end, models->end());
    }

    double score_model(const CameraPose &pose, size_t *inlier_count) const {
        if (generated == rejected || !acceptable(pose)) {
            *inlier_count = 0;
            return std::numeric_limits<double>::max();
        }
        return AbsolutePoseEstimator::score_model(pose, inlier_count);
    }

    void refine_model(CameraPose *pose) const {
        // Do not refine the identity placeholder when all hypotheses were rejected.
        if (generated == rejected || !acceptable(*pose))
            return;
        const CameraPose before = *pose;
        AbsolutePoseEstimator::refine_model(pose);
        if (!acceptable(*pose)) {
            *pose = before;
            ++rejected_refinements;
        }
    }

    size_t generated = 0, rejected = 0, nonfinite = 0;
    mutable size_t rejected_refinements = 0;

  private:
    Point3D camera_up, world_up;
    double threshold_deg;
    bool enforce_refinement_gravity;
};

py::tuple estimate(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                   const py::dict &camera_dict, const py::dict &ransac_dict,
                   const Point3D &camera_up, const Point3D &world_up, double threshold_deg,
                   bool enforce_refinement_gravity) {
    if (points2D.size() != points3D.size())
        throw py::value_error("2D and 3D correspondence counts differ");
    if (!camera_up.allFinite() || !world_up.allFinite() || camera_up.norm() < 1e-12 || world_up.norm() < 1e-12)
        throw py::value_error("camera_up and world_up must be finite nonzero vectors");
    if (!std::isfinite(threshold_deg) || threshold_deg <= 0.0 || threshold_deg > 180.0)
        throw py::value_error("gravity threshold must be in (0, 180] degrees");
    for (size_t k = 0; k < points2D.size(); ++k)
        if (!points2D[k].allFinite() || !points3D[k].allFinite())
            throw py::value_error("PnP correspondences must be finite");

    const Camera camera = camera_from_dict(camera_dict);
    if (!std::isfinite(camera.focal()) || camera.focal() <= 0.0)
        throw py::value_error("Camera focal length must be positive");
    RansacOptions opt;
    update_ransac_options(ransac_dict, opt);
    BundleOptions bundle_opt;
    bundle_opt.loss_scale = opt.max_reproj_error * 0.5; // Same default as PoseLib's Python binding.
    RansacOptions scaled_opt = opt;
    scaled_opt.max_reproj_error /= camera.focal();

    std::vector<Point2D> calibrated(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k)
        camera.unproject(points2D[k], &calibrated[k]);
    GravityEstimator estimator(scaled_opt, calibrated, points3D, camera_up, world_up, threshold_deg,
                               enforce_refinement_gravity);
    CameraPose pose;
    pose.q << 1.0, 0.0, 0.0, 0.0;
    pose.t.setZero();
    RansacStats stats;
    std::vector<char> inliers(points2D.size(), false);
    bool final_refinement_accepted = false;
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
            if (estimator.acceptable(refined)) {
                pose = refined;
                final_refinement_accepted = true;
            }
            // Report inliers of the returned pose, not the pre-BA pose.
            get_inliers(pose, calibrated, points3D, sq_threshold, &inliers);
        }
    }
    const size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);
    const bool success = estimator.generated > estimator.rejected && stats.num_inliers > 3 &&
                         num_inliers > 3 && estimator.acceptable(pose);
    py::dict meta;
    write_to_dict(stats, meta);
    meta["success"] = success;
    meta["num_inliers"] = num_inliers;
    meta["inlier_ratio"] = points2D.empty() ? 0.0 : double(num_inliers) / points2D.size();
    meta["inliers"] = convert_inlier_vector(inliers);
    meta["generated_hypotheses"] = estimator.generated;
    meta["rejected_hypotheses"] = estimator.rejected;
    meta["nonfinite_hypotheses"] = estimator.nonfinite;
    meta["rejected_refinements"] = estimator.rejected_refinements;
    meta["final_refinement_accepted"] = final_refinement_accepted;
    meta["enforce_refinement_gravity"] = enforce_refinement_gravity;
    meta["gravity_error_deg"] = success ? py::cast(estimator.error_deg(pose)) : py::none();
    return py::make_tuple(success ? py::cast(pose.Rt()) : py::none(), meta);
}
} // namespace

PYBIND11_MODULE(_gravity_pnp, m) {
    m.doc() = "Gravity-gated PoseLib 2.0.5 P3P LO-RANSAC (world-to-camera output).";
    m.attr("poselib_version") = "2.0.5";
    m.def("estimate_absolute_pose", &estimate, py::arg("points2D"), py::arg("points3D"),
          py::arg("camera"), py::arg("ransac_options"), py::arg("camera_up"),
          py::arg("world_up"), py::arg("threshold_deg"), py::arg("enforce_refinement_gravity") = true);
}

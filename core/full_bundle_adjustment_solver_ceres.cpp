#include "full_bundle_adjustment_solver_ceres.h"

namespace analytic_solver {

FullBundleAdjustmentSolverCeres::FullBundleAdjustmentSolverCeres() {}

FullBundleAdjustmentSolverCeres::~FullBundleAdjustmentSolverCeres() {}

void FullBundleAdjustmentSolverCeres::Reset() {}

void FullBundleAdjustmentSolverCeres::AddCamera(const int camera_index,
                                                const Camera &camera) {
  (void)camera;
}

void FullBundleAdjustmentSolverCeres::AddPose(_BA_Pose *original_pose) {}

void FullBundleAdjustmentSolverCeres::AddPoint(_BA_Point *original_point) {}

void FullBundleAdjustmentSolverCeres::AddObservation(const int camera_index,
                                                     _BA_Pose *related_pose,
                                                     _BA_Point *related_point,
                                                     const _BA_Pixel &pixel) {}

void FullBundleAdjustmentSolverCeres::MakePoseFixed(
    _BA_Pose *original_pose_to_be_fixed) {}

void FullBundleAdjustmentSolverCeres::MakePointFixed(
    _BA_Point *original_point_to_be_fixed) {}

bool FullBundleAdjustmentSolverCeres::Solve(Options options,
                                            Summary *summary = nullptr) {}

std::string FullBundleAdjustmentSolverCeres::GetSolverStatistics() const {}

}  // namespace analytic_solver
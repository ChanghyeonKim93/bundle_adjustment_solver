#include "full_bundle_adjustment_solver_analytic.h"

FullBundleAdjustmentSolverAnalytic::FullBundleAdjustmentSolverAnalytic() {}
FullBundleAdjustmentSolverAnalytic::~FullBundleAdjustmentSolverAnalytic() {}

void FullBundleAdjustmentSolverAnalytic::RegisterCamera(const CameraId camera_id, const Camera &camera) {}
void FullBundleAdjustmentSolverAnalytic::RegisterPose(Pose *world_to_camera_link_pose) {}
void FullBundleAdjustmentSolverAnalytic::RegisterPosition(Position *world_position) {}
void FullBundleAdjustmentSolverAnalytic::FixRegisteredPose(const Pose *world_to_camera_link_pose) {}
void FullBundleAdjustmentSolverAnalytic::FixRegisteredPosition(const Position *world_position) {}

bool FullBundleAdjustmentSolverAnalytic::Solve(const Options &options, Summary *summary) {}
void FullBundleAdjustmentSolverAnalytic::AddPointObservation() {}

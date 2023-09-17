#ifndef FULL_BUNDLE_ADJUSTMENT_SOLVER_ANALYTIC_H_
#define FULL_BUNDLE_ADJUSTMENT_SOLVER_ANALYTIC_H_

#include "full_bundle_adjustment_solver.h"

class FullBundleAdjustmentSolverAnalytic : public FullBundleAdjustmentSolver {
 public:
  FullBundleAdjustmentSolverAnalytic();
  ~FullBundleAdjustmentSolverAnalytic();

  void RegisterCamera(const CameraId camera_id, const Camera &camera) final;
  void RegisterPose(Pose *world_to_camera_link_pose) final;
  void RegisterPosition(Position *world_position) final;
  void FixRegisteredPose(const Pose *world_to_camera_link_pose) final;
  void FixRegisteredPosition(const Position *world_position) final;

  bool Solve(const Options &options, Summary *summary) final;

 public:
  void AddPointObservation() final;
};

#endif
#ifndef FULL_BUNDLE_ADJUSTMENT_SOLVER_H_
#define FULL_BUNDLE_ADJUSTMENT_SOLVER_H_

#include <iostream>

#include "options_and_summary.h"
#include "types.h"

class FullBundleAdjustmentSolver {
 public:
  ~FullBundleAdjustmentSolver() {}

  virtual void RegisterCamera(const CameraId camera_id, const Camera &camera) = 0;
  virtual void RegisterPose(Pose *world_to_camera_link_pose) = 0;
  virtual void RegisterPosition(Position *world_position) = 0;
  virtual void FixRegisteredPose(const Pose *world_to_camera_link_pose) = 0;
  virtual void FixRegisteredPosition(const Position *world_position) = 0;

  virtual bool Solve(const Options &options, Summary *summary) = 0;

 public:
  virtual void AddPointObservation() = 0;

 private:
};

#endif
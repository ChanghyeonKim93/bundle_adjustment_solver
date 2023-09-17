#ifndef TYPES_H_
#define TYPES_H_

#include "eigen3/Eigen/Dense"

using Pose = Eigen::Isometry3f;
using Position = Eigen::Vector3f;
using CameraId = int32_t;

struct Camera {
  float fx{0.0f};
  float fy{0.0f};
  float cx{0.0f};
  float cy{0.0f};
  Pose camera_to_camera_link_pose{Pose::Identity()};
};

#endif
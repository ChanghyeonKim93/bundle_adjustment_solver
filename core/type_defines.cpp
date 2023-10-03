#include "type_defines.h"

namespace visual_navigation {

Camera::Camera() {}

Camera::~Camera() {}

Camera::Camera(const Camera& camera)
    : camera_intrinsic_parameters_(camera.camera_intrinsic_parameters_),
      body_to_camera_pose_(camera.body_to_camera_pose_),
      camera_to_body_pose_(body_to_camera_pose_.inverse()) {}

Camera& Camera::operator=(const Camera& camera) {
  this->camera_intrinsic_parameters_ = camera.GetCameraIntrinsicParameters();
  this->body_to_camera_pose_ = camera.GetBodyToCameraPose();
  this->camera_to_body_pose_ = camera.GetCameraToBodyPose();
  return *this;
}

const CameraIntrinsicParameters& Camera::GetCameraIntrinsicParameters() const {}

float Camera::GetFx() const { return camera_intrinsic_parameters_.fx; }

float Camera::GetFy() const { return camera_intrinsic_parameters_.fy; }

float Camera::GetCx() const { return camera_intrinsic_parameters_.cx; }

float Camera::GetCy() const { return camera_intrinsic_parameters_.cy; }

float Camera::GetInverseFx() const {
  return 1.0f / camera_intrinsic_parameters_.fx;
}

float Camera::GetInverseFy() const {
  return 1.0f / camera_intrinsic_parameters_.fy;
}

Pose Camera::GetBodyToCameraPose() const { return body_to_camera_pose_; }

Pose Camera::GetCameraToBodyPose() const { return camera_to_body_pose_; }

}  // namespace visual_navigation

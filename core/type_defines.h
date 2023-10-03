#ifndef _TYPE_DEFINES_H_
#define _TYPE_DEFINES_H_

#include <vector>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

namespace visual_navigation {

using Pose = Eigen::Transform<float, 3, 1>;
using Point = Eigen::Matrix<float, 3, 1>;
using Pixel = Eigen::Matrix<float, 2, 1>;

struct CameraIntrinsicParameters {
  float fx{0.0f};
  float fy{0.0f};
  float cx{0.0f};
  float cy{0.0f};
};

/*
 Test& operator=(const Test &t) {
        this->m_i = t.m_i;
        cout << "call the copy assignment operator" << endl;
        return *this;
    }
*/

class Camera {
 public:
  Camera();
  ~Camera();
  Camera(const Camera& camera);
  Camera& operator=(const Camera& camera);

 public:
  const CameraIntrinsicParameters& GetCameraIntrinsicParameters() const;
  float GetFx() const;
  float GetFy() const;
  float GetCx() const;
  float GetCy() const;
  float GetInverseFx() const;
  float GetInverseFy() const;
  Pose GetBodyToCameraPose() const;
  Pose GetCameraToBodyPose() const;

 private:
  CameraIntrinsicParameters camera_intrinsic_parameters_;
  Pose body_to_camera_pose_;
  Pose camera_to_body_pose_;
};

}  // namespace visual_navigation

#endif
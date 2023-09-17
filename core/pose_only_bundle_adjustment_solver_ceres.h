#ifndef _POSE_ONLY_BUNDLE_ADJUSTMENT_SOLVER_CERES_H_
#define _POSE_ONLY_BUNDLE_ADJUSTMENT_SOLVER_CERES_H_

#include <iostream>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

class ReprojectionCostFunctor_3dof_numerical;
class ReprojectionCostFunctor_6dof_numerical;
// class ReprojectionCostFunctor_6dof_analytic;

class ReprojectionCostFunctor_3dof_numerical {
 public:
  ReprojectionCostFunctor_3dof_numerical() = delete;
  // ## READ THIS!
  // When the target camera is changed, you should call "SetCameraIntrinsicParameters()" to update the camera related
  // parameters, such as 'camera_to_baselink_pose (fixed extrinsic pose)' and 'fx,fy,cx,cy (fixed intrinsics)'
  ReprojectionCostFunctor_3dof_numerical(const Eigen::Vector3d &world_position, const Eigen::Vector2d &pixel_matched);

  template <typename T>
  bool operator()(const T *const relative_baselink_parameter, T *residuals) const {
    const Eigen::Matrix3d camera_to_baselink_rotation = pose_camera_to_base_.linear();
    const Eigen::Vector3d camera_to_baselink_translation = pose_camera_to_base_.translation();
    const double &r11 = camera_to_baselink_rotation(0, 0), &r12 = camera_to_baselink_rotation(0, 1),
                 &r13 = camera_to_baselink_rotation(0, 2);
    const double &r21 = camera_to_baselink_rotation(1, 0), &r22 = camera_to_baselink_rotation(1, 1),
                 &r23 = camera_to_baselink_rotation(1, 2);
    const double &r31 = camera_to_baselink_rotation(2, 0), &r32 = camera_to_baselink_rotation(2, 1),
                 &r33 = camera_to_baselink_rotation(2, 2);

    // params: wx wy wz  tx ty rz
    T tx = relative_baselink_parameter[0];
    T ty = relative_baselink_parameter[1];
    T cos_psi = cos(relative_baselink_parameter[2]);
    T sin_psi = sin(relative_baselink_parameter[2]);

    T world_position[3];
    world_position[0] = T(world_position_.x());
    world_position[1] = T(world_position_.y());
    world_position[2] = T(world_position_.z());

    T warped_base_position[3];
    warped_base_position[0] = cos_psi * world_position[0] - sin_psi * world_position[1] + tx;
    warped_base_position[1] = sin_psi * world_position[0] + cos_psi * world_position[1] + ty;
    warped_base_position[2] = world_position[2];

    const auto &xb = warped_base_position[0];
    const auto &yb = warped_base_position[1];
    const auto &zb = warped_base_position[2];

    T local_position[3];
    local_position[0] = r11 * xb + r12 * yb + r13 * zb + camera_to_baselink_translation.x();
    local_position[1] = r21 * xb + r22 * yb + r23 * zb + camera_to_baselink_translation.y();
    local_position[2] = r31 * xb + r32 * yb + r33 * zb + camera_to_baselink_translation.z();

    const T inverse_z = T(1.0) / local_position[2];

    residuals[0] = fx_ * local_position[0] * inverse_z + cx_ - pixel_matched_.x();
    residuals[1] = fy_ * local_position[1] * inverse_z + cy_ - pixel_matched_.y();

    return true;
  };

 private:
  Eigen::Vector3d world_position_;
  Eigen::Vector2d pixel_matched_;

 public:
  static void SetPoseBaseToCamera(const Eigen::Isometry3d &pose_base_to_camera);
  static void SetCameraIntrinsicParameters(const double fx, const double fy, const double cx, const double cy);

  static Eigen::Isometry3d pose_base_to_camera_;
  static Eigen::Isometry3d pose_camera_to_base_;
  static double fx_;
  static double fy_;
  static double cx_;
  static double cy_;
};

class ReprojectionCostFunctor_6dof_numerical {
 public:
  ReprojectionCostFunctor_6dof_numerical() = delete;

  // ## READ THIS!
  // When the target camera is changed, you should call "SetCameraIntrinsicParameters()" to update the camera related
  // parameters, such as 'camera_to_baselink_pose (fixed extrinsic pose)' and 'fx,fy,cx,cy (fixed intrinsics)'
  ReprojectionCostFunctor_6dof_numerical(const Eigen::Vector3d &world_position, const Eigen::Vector2d &pixel_matched);

  template <typename T>
  bool operator()(const T *const params, T *residuals) const {
    // params: wx wy wz  tx ty rz
    T world_position[3];
    world_position[0] = T(world_position_(0));
    world_position[1] = T(world_position_(1));
    world_position[2] = T(world_position_(2));

    T warped_position[3];
    // Rotate first
    ceres::AngleAxisRotatePoint(params, world_position, warped_position);

    // Translate
    warped_position[0] += params[3];
    warped_position[1] += params[4];
    warped_position[2] += params[5];

    const T inverse_z = T(1.0) / warped_position[2];
    residuals[0] = fx_ * warped_position[0] * inverse_z + cx_ - pixel_matched_.x();
    residuals[1] = fy_ * warped_position[1] * inverse_z + cy_ - pixel_matched_.y();

    return true;
  }

 private:
  Eigen::Vector3d world_position_;
  Eigen::Vector2d pixel_matched_;

 public:
  static void SetCameraIntrinsicParameters(const double fx, const double fy, const double cx, const double cy);

  static double fx_;
  static double fy_;
  static double cx_;
  static double cy_;
};

// class ReprojectionCostFunctor_6dof_analytic
//     : public ceres::SizedCostFunction<2, 6>
// {
// public:
//   ReprojectionCostFunctor_6dof_analytic(
//       const Eigen::Vector3d &world_position,
//       const Eigen::Vector2d &pixel_matched);

//   ~ReprojectionCostFunctor_6dof_analytic();

//   bool Evaluate(
//       double const *const *parameters, double *residuals, double **jacobians) const
//   {
//     const double &fx = ReprojectionCostFunctor_6dof_analytic::fx_;
//     const double &fy = ReprojectionCostFunctor_6dof_analytic::fy_;
//     const double &cx = ReprojectionCostFunctor_6dof_analytic::cx_;
//     const double &cy = ReprojectionCostFunctor_6dof_analytic::cy_;

//     Eigen::Matrix<double, 3, 3> R_c2w;
//     geometry::so3Exp(parameters[0][3], parameters[0][4], parameters[0][5], R_c2w);

//     const Eigen::Vector3d rotated_position = R_c2w * world_position_;
//     const Eigen::Vector3d local_position(
//         rotated_position.x() + parameters[0][0],
//         rotated_position.y() + parameters[0][1],
//         rotated_position.z() + parameters[0][2]);

//     const double inverse_z = 1.0f / local_position(2);
//     const double x_inverse_z = local_position(0) * inverse_z;
//     const double y_inverse_z = local_position(1) * inverse_z;
//     const double fx_x_inverse_z = fx_ * x_inverse_z;
//     const double fy_y_inverse_z = fy_ * y_inverse_z;

//     const double projected_u = fx_x_inverse_z + cx_;
//     const double projected_v = fy_y_inverse_z + cy_;

//     // residual for u and v
//     // Fill residual
//     residuals[0] = projected_u - pixel_matched_.x();
//     residuals[1] = projected_v - pixel_matched_.y();

//     const double &xr = rotated_position.x();
//     const double &yr = rotated_position.y();
//     const double &zr = rotated_position.z();
//     if (jacobians != nullptr)
//     {
//       if (jacobians[0] != nullptr)
//       {
//         jacobians[0][0] = fx_ * inverse_z;
//         jacobians[0][1] = 0.0f;
//         jacobians[0][2] = -fx_x_inverse_z * inverse_z;
//         jacobians[0][3] = jacobians[0][2] * yr;
//         jacobians[0][4] = jacobians[0][0] * zr - jacobians[0][2] * xr;
//         jacobians[0][5] = -jacobians[0][0] * yr;

//         jacobians[0][6] = 0.0f;
//         jacobians[0][7] = fy_ * inverse_z;
//         jacobians[0][8] = -fy_y_inverse_z * inverse_z;
//         jacobians[0][9] = -jacobians[0][7] * zr + jacobians[0][8] * yr;
//         jacobians[0][10] = -jacobians[0][8] * xr;
//         jacobians[0][11] = jacobians[0][7] * xr;
//       }
//     }
//     return true;
//   }

// private:
//   Eigen::Vector3d world_position_;
//   Eigen::Vector2d pixel_matched_;

// public:
//   static void SetCameraIntrinsicParameters(
//       const double fx, const double fy, const double cx, const double cy);
//   static double fx_;
//   static double fy_;
//   static double cx_;
//   static double cy_;
// };

#endif
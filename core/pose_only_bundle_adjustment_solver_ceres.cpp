#include "pose_only_bundle_adjustment_solver_ceres.h"

double ReprojectionCostFunctor_3dof_numerical::fx_ = 0;
double ReprojectionCostFunctor_3dof_numerical::fy_ = 0;
double ReprojectionCostFunctor_3dof_numerical::cx_ = 0;
double ReprojectionCostFunctor_3dof_numerical::cy_ = 0;
Eigen::Isometry3d ReprojectionCostFunctor_3dof_numerical::pose_base_to_camera_ = Eigen::Isometry3d::Identity();
Eigen::Isometry3d ReprojectionCostFunctor_3dof_numerical::pose_camera_to_base_ = Eigen::Isometry3d::Identity();
ReprojectionCostFunctor_3dof_numerical::ReprojectionCostFunctor_3dof_numerical(
    const Eigen::Vector3d &world_position,
    const Eigen::Vector2d &pixel_matched)
    : world_position_(world_position), pixel_matched_(pixel_matched) {}
void ReprojectionCostFunctor_3dof_numerical::SetPoseBaseToCamera(
    const Eigen::Isometry3d &pose_base_to_camera)
{
  ReprojectionCostFunctor_3dof_numerical::pose_base_to_camera_ = pose_base_to_camera;
  ReprojectionCostFunctor_3dof_numerical::pose_camera_to_base_ = pose_base_to_camera_.inverse();
}
void ReprojectionCostFunctor_3dof_numerical::SetCameraIntrinsicParameters(
    const double fx, const double fy, const double cx, const double cy)
{
  ReprojectionCostFunctor_3dof_numerical::fx_ = fx;
  ReprojectionCostFunctor_3dof_numerical::fy_ = fy;
  ReprojectionCostFunctor_3dof_numerical::cx_ = cx;
  ReprojectionCostFunctor_3dof_numerical::cy_ = cy;
}

double ReprojectionCostFunctor_6dof_numerical::fx_ = 0;
double ReprojectionCostFunctor_6dof_numerical::fy_ = 0;
double ReprojectionCostFunctor_6dof_numerical::cx_ = 0;
double ReprojectionCostFunctor_6dof_numerical::cy_ = 0;
ReprojectionCostFunctor_6dof_numerical::ReprojectionCostFunctor_6dof_numerical(
    const Eigen::Vector3d &world_position,
    const Eigen::Vector2d &pixel_matched)
    : world_position_(world_position), pixel_matched_(pixel_matched) {}
void ReprojectionCostFunctor_6dof_numerical::SetCameraIntrinsicParameters(
    const double fx, const double fy, const double cx, const double cy)
{
  ReprojectionCostFunctor_6dof_numerical::fx_ = fx;
  ReprojectionCostFunctor_6dof_numerical::fy_ = fy;
  ReprojectionCostFunctor_6dof_numerical::cx_ = cx;
  ReprojectionCostFunctor_6dof_numerical::cy_ = cy;
}

// double ReprojectionCostFunctor_6dof_analytic::fx_ = 0;
// double ReprojectionCostFunctor_6dof_analytic::fy_ = 0;
// double ReprojectionCostFunctor_6dof_analytic::cx_ = 0;
// double ReprojectionCostFunctor_6dof_analytic::cy_ = 0;
// ReprojectionCostFunctor_6dof_analytic::ReprojectionCostFunctor_6dof_analytic(
//     const Eigen::Vector3d &world_position,
//     const Eigen::Vector2d &pixel_matched)
//     : world_position_(world_position), pixel_matched_(pixel_matched) {}
// void ReprojectionCostFunctor_6dof_analytic::SetCameraIntrinsicParameters(
//     const double fx, const double fy, const double cx, const double cy)
// {
//   ReprojectionCostFunctor_6dof_analytic::fx_ = fx;
//   ReprojectionCostFunctor_6dof_analytic::fy_ = fy;
//   ReprojectionCostFunctor_6dof_analytic::cx_ = cx;
//   ReprojectionCostFunctor_6dof_analytic::cy_ = cy;
// }
// ReprojectionCostFunctor_6dof_analytic::~ReprojectionCostFunctor_6dof_analytic() {}

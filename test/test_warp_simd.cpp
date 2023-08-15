#include <iostream>
#include <random>
#include <unordered_map>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/highgui.hpp"

#include "utility/timer.h"
#include "utility/simd_library.h"

#include "core/full_bundle_adjustment_solver.h"

using Numeric = float;
std::vector<Eigen::Matrix<Numeric, 3, 1>> GenerateWorldPosition()
{
  std::vector<Eigen::Matrix<Numeric, 3, 1>> true_world_position_list;
  // Generate 3D points and projections
  const float x_nominal = 8.5f;
  const float z_min = 0.7f;
  const float z_max = 1.7f;
  const float y_min = 0.0f;
  const float y_max = 26.0f;

  const float y_step = 0.001f;
  const float z_step = 0.001f;

  for (float z = z_min; z <= z_max; z += z_step)
  {
    for (float y = y_min; y <= y_max; y += y_step)
    {
      Eigen::Matrix<Numeric, 3, 1> world_position;
      world_position.x() = x_nominal;
      world_position.y() = y;
      world_position.z() = z;
      true_world_position_list.push_back(world_position);
    }
  }
  return true_world_position_list;
}

int main()
{
  simd::PointWarper pw;

  const int num_fixed_poses = 5;
  const float point_error_level = 0.5;

  std::vector<Eigen::Vector3f> X_list = GenerateWorldPosition();
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();

  timer::StopWatch stopwatch("abc");
  stopwatch.Start();
  std::vector<Eigen::Vector3f> warped_X_list;
  warped_X_list.resize(X_list.size());
  for (int idx = 0; idx < X_list.size(); ++idx)
    warped_X_list[idx] = pose * pose * X_list[idx];
  const double time1 = stopwatch.GetLapTimeFromLatest();
  std::cout << time1 << std::endl;

  stopwatch.Start();
  pw.Warp(X_list, pose, warped_X_list);
  const double time2 = stopwatch.GetLapTimeFromLatest();
  std::cout << time2 << std::endl;

  return 0;
}
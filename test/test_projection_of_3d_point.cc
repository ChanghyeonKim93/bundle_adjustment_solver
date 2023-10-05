#include <iostream>
#include <vector>

#include "eigen3/Eigen/Dense"

using Pixel = Eigen::Vector2d;
using Position = Eigen::Vector3d;
using Pose = Eigen::Isometry3d;

int main() {
  std::vector<Pixel> left_pixel_list{{100, 50},  {200, 50},  {300, 50},
                                     {100, 100}, {200, 100}, {300, 100},
                                     {100, 150}, {200, 150}, {300, 150}};
  std::vector<Pixel> right_pixel_list{{90, 50},  {190, 50},  {290, 50},
                                      {75, 100}, {175, 100}, {275, 100},
                                      {50, 150}, {150, 150}, {250, 150}};

  std::vector<Position> left_local_position_list = {
      {-1.0, -0.5, 5.0}, {0.0, -0.5, 5.0}, {1.0, -0.5, 5.0},
      {-0.4, 0.0, 2.0},  {0.0, 0.0, 2.0},  {0.4, 0.0, 2.0},
      {-0.2, 0.1, 1.0},  {0.0, 0.1, 1.0},  {0.2, 0.1, 1.0}};

  const int image_height = 200;
  const int image_width = 400;
  const double fx = 500.0;
  const double fy = 500.0;
  const double cx = 200.0;
  const double cy = 100.0;

  const double baseline = 0.1;

  Pose left_pose = Pose::Identity();
  Pose right_pose = Pose::Identity();
  right_pose.translation().x() += baseline;

  const double baseline_fx = baseline * fx;

  for (int index = 0; index < 9; ++index) {
    const auto& left_pixel = left_pixel_list[index];
    const auto& right_pixel = right_pixel_list[index];

    const double disparity = left_pixel.x() - right_pixel.x();
    const double local_depth = baseline_fx / disparity;

    Position local_position;
    local_position.x() = (left_pixel.x() - cx) / fx * local_depth;
    local_position.y() = (left_pixel.y() - cy) / fy * local_depth;
    local_position.z() = local_depth;
    // left_local_position_list.push_back(std::move(local_position));
  }

  return 0;
}

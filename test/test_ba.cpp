#include <iostream>
#include <random>
#include <unordered_map>

#include "core/full_bundle_adjustment_solver.h"
#include "core/pose_only_bundle_adjustment_solver.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "utility/simd_library.h"
#include "utility/timer.h"

using Numeric = float;
bool in_image(const analytic_solver::_BA_Pixel &pixel, const int n_cols, const int n_rows) {
  return (pixel.x() < n_cols && pixel.x() > 0 && pixel.y() < n_rows && pixel.y() > 0);
};

struct Frame {
  int id;
  Eigen::Transform<Numeric, 3, 1> pose;
  std::vector<int> observed_landmark_id_list;
  std::vector<analytic_solver::_BA_Pixel> observed_pixel_list;
};

struct StereoFrame {
  int id;
  Eigen::Transform<Numeric, 3, 1> pose;

  struct {
    std::vector<int> landmark_id_list;
    std::vector<analytic_solver::_BA_Pixel> pixel_list;
  } left;
  struct {
    std::vector<int> landmark_id_list;
    std::vector<analytic_solver::_BA_Pixel> pixel_list;
  } right;
};

struct Landmark {
  int id;
  Eigen::Matrix<Numeric, 3, 1> world_position;
  std::vector<int> frame_id_list;
};

std::vector<Eigen::Matrix<Numeric, 3, 1>> GenerateWorldPosition() {
  std::vector<Eigen::Matrix<Numeric, 3, 1>> true_world_position_list;
  // Generate 3D points and projections
  const float x_nominal = 8.5f;
  const float z_min = 0.7f;
  const float z_max = 1.7f;
  const float y_min = 0.0f;
  const float y_max = 26.0f;

  const float y_step = 0.1f;
  const float z_step = 0.1f;

  for (float z = z_min; z <= z_max; z += z_step) {
    for (float y = y_min; y <= y_max; y += y_step) {
      Eigen::Matrix<Numeric, 3, 1> world_position;
      world_position.x() = x_nominal;
      world_position.y() = y;
      world_position.z() = z;
      true_world_position_list.push_back(world_position);
    }
  }
  return true_world_position_list;
}

void GetStereoInstrinsicAndExtrinsic(analytic_solver::_BA_Camera &camera_left,
                                     analytic_solver::_BA_Camera &camera_right) {
  Eigen::Transform<Numeric, 3, 1> pose_left_to_right;
  pose_left_to_right.linear() = Eigen::Matrix<Numeric, 3, 3>::Identity();
  pose_left_to_right.translation().setZero();
  pose_left_to_right.translation().x() += 0.12;

  camera_left.fx = 525.0;
  camera_left.fy = 525.0;
  camera_left.cx = 320.0;
  camera_left.cy = 240.0;
  camera_left.pose_cam0_to_this = Eigen::Transform<Numeric, 3, 1>::Identity();
  camera_left.pose_this_to_cam0 = Eigen::Transform<Numeric, 3, 1>::Identity();

  camera_right.fx = 525.0;
  camera_right.fy = 525.0;
  camera_right.cx = 320.0;
  camera_right.cy = 240.0;
  camera_right.pose_cam0_to_this = pose_left_to_right;
  camera_right.pose_this_to_cam0 = pose_left_to_right.inverse();
}

int main() {
  simd::PointWarper pw;

  const int num_fixed_poses = 5;
  const float std_pixel_error = 0.5;
  const float point_error_level = 0.5;
  const float position_error_level = 0.2;
  const int n_cols = 640;
  const int n_rows = 480;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> pixel_err(0, std_pixel_error);
  std::uniform_real_distribution<float> point_err(-point_error_level, point_error_level);
  std::uniform_real_distribution<float> position_err(-position_error_level, position_error_level);

  analytic_solver::_BA_Camera cam_left, cam_right;
  GetStereoInstrinsicAndExtrinsic(cam_left, cam_right);

  std::vector<analytic_solver::_BA_Camera> camera_list;
  camera_list.push_back(cam_left);
  camera_list.push_back(cam_right);

  std::unordered_map<int, StereoFrame> stereoframe_pool;
  std::unordered_map<int, Landmark> landmark_pool;

  // generate world position list
  std::vector<Eigen::Matrix<Numeric, 3, 1>> true_X_list = GenerateWorldPosition();
  std::vector<Eigen::Matrix<Numeric, 3, 1>> X_list = true_X_list;

  // Generate camera pose list
  std::vector<Eigen::Transform<Numeric, 3, 1>> T_wc_list;
  Eigen::Transform<Numeric, 3, 1> pose_base_to_camera = Eigen::Transform<Numeric, 3, 1>::Identity();
  pose_base_to_camera.linear() =
      Eigen::AngleAxis<Numeric>(M_PI_2, Eigen::Matrix<Numeric, 3, 1>::UnitY()).toRotationMatrix() *
      Eigen::AngleAxis<Numeric>(-M_PI_2, Eigen::Matrix<Numeric, 3, 1>::UnitZ()).toRotationMatrix();

  // camera poses
  int frame_count = 0;
  const float x_step = 0.005f;
  const float y_step = 0.2f;
  const float yaw_step = 0.01f;
  Eigen::Transform<Numeric, 3, 1> pose_world_to_base = Eigen::Transform<Numeric, 3, 1>::Identity();
  pose_world_to_base.linear() =
      Eigen::AngleAxis<Numeric>(-0.2, Eigen::Matrix<Numeric, 3, 1>::UnitZ()).toRotationMatrix();
  pose_world_to_base.translation().x() = -1.0;
  pose_world_to_base.translation().y() = -2.5;
  pose_world_to_base.translation().z() = 0.0;
  for (int index = 0; index < 60; ++index) {
    pose_world_to_base.linear() =
        pose_world_to_base.linear() *
        Eigen::AngleAxis<Numeric>(yaw_step, Eigen::Matrix<Numeric, 3, 1>::UnitZ()).toRotationMatrix();
    pose_world_to_base.translation().x() += x_step;
    pose_world_to_base.translation().y() += y_step;
    Eigen::Transform<Numeric, 3, 1> T_wc = pose_world_to_base * pose_base_to_camera;
    T_wc_list.push_back(T_wc);

    StereoFrame stereo_frame;
    stereo_frame.id = frame_count;
    stereo_frame.pose = T_wc;

    stereoframe_pool.insert({frame_count, stereo_frame});
    ++frame_count;
  }

  // Pose error on opt poses
  for (int index = num_fixed_poses; index < 60; ++index) {
    stereoframe_pool.at(index).pose.translation().x() += position_err(gen);
    stereoframe_pool.at(index).pose.translation().y() += position_err(gen);
    stereoframe_pool.at(index).pose.translation().z() += position_err(gen);
  }

  for (int frame_id = 0; frame_id < T_wc_list.size(); ++frame_id) {
    auto &stereo_frame = stereoframe_pool[frame_id];
    const auto T_cw = T_wc_list[frame_id].inverse();

    for (int landmark_id = 0; landmark_id < true_X_list.size(); ++landmark_id) {
      const auto &Xi = true_X_list[landmark_id];

      Landmark landmark;
      landmark.id = landmark_id;
      landmark.world_position = true_X_list[landmark_id];

      const auto local_position = T_cw * Xi;
      const float inverse_z_left = 1.0 / local_position(2);
      analytic_solver::_BA_Pixel pixel_left;
      pixel_left.x() = camera_list[0].fx * local_position(0) * inverse_z_left + camera_list[0].cx + pixel_err(gen);
      pixel_left.y() = camera_list[0].fy * local_position(1) * inverse_z_left + camera_list[0].cy + pixel_err(gen);

      const bool is_seen_left = in_image(pixel_left, n_cols, n_rows);
      if (is_seen_left) {
        stereo_frame.left.landmark_id_list.push_back(landmark_id);
        stereo_frame.left.pixel_list.push_back(pixel_left);
      }

      const auto right_position = camera_list[1].pose_this_to_cam0 * local_position;
      const float inverse_z_right = 1.0 / right_position(2);
      analytic_solver::_BA_Pixel pixel_right;
      pixel_right.x() = camera_list[1].fx * right_position(0) * inverse_z_right + camera_list[1].cx + pixel_err(gen);
      pixel_right.y() = camera_list[1].fy * right_position(1) * inverse_z_right + camera_list[1].cy + pixel_err(gen);

      const bool is_seen_right = in_image(pixel_right, n_cols, n_rows);
      if (is_seen_right) {
        stereo_frame.right.landmark_id_list.push_back(landmark_id);
        stereo_frame.right.pixel_list.push_back(pixel_right);
      }

      landmark.world_position.x() += point_err(gen);
      landmark.world_position.y() += point_err(gen);
      landmark.world_position.z() += point_err(gen);
      landmark_pool[landmark_id] = landmark;
    }
  }

  // Solve problem
  analytic_solver::FullBundleAdjustmentSolver ba_solver;
  for (const auto &camera : camera_list) ba_solver.AddCamera(camera);

  for (auto &[frame_id, stereoframe] : stereoframe_pool) ba_solver.AddPose(&stereoframe.pose);

  for (auto &[landmark_id, landmark] : landmark_pool) ba_solver.AddPoint(&landmark.world_position);

  std::vector<int> fixed_pose_list;
  for (int index = 0; index < num_fixed_poses; ++index) fixed_pose_list.push_back(index);
  for (const auto &frame_id : fixed_pose_list) {
    auto &pose = stereoframe_pool[frame_id].pose;
    ba_solver.MakePoseFixed(&pose);
  }
  ba_solver.MakePointFixed({});

  ba_solver.FinalizeParameters();  // This line is necessary before solving the problem.

  for (auto &[frame_id, stereoframe] : stereoframe_pool) {
    int camera_id = 0;  // left camera
    for (int index = 0; index < stereoframe.left.landmark_id_list.size(); ++index) {
      auto &landmark = landmark_pool[stereoframe.left.landmark_id_list[index]];
      const auto &left_pixel = stereoframe.left.pixel_list[index];

      ba_solver.AddObservation(camera_id, &stereoframe.pose, &landmark.world_position, left_pixel);
    }

    camera_id = 1;  // right camera
    for (int index = 0; index < stereoframe.right.landmark_id_list.size(); ++index) {
      auto &landmark = landmark_pool[stereoframe.right.landmark_id_list[index]];
      const auto &right_pixel = stereoframe.right.pixel_list[index];

      ba_solver.AddObservation(camera_id, &stereoframe.pose, &landmark.world_position, right_pixel);
    }
  }

  ba_solver.GetSolverStatistics();

  // std::cout << "Before: \n";
  // for (size_t index = 0; index < T_wc_list.size(); ++index)
  // {
  //   const auto &true_pose = T_wc_list[index];
  //   const auto &est_pose = stereoframe_pool[index].pose;
  //   std::cout << "true pose:\n"
  //             << true_pose.linear() << "\n"
  //             << true_pose.translation().transpose() << "\n";
  //   std::cout << "est  pose:\n"
  //             << est_pose.linear() << "\n"
  //             << est_pose.translation().transpose() << "\n";
  // }

  analytic_solver::Options options;
  options.iteration_handle.max_num_iterations = 3000;
  // options.trust_region_handle.initial_lambda = 100.0;
  analytic_solver::Summary summary;
  ba_solver.Solve(options, &summary);

  std::cout << summary.BriefReport() << std::endl;

  std::cout << "After: \n";
  // for (size_t index = 0; index < T_wc_list.size(); ++index)
  // {
  //   const auto &true_pose = T_wc_list[index];
  //   const auto &est_pose = stereoframe_pool[index].pose;
  //   std::cout << "true pose:\n"
  //             << true_pose.linear() << "\n"
  //             << true_pose.translation().transpose() << "\n";
  //   std::cout << "est  pose:\n"
  //             << est_pose.linear() << "\n"
  //             << est_pose.translation().transpose() << "\n";
  // }

  // for (size_t index = 0; index < true_X_list.size(); ++index)
  // {
  //   const auto &true_X = true_X_list[index];
  //   const auto &est_X = landmark_pool[index].world_position;
  //   std::cout << "true X:\n"
  //             << true_X.transpose() << "\n";
  //   std::cout << "est  X:\n"
  //             << est_X.transpose() << "\n";
  // }

  return 1;
}
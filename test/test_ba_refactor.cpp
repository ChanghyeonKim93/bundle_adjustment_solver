#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>

#include "core/full_bundle_adjustment_solver.h"
#include "core/full_bundle_adjustment_solver_refactor.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "utility/simd_library.h"
#include "utility/timer.h"

using Numeric = double;
using Pose = Eigen::Transform<Numeric, 3, 1>;
using Point = Eigen::Matrix<Numeric, 3, 1>;

bool IsInImage(const visual_navigation::analytic_solver::_BA_Pixel &pixel,
               const int image_width, const int image_height) {
  return (pixel.x() < image_width && pixel.x() > 0 &&
          pixel.y() < image_height && pixel.y() > 0);
};

struct Frame {
  int id;
  Pose world_to_reference_camera_pose;
  std::vector<int> observed_landmark_id_list;
  std::vector<visual_navigation::analytic_solver::_BA_Pixel>
      observed_pixel_list;
};

struct StereoFrame {
  int id;
  Pose pose;

  struct {
    std::vector<int> landmark_id_list;
    std::vector<visual_navigation::analytic_solver::_BA_Pixel> pixel_list;
  } left;
  struct {
    std::vector<int> landmark_id_list;
    std::vector<visual_navigation::analytic_solver::_BA_Pixel> pixel_list;
  } right;
};

struct Landmark {
  int id;
  Point world_point;
  std::vector<int> frame_id_list;
};

std::vector<Point> GenerateWorldPosition() {
  std::vector<Point> true_world_point_list;

  // Generate 3D points and projections
  const float x_nominal = 8.5f;
  const float z_min = 1.7f;
  const float z_max = 5.7f;
  const float y_min = 0.0f;
  const float y_max = 26.0f;

  const float y_step = 0.4f;
  const float z_step = 0.4f;

  for (float z = z_min; z <= z_max; z += z_step) {
    for (float y = y_min; y <= y_max; y += y_step) {
      Point world_point;
      world_point.x() = x_nominal;
      world_point.y() = y;
      world_point.z() = z;
      true_world_point_list.push_back(std::move(world_point));
    }
  }

  return true_world_point_list;
}

void GetStereoInstrinsicAndExtrinsic(
    visual_navigation::analytic_solver::OptimizerCamera &camera_left,
    visual_navigation::analytic_solver::OptimizerCamera &camera_right) {
  Pose left_to_right_pose;
  left_to_right_pose.linear() = Eigen::Matrix<Numeric, 3, 3>::Identity();
  left_to_right_pose.translation().setZero();
  left_to_right_pose.translation().x() += 0.12;

  camera_left.fx = 525.0f;
  camera_left.fy = 525.0f;
  camera_left.cx = 320.0f;
  camera_left.cy = 240.0f;
  camera_left.camera_to_body_pose = Pose::Identity();

  camera_right.fx = 525.0f;
  camera_right.fy = 525.0f;
  camera_right.cx = 320.0f;
  camera_right.cy = 240.0f;
  camera_right.camera_to_body_pose = left_to_right_pose.inverse();
}

int main() {
  const int num_total_poses = 60;
  const int num_fixed_poses = 5;

  const float pixel_error = 0.0f;
  const float point_error_level = 0.5f;
  const float pose_translation_error_level = 0.1f;
  const int image_width = 640;
  const int image_height = 480;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> pixel_err(0, pixel_error);
  std::uniform_real_distribution<float> point_err(-point_error_level,
                                                  point_error_level);
  std::uniform_real_distribution<float> position_err(
      -pose_translation_error_level, pose_translation_error_level);

  visual_navigation::analytic_solver::OptimizerCamera cam_left, cam_right;
  GetStereoInstrinsicAndExtrinsic(cam_left, cam_right);

  std::vector<visual_navigation::analytic_solver::OptimizerCamera> camera_list;
  camera_list.push_back(cam_left);
  camera_list.push_back(cam_right);

  std::unordered_map<int, StereoFrame> stereoframe_pool;
  std::unordered_map<int, Landmark> landmark_pool;

  // generate world position list
  std::vector<Point> true_world_point_list = GenerateWorldPosition();
  std::vector<Point> world_point_list = true_world_point_list;

  // Generate camera pose list
  std::vector<Pose> world_to_reference_camera_pose_list;
  Pose base_to_camera_pose = Pose::Identity();
  base_to_camera_pose.linear() =
      Eigen::AngleAxis<Numeric>(M_PI_2, Eigen::Matrix<Numeric, 3, 1>::UnitY())
          .toRotationMatrix() *
      Eigen::AngleAxis<Numeric>(-M_PI_2, Eigen::Matrix<Numeric, 3, 1>::UnitZ())
          .toRotationMatrix();

  // camera poses
  int frame_count = 0;
  const float x_step = 0.005f;
  const float y_step = 0.2f;
  const float yaw_step = 0.005f;
  Pose world_to_base_pose = Pose::Identity();
  world_to_base_pose.linear() =
      Eigen::AngleAxis<Numeric>(-0.1, Eigen::Matrix<Numeric, 3, 1>::UnitZ())
          .toRotationMatrix();
  world_to_base_pose.translation().x() = -4.0;
  world_to_base_pose.translation().y() = -2.5;
  world_to_base_pose.translation().z() = 0.0;

  for (int index = 0; index < num_total_poses; ++index) {
    world_to_base_pose.linear() =
        world_to_base_pose.linear() *
        Eigen::AngleAxis<Numeric>(yaw_step,
                                  Eigen::Matrix<Numeric, 3, 1>::UnitZ())
            .toRotationMatrix();
    world_to_base_pose.translation().x() += x_step;
    world_to_base_pose.translation().y() += y_step;
    Pose T_wc = world_to_base_pose * base_to_camera_pose;
    world_to_reference_camera_pose_list.push_back(T_wc);

    StereoFrame stereo_frame;
    stereo_frame.id = frame_count;
    stereo_frame.pose = T_wc;

    stereoframe_pool.insert({frame_count, stereo_frame});
    ++frame_count;
  }

  // Pose error on opt poses
  for (int index = num_fixed_poses; index < num_total_poses; ++index) {
    stereoframe_pool.at(index).pose.translation().x() += position_err(gen);
    stereoframe_pool.at(index).pose.translation().y() += position_err(gen);
    stereoframe_pool.at(index).pose.translation().z() += position_err(gen);
  }

  for (int frame_id = 0; frame_id < world_to_reference_camera_pose_list.size();
       ++frame_id) {
    auto &stereo_frame = stereoframe_pool[frame_id];
    const auto &reference_camera_to_world_pose =
        world_to_reference_camera_pose_list[frame_id].inverse();

    for (int landmark_id = 0; landmark_id < true_world_point_list.size();
         ++landmark_id) {
      const auto &Xi = true_world_point_list[landmark_id];

      Landmark landmark;
      landmark.id = landmark_id;
      landmark.world_point = true_world_point_list[landmark_id];

      const auto local_point = reference_camera_to_world_pose * Xi;
      const float inverse_z_left = 1.0 / local_point(2);
      visual_navigation::analytic_solver::_BA_Pixel pixel_left;
      pixel_left.x() = camera_list[0].fx * local_point(0) * inverse_z_left +
                       camera_list[0].cx + pixel_err(gen);
      pixel_left.y() = camera_list[0].fy * local_point(1) * inverse_z_left +
                       camera_list[0].cy + pixel_err(gen);

      const bool is_seen_left =
          IsInImage(pixel_left, image_width, image_height);
      if (is_seen_left) {
        stereo_frame.left.landmark_id_list.push_back(landmark_id);
        stereo_frame.left.pixel_list.push_back(pixel_left);
      }

      const auto right_position =
          camera_list[1].camera_to_body_pose * local_point;
      const float inverse_z_right = 1.0 / right_position(2);
      visual_navigation::analytic_solver::_BA_Pixel pixel_right;
      pixel_right.x() =
          camera_list[1].fx * right_position(0) * inverse_z_right +
          camera_list[1].cx + pixel_err(gen);
      pixel_right.y() =
          camera_list[1].fy * right_position(1) * inverse_z_right +
          camera_list[1].cy + pixel_err(gen);

      const bool is_seen_right =
          IsInImage(pixel_right, image_width, image_height);
      if (is_seen_right) {
        stereo_frame.right.landmark_id_list.push_back(landmark_id);
        stereo_frame.right.pixel_list.push_back(pixel_right);
      }

      landmark.world_point.x() += point_err(gen);
      landmark.world_point.y() += point_err(gen);
      landmark.world_point.z() += point_err(gen);
      landmark_pool[landmark_id] = landmark;
    }
  }

  // Solve problem
  visual_navigation::analytic_solver::FullBundleAdjustmentSolverRefactor
      ba_solver;
  for (int camera_index = 0; camera_index < camera_list.size(); ++camera_index)
    ba_solver.RegisterCamera(camera_index, camera_list[camera_index]);

  for (auto &[frame_id, stereoframe] : stereoframe_pool)
    ba_solver.RegisterWorldToBodyPose(&(stereoframe.pose));

  for (auto &[landmark_id, landmark] : landmark_pool)
    ba_solver.RegisterWorldPoint(&(landmark.world_point));

  std::vector<int> fixed_pose_list;
  for (int index = 0; index < num_fixed_poses; ++index)
    fixed_pose_list.push_back(index);
  for (const auto &frame_id : fixed_pose_list) {
    auto &pose = stereoframe_pool[frame_id].pose;
    ba_solver.MakePoseFixed(&pose);
  }
  // ba_solver.MakePointFixed({});

  for (auto &[frame_id, stereoframe] : stereoframe_pool) {
    int camera_index = 0;  // left camera
    for (int index = 0; index < stereoframe.left.landmark_id_list.size();
         ++index) {
      auto &landmark = landmark_pool[stereoframe.left.landmark_id_list[index]];
      const auto &left_pixel = stereoframe.left.pixel_list[index];

      ba_solver.AddObservation(camera_index, &stereoframe.pose,
                               &landmark.world_point, left_pixel);
    }

    camera_index = 1;  // right camera
    for (int index = 0; index < stereoframe.right.landmark_id_list.size();
         ++index) {
      auto &landmark = landmark_pool[stereoframe.right.landmark_id_list[index]];
      const auto &right_pixel = stereoframe.right.pixel_list[index];

      ba_solver.AddObservation(camera_index, &stereoframe.pose,
                               &landmark.world_point, right_pixel);
    }
  }

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

  visual_navigation::analytic_solver::Options options;
  options.solver_type =
      visual_navigation::analytic_solver::SolverType::LEVENBERG_MARQUARDT;
  options.iteration_handle.max_num_iterations = 300;
  options.convergence_handle.threshold_cost_change = 1e-6f;
  options.convergence_handle.threshold_step_size = 1e-6f;
  // options.trust_region_handle.initial_lambda = 100.0;
  visual_navigation::analytic_solver::Summary summary;
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
#include <iostream>
#include <random>
#include <vector>

#include "core/hybrid_visual_odometry/pose_optimizer.h"
#include "core/util/timer.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"

int main() {
  try {
    Eigen::Isometry3f pose_left_to_right;
    pose_left_to_right.linear() = Eigen::Matrix3f::Identity();
    pose_left_to_right.translation().x() = 0.05;
    pose_left_to_right.translation().y() = 0.0;
    pose_left_to_right.translation().z() = 0.0;

    Eigen::Isometry3f pose_world_to_current_frame;
    pose_world_to_current_frame.linear() = Eigen::AngleAxisf(-0.12, Eigen::Vector3f::UnitY()).toRotationMatrix();
    pose_world_to_current_frame.translation().x() = 0.4;
    pose_world_to_current_frame.translation().y() = 0.012;
    pose_world_to_current_frame.translation().z() = -0.5;

    // Generate 3D points and projections
    const size_t n_cols = 640;
    const size_t n_rows = 480;
    const float fx = 338.0;
    const float fy = 338.0;
    const float cx = 320.0;
    const float cy = 240.0;

    constexpr size_t num_points = 100000;

    std::random_device rd;
    std::mt19937 gen(rd());

    const float z_default = 1.2f;
    const float z_deviation = 5.0f;
    const float x_deviation = 1.7f;
    const float y_deviation = 1.3f;
    const float pixel_error = 0.0f;
    std::uniform_real_distribution<float> dist_x(-x_deviation, x_deviation);
    std::uniform_real_distribution<float> dist_y(-y_deviation, y_deviation);
    std::uniform_real_distribution<float> dist_z(0, z_deviation);
    std::normal_distribution<float> dist_pixel(0, pixel_error);

    std::vector<Eigen::Vector3f> true_world_position_list;
    std::vector<Eigen::Vector2f> true_left_pixel_list;
    std::vector<Eigen::Vector2f> true_right_pixel_list;
    std::vector<Eigen::Vector3f> world_position_list;
    std::vector<Eigen::Vector2f> left_pixel_list;
    std::vector<Eigen::Vector2f> right_pixel_list;
    for (size_t index = 0; index < num_points; ++index) {
      Eigen::Vector3f world_position;
      world_position.x() = dist_x(gen);
      world_position.y() = dist_y(gen);
      world_position.z() = dist_z(gen) + z_default;
      true_world_position_list.push_back(world_position);
      world_position_list.push_back(world_position);

      const Eigen::Vector3f left_local_position = pose_world_to_current_frame.inverse() * world_position;
      const Eigen::Vector3f right_local_position = pose_left_to_right.inverse() * left_local_position;

      Eigen::Vector2f left_pixel;
      const float inverse_z_left = 1.0 / left_local_position.z();
      left_pixel.x() = fx * left_local_position.x() * inverse_z_left + cx;
      left_pixel.y() = fy * left_local_position.y() * inverse_z_left + cy;
      true_left_pixel_list.push_back(left_pixel);

      Eigen::Vector2f right_pixel;
      const float inverse_z_right = 1.0 / right_local_position.z();
      right_pixel.x() = fx * right_local_position.x() * inverse_z_right + cx;
      right_pixel.y() = fy * right_local_position.y() * inverse_z_right + cy;
      true_right_pixel_list.push_back(left_pixel);

      left_pixel.x() += dist_pixel(gen);
      left_pixel.y() += dist_pixel(gen);
      left_pixel_list.push_back(left_pixel);
      right_pixel.x() += dist_pixel(gen);
      right_pixel.y() += dist_pixel(gen);
      right_pixel_list.push_back(right_pixel);
    }

    // Solve
    Eigen::Isometry3f pose_world_to_current_frame_optimized;
    Eigen::Isometry3f pose_world_to_current_frame_initial_guess;
    pose_world_to_current_frame_initial_guess = Eigen::Isometry3f::Identity();
    pose_world_to_current_frame_initial_guess.translation().x() -= 0.2;
    pose_world_to_current_frame_initial_guess.translation().y() -= 0.5;
    pose_world_to_current_frame_optimized = pose_world_to_current_frame_initial_guess;
    std::unique_ptr<analytic_solver::PoseOptimizer> pose_optimizer = std::make_unique<analytic_solver::PoseOptimizer>();
    analytic_solver::Summary summary;
    analytic_solver::Options options;
    options.iteration_handle.max_num_iterations = 100;
    options.convergence_handle.threshold_cost_change = 1e-6;
    options.convergence_handle.threshold_step_size = 1e-6;
    options.outlier_handle.threshold_huber_loss = 1.5;
    options.outlier_handle.threshold_outlier_rejection = 2.5;
    std::vector<bool> mask_inlier_left;
    std::vector<bool> mask_inlier_right;
    pose_optimizer->SolveStereoPoseOnlyBundleAdjustment6Dof(
        world_position_list, left_pixel_list, right_pixel_list, fx, fy, cx, cy, fx, fy, cx, cy, pose_left_to_right,
        pose_world_to_current_frame_optimized, mask_inlier_left, mask_inlier_right, options, &summary);

    std::cout << summary.BriefReport() << std::endl;

    std::vector<Eigen::Isometry3f> debug_pose_list = pose_optimizer->GetDebugPoses();

    // for (size_t iter = 0; iter < debug_pose_list.size(); ++iter)
    // {
    //   const Eigen::Isometry3f &pose_world_to_current_temp = debug_pose_list[iter];
    //   std::vector<Eigen::Vector2f> projected_pixel_list;
    //   for (size_t index = 0; index < num_points; ++index)
    //   {
    //     const Eigen::Vector3f local_position = pose_world_to_current_temp.inverse() * world_position_list[index];

    //     Eigen::Vector2f pixel;
    //     const float inverse_z = 1.0 / local_position.z();
    //     pixel.x() = fx * local_position.x() * inverse_z + cx;
    //     pixel.y() = fy * local_position.y() * inverse_z + cy;

    //     projected_pixel_list.push_back(pixel);
    //   }

    //   cv::Mat image_blank = cv::Mat::zeros(cv::Size(n_cols, n_rows), CV_8UC3);
    //   for (size_t index = 0; index < pixel_list.size(); ++index)
    //   {
    //     const Eigen::Vector2f &pixel = pixel_list[index];
    //     const Eigen::Vector2f &projected_pixel = projected_pixel_list[index];
    //     cv::circle(image_blank, cv::Point2f(pixel.x(), pixel.y()), 4, cv::Scalar(255, 0, 0), 1);
    //     cv::circle(image_blank, cv::Point2f(projected_pixel.x(), projected_pixel.y()), 2, cv::Scalar(0, 0, 255), 1);
    //   }
    //   cv::imshow("optimization process visualization", image_blank);
    //   cv::waitKey(0);
    // }

    std::cout << "Compare pose:\n";

    Eigen::Matrix<float, 3, 4> pose_true;
    pose_true << pose_world_to_current_frame.linear(), pose_world_to_current_frame.translation();
    std::cout << "truth:\n";
    std::cout << pose_true << std::endl;

    Eigen::Matrix<float, 3, 4> pose_initial_guess;
    pose_initial_guess << pose_world_to_current_frame_initial_guess.linear(),
        pose_world_to_current_frame_initial_guess.translation();
    std::cout << "Initial guess:\n";
    std::cout << pose_initial_guess << std::endl;

    Eigen::Matrix<float, 3, 4> pose_optimized;
    pose_optimized << pose_world_to_current_frame_optimized.linear(),
        pose_world_to_current_frame_optimized.translation();
    std::cout << "Estimated:\n";
    std::cout << pose_optimized << std::endl;
  } catch (std::exception &e) {
    std::cout << "e.what(): " << e.what() << std::endl;
  }

  return 0;
};
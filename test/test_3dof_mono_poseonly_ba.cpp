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
#include "pose_optimizer_ceres.h"

void GeneratePoseOnlyBundleAdjustmentSimulationData(
    const size_t num_points, const float pixel_error,
    const Eigen::Isometry3f &pose_world_to_current, const size_t n_cols,
    const size_t n_rows, const float fx, const float fy, const float cx,
    const float cy, std::vector<Eigen::Vector3f> &true_world_position_list,
    std::vector<Eigen::Vector2f> &true_pixel_list,
    std::vector<Eigen::Vector3f> &world_position_list,
    std::vector<Eigen::Vector2f> &pixel_list) {
  // Generate 3D points and projections
  std::random_device rd;
  std::mt19937 gen(rd());

  const float x_default = 1.2f;
  const float z_default = 0.7f;
  const float x_deviation = 3.2f;
  const float y_deviation = 2.1f;
  const float z_deviation = 2.5f;
  std::uniform_real_distribution<float> dist_x(0, x_deviation);
  std::uniform_real_distribution<float> dist_y(-y_deviation, y_deviation);
  std::uniform_real_distribution<float> dist_z(0, z_deviation);
  std::normal_distribution<float> dist_pixel(0, pixel_error);

  for (size_t index = 0; index < num_points; ++index) {
    Eigen::Vector3f world_position;
    world_position.x() = dist_x(gen) + x_default;
    world_position.y() = dist_y(gen);
    world_position.z() = dist_z(gen) + z_default;

    const Eigen::Vector3f local_position =
        pose_world_to_current.inverse() * world_position;

    Eigen::Vector2f pixel;
    const float inverse_z = 1.0 / local_position.z();
    pixel.x() = fx * local_position.x() * inverse_z + cx;
    pixel.y() = fy * local_position.y() * inverse_z + cy;

    true_world_position_list.push_back(world_position);
    true_pixel_list.push_back(pixel);
  }

  for (size_t index = 0; index < num_points; ++index) {
    const Eigen::Vector2f &true_pixel = true_pixel_list[index];
    const Eigen::Vector3f &world_position = true_world_position_list[index];

    world_position_list.push_back(world_position);

    Eigen::Vector2f pixel = true_pixel;
    pixel.x() += dist_pixel(gen);
    pixel.y() += dist_pixel(gen);
    pixel_list.push_back(pixel);
  }
}

int main() {
  try {
    // Camera parameters
    const size_t n_cols = 640;
    const size_t n_rows = 480;
    const float fx = 338.0;
    const float fy = 338.0;
    const float cx = 320.0;
    const float cy = 240.0;

    // Initialize robot pose
    Eigen::Isometry3f pose_base_to_idle_camera;
    pose_base_to_idle_camera.linear() =
        Eigen::AngleAxisf(-40 * M_PI / 180.0, Eigen::Vector3f::UnitY())
            .toRotationMatrix();
    pose_base_to_idle_camera.translation().x() = 0;
    pose_base_to_idle_camera.translation().y() = 0;
    pose_base_to_idle_camera.translation().z() = 0.1;

    Eigen::Isometry3f pose_idle_camera_to_camera;
    pose_idle_camera_to_camera.linear() =
        Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisf(-M_PI_2, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    pose_idle_camera_to_camera.translation() << 0, 0, 0;

    const Eigen::Isometry3f pose_base_to_camera =
        pose_base_to_idle_camera * pose_idle_camera_to_camera;

    Eigen::Isometry3f pose_base_last_to_base_current;
    pose_base_last_to_base_current.linear() =
        Eigen::AngleAxisf(0.3, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    pose_base_last_to_base_current.translation().x() = 0.5;
    pose_base_last_to_base_current.translation().y() = -0.04;
    pose_base_last_to_base_current.translation().z() = 0;

    Eigen::Isometry3f pose_world_to_current_true;
    pose_world_to_current_true =
        pose_base_last_to_base_current * pose_base_to_camera;
    Eigen::Isometry3f pose_world_to_last;
    pose_world_to_last = pose_base_to_camera;

    // Generate 3D points and projections
    constexpr size_t num_points = 100000;
    constexpr float pixel_error = 0.0;
    std::vector<Eigen::Vector3f> true_world_position_list;
    std::vector<Eigen::Vector2f> true_pixel_list;
    std::vector<Eigen::Vector3f> world_position_list;
    std::vector<Eigen::Vector2f> pixel_list;
    GeneratePoseOnlyBundleAdjustmentSimulationData(
        num_points, pixel_error, pose_world_to_current_true, n_cols, n_rows, fx,
        fy, cx, cy, true_world_position_list, true_pixel_list,
        world_position_list, pixel_list);

    // Make initial guess
    Eigen::Isometry3f pose_world_to_current_native_solver;
    Eigen::Isometry3f pose_world_to_current_initial_guess;
    pose_world_to_current_initial_guess = pose_base_to_camera;
    pose_world_to_current_native_solver = pose_world_to_current_initial_guess;

    // 1) native solver
    std::unique_ptr<visual_navigation::analytic_solver::PoseOptimizer>
        pose_optimizer = std::make_unique<
            visual_navigation::analytic_solver::PoseOptimizer>();
    visual_navigation::analytic_solver::Summary summary;
    visual_navigation::analytic_solver::Options options;
    options.iteration_handle.max_num_iterations = 100;
    options.convergence_handle.threshold_cost_change = 1e-6;
    options.convergence_handle.threshold_step_size = 1e-6;
    options.outlier_handle.threshold_huber_loss = 1.5;
    options.outlier_handle.threshold_outlier_rejection = 2.5;
    std::vector<bool> mask_inlier;
    pose_optimizer->SolveMonocularPoseOnlyBundleAdjustment3Dof(
        world_position_list, pixel_list, fx, fy, cx, cy, pose_base_to_camera,
        pose_world_to_last, pose_world_to_current_native_solver, mask_inlier,
        options, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // 2) Ceres (numerical autodiff)
    double param_b2b1_ceres_numerical[3] = {0, 0, 0};
    MonocularReprojectionErrorCostFunction3DofNumerical::SetPoseBaseToCamera(
        pose_base_to_camera.cast<double>());
    MonocularReprojectionErrorCostFunction3DofNumerical::
        SetCameraIntrinsicParameters(fx, fy, cx, cy);
    ceres::Problem problem_ceres_numerical;
    for (size_t index = 0; index < num_points; ++index) {
      const Eigen::Vector3d world_position =
          world_position_list[index].cast<double>();
      const Eigen::Vector2d pixel_matched = pixel_list[index].cast<double>();
      ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<
          MonocularReprojectionErrorCostFunction3DofNumerical, 2, 3>(
          new MonocularReprojectionErrorCostFunction3DofNumerical(
              world_position, pixel_matched));
      problem_ceres_numerical.AddResidualBlock(cost_function, nullptr,
                                               param_b2b1_ceres_numerical);
    }

    ceres::Solver::Options options_ceres_numerical;
    options_ceres_numerical.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary_ceres_numerical;
    ceres::Solve(options_ceres_numerical, &problem_ceres_numerical,
                 &summary_ceres_numerical);
    std::cout << summary_ceres_numerical.FullReport() << "\n";

    // Compare results
    std::cout << "Compare pose:\n";
    Eigen::Matrix<float, 3, 4> pose_true;
    Eigen::Matrix<float, 3, 4> pose_initial_guess;
    Eigen::Matrix<float, 3, 4> pose_native_solver;
    Eigen::Matrix<float, 3, 4> pose_ceres_autodiff;

    pose_true << pose_world_to_current_true.linear(),
        pose_world_to_current_true.translation();
    std::cout << "truth:\n" << pose_true << std::endl;

    pose_initial_guess << pose_world_to_current_initial_guess.linear(),
        pose_world_to_current_initial_guess.translation();
    std::cout << "Initial guess:\n" << pose_initial_guess << std::endl;

    pose_native_solver << pose_world_to_current_native_solver.linear(),
        pose_world_to_current_native_solver.translation();
    std::cout << "Estimated (native solver):\n"
              << pose_native_solver << std::endl;

    Eigen::Isometry3f pose_b2b1_autodiff;
    pose_b2b1_autodiff.linear() << cos(param_b2b1_ceres_numerical[2]),
        -sin(param_b2b1_ceres_numerical[2]), 0,
        sin(param_b2b1_ceres_numerical[2]), cos(param_b2b1_ceres_numerical[2]),
        0, 0, 0, 1;
    pose_b2b1_autodiff.translation() << param_b2b1_ceres_numerical[0],
        param_b2b1_ceres_numerical[1], 0;
    const Eigen::Isometry3f pose_b1b2_autodiff = pose_b2b1_autodiff.inverse();
    const Eigen::Isometry3f pose_world_to_current_ceres_autodiff =
        pose_b1b2_autodiff * pose_base_to_camera;
    pose_ceres_autodiff << pose_world_to_current_ceres_autodiff.linear(),
        pose_world_to_current_ceres_autodiff.translation();
    std::cout << "Estimated (ceres auto diff):\n"
              << pose_ceres_autodiff << std::endl;

    // Time consumption
    std::cout << "Time (native solver): " << summary.GetTotalTimeInSecond()
              << " [sec]" << std::endl;
    std::cout << "Time (Ceres autodiff): "
              << summary_ceres_numerical.total_time_in_seconds << " [sec]"
              << std::endl;

    // Draw images
    std::vector<Eigen::Isometry3f> debug_pose_list =
        pose_optimizer->GetDebugPoses();

    for (size_t iter = 0; iter < debug_pose_list.size(); ++iter) {
      const Eigen::Isometry3f &pose_world_to_current_temp =
          debug_pose_list[iter];
      std::vector<Eigen::Vector2f> projected_pixel_list;
      for (size_t index = 0; index < num_points; ++index) {
        const Eigen::Vector3f local_position =
            pose_world_to_current_temp.inverse() * world_position_list[index];

        Eigen::Vector2f pixel;
        const float inverse_z = 1.0 / local_position.z();
        pixel.x() = fx * local_position.x() * inverse_z + cx;
        pixel.y() = fy * local_position.y() * inverse_z + cy;

        projected_pixel_list.push_back(pixel);
      }

      cv::Mat image_blank = cv::Mat::zeros(cv::Size(n_cols, n_rows), CV_8UC3);
      for (size_t index = 0; index < pixel_list.size(); ++index) {
        const Eigen::Vector2f &pixel = pixel_list[index];
        const Eigen::Vector2f &projected_pixel = projected_pixel_list[index];
        cv::circle(image_blank, cv::Point2f(pixel.x(), pixel.y()), 4,
                   cv::Scalar(255, 0, 0), 1);
        cv::circle(image_blank,
                   cv::Point2f(projected_pixel.x(), projected_pixel.y()), 2,
                   cv::Scalar(0, 0, 255), 1);
      }
      cv::imshow("optimization process visualization", image_blank);
      cv::waitKey(0);
    }
  } catch (std::exception &e) {
    std::cout << "e.what(): " << e.what() << std::endl;
  }

  return 0;
};
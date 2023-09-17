#include <iostream>
#include <random>
#include <vector>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "core/pose_only_bundle_adjustment_solver.h"
#include "core/pose_only_bundle_adjustment_solver_ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "utility/geometry_library.h"
#include "utility/timer.h"

using Pose = Eigen::Isometry3f;
using Position = Eigen::Vector3f;
using Pixel = Eigen::Vector2f;

void GeneratePoseOnlyBundleAdjustmentSimulationData(
    const int num_points, const Pose &pose_world_to_current, const int n_cols, const int n_rows, const float fx,
    const float fy, const float cx, const float cy, std::vector<Position> &true_world_position_list,
    std::vector<Pixel> &true_pixel_list, std::vector<Position> &world_position_list, std::vector<Pixel> &pixel_list) {
  // Generate 3D points and projections
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

  for (int index = 0; index < num_points; ++index) {
    Position world_position;
    world_position.x() = dist_x(gen);
    world_position.y() = dist_y(gen);
    world_position.z() = dist_z(gen) + z_default;

    const Position local_position = pose_world_to_current.inverse() * world_position;

    Pixel pixel;
    const float inverse_z = 1.0 / local_position.z();
    pixel.x() = fx * local_position.x() * inverse_z + cx;
    pixel.y() = fy * local_position.y() * inverse_z + cy;

    true_world_position_list.push_back(world_position);
    true_pixel_list.push_back(pixel);
  }

  for (int index = 0; index < num_points; ++index) {
    const Position &world_position = true_world_position_list[index];
    const Pixel &true_pixel = true_pixel_list[index];

    world_position_list.push_back(world_position);

    Pixel pixel = true_pixel;
    pixel.x() += dist_pixel(gen);
    pixel.y() += dist_pixel(gen);
    pixel_list.push_back(pixel);
  }
}

int main() {
  try {
    // Camera parameters
    const int n_cols = 640;
    const int n_rows = 480;
    const float fx = 338.0;
    const float fy = 338.0;
    const float cx = 320.0;
    const float cy = 240.0;

    Pose pose_world_to_current;
    pose_world_to_current.linear() = Eigen::AngleAxisf(-0.5, Position::UnitY()).toRotationMatrix();
    pose_world_to_current.translation().x() = 0.2;
    pose_world_to_current.translation().y() = 0.3;
    pose_world_to_current.translation().z() = -1.9;

    // Generate 3D points and projections
    constexpr int num_points = 300000;
    std::vector<Position> true_world_position_list;
    std::vector<Pixel> true_pixel_list;
    std::vector<Position> world_position_list;
    std::vector<Pixel> pixel_list;
    GeneratePoseOnlyBundleAdjustmentSimulationData(num_points, pose_world_to_current, n_cols, n_rows, fx, fy, cx, cy,
                                                   true_world_position_list, true_pixel_list, world_position_list,
                                                   pixel_list);

    // Make initial guess
    Pose pose_world_to_current_initial_guess;
    pose_world_to_current_initial_guess = Pose::Identity();
    pose_world_to_current_initial_guess.translation().x() = 0.0;
    pose_world_to_current_initial_guess.translation().y() = 0.0;
    pose_world_to_current_initial_guess.translation().z() = 0.0;
    Eigen::Matrix<float, 4, 4> T_c2w_initial_guess;
    T_c2w_initial_guess << pose_world_to_current_initial_guess.linear(),
        pose_world_to_current_initial_guess.translation(), 0, 0, 0, 1;
    Eigen::Matrix<float, 6, 1> xi_c2w_initial_guess;
    geometry::SE3Log_f(T_c2w_initial_guess, xi_c2w_initial_guess);

    // 1) native solver
    Pose pose_world_to_current_native_solver;
    pose_world_to_current_native_solver = pose_world_to_current_initial_guess;
    std::unique_ptr<analytic_solver::PoseOnlyBundleAdjustmentSolver> pose_optimizer =
        std::make_unique<analytic_solver::PoseOnlyBundleAdjustmentSolver>();
    analytic_solver::Summary summary_native;
    analytic_solver::Options options_native;
    options_native.iteration_handle.max_num_iterations = 100;
    options_native.convergence_handle.threshold_cost_change = 1e-6;
    options_native.convergence_handle.threshold_step_size = 1e-6;
    options_native.outlier_handle.threshold_huber_loss = 1.0;
    options_native.outlier_handle.threshold_outlier_rejection = 2.5;
    options_native.solver_type = analytic_solver::solver_type_enum::GAUSS_NEWTON;

    std::vector<bool> mask_inlier;
    pose_optimizer->Solve_Monocular_6Dof(world_position_list, pixel_list, fx, fy, cx, cy,
                                         pose_world_to_current_native_solver, mask_inlier, options_native,
                                         &summary_native);
    std::cout << "[SUMMARY of NATIVE SOLVER]:\n";
    std::cout << summary_native.BriefReport() << std::endl;

    // 2) Ceres (analytic jacobian)
    // double parameters_ceres_analytic[6] = {
    //     xi_c2w_initial_guess(0), xi_c2w_initial_guess(1), xi_c2w_initial_guess(2),
    //     xi_c2w_initial_guess(3), xi_c2w_initial_guess(4), xi_c2w_initial_guess(5)};
    // ceres::Problem problem_ceres_analytic;
    // ReprojectionCostFunctor_6dof_numerical::SetCameraIntrinsicParameters(fx, fy, cx, cy);
    // for (int index = 0; index < num_points; ++index)
    // {
    //   const Eigen::Vector3d world_position = world_position_list[index].cast<double>();
    //   const Eigen::Vector2d pixel_matched = pixel_list[index].cast<double>();
    //   ceres::CostFunction *cost_function = new MonocularReprojectionErrorCostFunction6DofAnalytic(world_position,
    //   pixel_matched); problem_ceres_analytic.AddResidualBlock(cost_function, nullptr, parameters_ceres_analytic);
    // }

    // ceres::Solver::Options options_ceres_analytic;
    // options_ceres_analytic.minimizer_progress_to_stdout = false;
    // ceres::Solver::Summary summary_ceres_analytic;
    // ceres::Solve(options_ceres_analytic, &problem_ceres_analytic, &summary_ceres_analytic);
    // std::cout << summary_ceres_analytic.FullReport() << "\n";

    // Position w_c2w_ceres_analytic;
    // w_c2w_ceres_analytic << parameters_ceres_analytic[3], parameters_ceres_analytic[4], parameters_ceres_analytic[5];
    // Eigen::Matrix<float, 3, 3> R_c2w_ceres_analytic;
    // geometry::so3Exp_f(w_c2w_ceres_analytic, R_c2w_ceres_analytic);
    // Position t_c2w_ceres_analytic;
    // t_c2w_ceres_analytic << parameters_ceres_analytic[0], parameters_ceres_analytic[1], parameters_ceres_analytic[2];
    // Pose pose_world_to_current_ceres_analytic;
    // pose_world_to_current_ceres_analytic.linear() = R_c2w_ceres_analytic.transpose();
    // pose_world_to_current_ceres_analytic.translation() = -R_c2w_ceres_analytic.transpose() * t_c2w_ceres_analytic;

    // 3) Ceres (numerical autodiff)
    timer::tic();
    double param_c2w_ceres_numerical[6] = {0, 0, 0, 0, 0, 0};
    ReprojectionCostFunctor_6dof_numerical::SetCameraIntrinsicParameters(fx, fy, cx, cy);
    ceres::Problem problem_ceres_numerical;
    for (int index = 0; index < num_points; ++index) {
      const Eigen::Vector3d world_position = world_position_list[index].cast<double>();
      const Eigen::Vector2d pixel_matched = pixel_list[index].cast<double>();
      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<ReprojectionCostFunctor_6dof_numerical, 2, 6>(
              new ReprojectionCostFunctor_6dof_numerical(world_position, pixel_matched));
      problem_ceres_numerical.AddResidualBlock(cost_function, nullptr, param_c2w_ceres_numerical);
    }
    const double time_ceres_numerical_data_insertion = timer::toc(0);

    timer::tic();
    ceres::Solver::Options options_ceres_numerical;
    options_ceres_numerical.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary_ceres_numerical;
    ceres::Solve(options_ceres_numerical, &problem_ceres_numerical, &summary_ceres_numerical);
    std::cout << "[SUMMARY of CERES NUMERICAL SOLVER]:\n";
    std::cout << summary_ceres_numerical.BriefReport() << "\n";
    // std::cout << summary_ceres_numerical.FullReport() << "\n";
    const double time_ceres_numerical_problem_solve = timer::toc(0);

    Eigen::Matrix<float, 3, 1> w_c2w_ceres_numerical;
    w_c2w_ceres_numerical << param_c2w_ceres_numerical[0], param_c2w_ceres_numerical[1], param_c2w_ceres_numerical[2];
    Eigen::Matrix<float, 3, 3> R_c2w_ceres_numerical;
    Position t_c2w_ceres_numerical;
    geometry::so3Exp_f(w_c2w_ceres_numerical, R_c2w_ceres_numerical);
    t_c2w_ceres_numerical << param_c2w_ceres_numerical[3], param_c2w_ceres_numerical[4], param_c2w_ceres_numerical[5];
    Pose pose_world_to_current_ceres_numerical;
    pose_world_to_current_ceres_numerical.linear() = R_c2w_ceres_numerical.transpose();
    pose_world_to_current_ceres_numerical.translation() = -R_c2w_ceres_numerical.transpose() * t_c2w_ceres_numerical;

    // Compare results
    Eigen::Matrix<float, 3, 4> pose_true;
    Eigen::Matrix<float, 3, 4> pose_initial_guess;
    Eigen::Matrix<float, 3, 4> pose_native_solver;
    // Eigen::Matrix<float, 3, 4> pose_ceres_analytic;
    Eigen::Matrix<float, 3, 4> pose_ceres_numerical;

    std::cout << "Compare pose:\n";

    pose_true << pose_world_to_current.linear(), pose_world_to_current.translation();
    std::cout << "truth:\n" << pose_true << std::endl;

    pose_initial_guess << pose_world_to_current_initial_guess.linear(),
        pose_world_to_current_initial_guess.translation();
    std::cout << "Initial guess:\n" << pose_initial_guess << std::endl;

    pose_native_solver << pose_world_to_current_native_solver.linear(),
        pose_world_to_current_native_solver.translation();
    std::cout << "Estimated (native solver):\n" << pose_native_solver << std::endl;

    // pose_ceres_analytic << pose_world_to_current_ceres_analytic.linear(),
    // pose_world_to_current_ceres_analytic.translation(); std::cout << "Estimated (ceres analytic):\n"
    //           << pose_ceres_analytic << std::endl;

    pose_ceres_numerical << pose_world_to_current_ceres_numerical.linear(),
        pose_world_to_current_ceres_numerical.translation();
    std::cout << "Estimated (ceres numerical):\n" << pose_ceres_numerical << std::endl;

    std::cout << "Total time in sec         (native):" << summary_native.GetTotalTimeInSecond() << " [sec]"
              << std::endl;
    // std::cout << "Total time in sec (ceres_analytic):" << summary_ceres_analytic.total_time_in_seconds << " [sec]" <<
    // std::endl;
    std::cout << "Total time in sec (ceres_autodiff):" << summary_ceres_numerical.total_time_in_seconds << " [sec]"
              << std::endl;
    std::cout << "[Details] TIME ANALYSIS of Ceres Numerical Solver: \n";
    std::cout << "          insertion: " << time_ceres_numerical_data_insertion * 0.001 << " [s]\n";
    std::cout << "              solve: " << time_ceres_numerical_problem_solve * 0.001 << " [s]\n";
    std::cout << "              total: "
              << (time_ceres_numerical_data_insertion + time_ceres_numerical_problem_solve) * 0.001 << " [ms]\n";

    std::cout << "\nSpeed up ratio by using natvie solver: ";
    std::cout << (time_ceres_numerical_data_insertion + time_ceres_numerical_problem_solve) * 0.001 /
                     summary_native.GetTotalTimeInSecond()
              << " times faster\n";

    // Draw images
    std::vector<Pose> debug_pose_list = pose_optimizer->GetDebugPoses();

    for (int iter = 0; iter < debug_pose_list.size(); ++iter) {
      const Pose &pose_world_to_current_temp = debug_pose_list[iter];
      std::vector<Pixel> projected_pixel_list;
      for (int index = 0; index < num_points; ++index) {
        const Position local_position = pose_world_to_current_temp.inverse() * world_position_list[index];

        Pixel pixel;
        const float inverse_z = 1.0 / local_position.z();
        pixel.x() = fx * local_position.x() * inverse_z + cx;
        pixel.y() = fy * local_position.y() * inverse_z + cy;

        projected_pixel_list.push_back(pixel);
      }

      cv::Mat image_blank = cv::Mat::zeros(cv::Size(n_cols, n_rows), CV_8UC3);
      for (int index = 0; index < pixel_list.size(); ++index) {
        const Pixel &pixel = pixel_list[index];
        const Pixel &projected_pixel = projected_pixel_list[index];
        cv::circle(image_blank, cv::Point2f(pixel.x(), pixel.y()), 4, cv::Scalar(255, 0, 0), 1);
        cv::circle(image_blank, cv::Point2f(projected_pixel.x(), projected_pixel.y()), 2, cv::Scalar(0, 0, 255), 1);
      }
      cv::imshow("optimization process visualization", image_blank);
      cv::waitKey(0);
    }
  } catch (std::exception &e) {
    std::cout << "e.what(): " << e.what() << std::endl;
  }

  return 0;
};
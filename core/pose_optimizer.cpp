#include "pose_optimizer.h"
namespace analytic_solver
{
  std::string Summary::BriefReport()
  {
    const auto default_precision{std::cout.precision()};
    std::stringstream ss;
    ss << "iter ";              // 5
    ss << "     cost     ";     // 14
    ss << " cost_change ";      // 13
    ss << "  |step|  ";         // 10
    ss << " |gradient| ";       // 12
    ss << " iter_time [ms] \n"; // 11

    const size_t num_iterations = optimization_info_list_.size();
    for (size_t iteration = 0; iteration < num_iterations; ++iteration)
    {
      const OptimizationInfo &optimization_info = optimization_info_list_[iteration];
      ss << std::setw(4) << iteration << " ";
      ss << " " << std::scientific << optimization_info.cost;
      ss << "    " << std::setprecision(2) << std::scientific << optimization_info.cost_change;
      ss << "   " << std::setprecision(2) << std::scientific << optimization_info.abs_step;
      ss << "   " << std::setprecision(2) << std::scientific << optimization_info.abs_gradient;
      ss << "      " << std::setprecision(2) << std::scientific << optimization_info.iter_time << "\n";
      ss << std::setprecision(default_precision); // restore defaults
    }
    ss << std::setprecision(5);
    ss << "Analytic Solver Report: Iterations: " << num_iterations << ", Total time [ms]: " << total_time_in_millisecond_ << ", Initial cost: " << optimization_info_list_.front().cost << ", Final cost: " << optimization_info_list_.back().cost;
    ss << ", Termination : " << (convergence_status_ ? "CONVERGENCE" : "NO_CONVERGENCE") << "\n";
    if (max_iteration_ == num_iterations)
      ss << " WARNIING: MAX ITERATION is reached ! The solution could be local minima.\n";
    ss << std::setprecision(default_precision); // restore defaults
    return ss.str();
  }
  Summary::Summary() {}
  Summary::~Summary() {}
  const double Summary::GetTotalTimeInSecond() const
  {
    return total_time_in_millisecond_ * 0.001;
  }

  PoseOptimizer::PoseOptimizer()
  {
  }
  PoseOptimizer::~PoseOptimizer() {}

  bool PoseOptimizer::SolveMonocularPoseOnlyBundleAdjustment6Dof(
      const std::vector<Eigen::Vector3f> &world_position_list,
      const std::vector<Eigen::Vector2f> &matched_pixel_list,
      const float fx, const float fy, const float cx, const float cy,
      Eigen::Isometry3f &pose_world_to_current_frame,
      std::vector<bool> &mask_inlier,
      Options options,
      Summary *summary)
  {
    timer::StopWatch stopwatch("SolveMonocularPoseOnlyBundleAdjustment6Dof");
    const auto &max_iteration = options.iteration_handle.max_num_iterations;
    const auto &threshold_convergence_delta_error = options.convergence_handle.threshold_cost_change;
    const auto &threshold_convergence_delta_pose = options.convergence_handle.threshold_step_size;
    const auto &threshold_huber_loss = options.outlier_handle.threshold_huber_loss;
    const auto &threshold_outlier_reproj_error = options.outlier_handle.threshold_outlier_rejection;
    if (summary != nullptr)
    {
      summary->max_iteration_ = max_iteration;
      summary->threshold_cost_change_ = threshold_convergence_delta_error;
      summary->threshold_step_size_ = threshold_convergence_delta_pose;
      summary->convergence_status_ = true;
    }
    debug_poses_.resize(0);
    if (world_position_list.size() != matched_pixel_list.size())
      throw std::runtime_error("In PoseOptimizer::SolveMonocularPoseOnlyBundleAdjustment6Dof(), world_position_list.size() != current_pixel_list.size()");

    bool is_success = true;

    const size_t n_pts = world_position_list.size();
    const float inverse_n_pts = 1.0f / static_cast<float>(n_pts);
    mask_inlier.resize(n_pts, true);

    const auto &MAX_ITERATION = max_iteration;
    const auto &THRES_HUBER = threshold_huber_loss; // pixels
    const auto &THRES_DELTA_XI = threshold_convergence_delta_pose;
    const auto &THRES_DELTA_ERROR = threshold_convergence_delta_error;
    const auto &THRES_REPROJ_ERROR = threshold_outlier_reproj_error; // pixels

    Eigen::Isometry3f pose_camera_to_world_optimized = pose_world_to_current_frame.inverse();

    stopwatch.Start();
    bool is_converged = true;
    float err_prev = 1e10f;
    float lambda = 1e-5f;
    std::vector<Eigen::Vector3f> warped_local_position_list;
    for (int iteration = 0; iteration < MAX_ITERATION; ++iteration)
    {
      Eigen::Matrix<float, 6, 6> JtWJ;
      Eigen::Matrix<float, 6, 1> mJtWr;
      JtWJ.setZero();
      mJtWr.setZero();

      // Warp and project 3d point & calculate error
      WarpPointList(pose_camera_to_world_optimized, world_position_list,
                    warped_local_position_list);
      float err_curr = 0.0f;
      size_t count_invalid = 0;
      for (size_t index = 0; index < n_pts; ++index)
      {
        const Eigen::Vector2f &matched_pixel = matched_pixel_list[index];
        const Eigen::Vector3f &local_position = warped_local_position_list[index];

        Eigen::Matrix<float, 6, 1> jacobian_transpose_u;
        Eigen::Matrix<float, 6, 1> jacobian_transpose_v;
        Eigen::Matrix<float, 2, 1> residual_vector;
        ComputeJacobianResidual_ReprojectionError_6Dof(
            local_position,
            matched_pixel, fx, fy, cx, cy,
            jacobian_transpose_u, jacobian_transpose_v, residual_vector);

        Eigen::Matrix<float, 6, 6> hessian_i;
        Eigen::Matrix<float, 6, 1> gradient_i;
        float error_i = 0;
        float error_nonweighted_i = 0;
        ComputeGradientHessian_ReprojectionError_6Dof(
            jacobian_transpose_u, jacobian_transpose_v, residual_vector,
            THRES_HUBER,
            gradient_i, hessian_i, error_i, error_nonweighted_i);

        // Add gradient and Hessian to original matrix
        mJtWr.noalias() -= gradient_i;
        AddHessianOnlyUpperTriangle_6Dof(hessian_i, JtWJ);
        err_curr += error_i;

        // Outlier rejection
        if (error_nonweighted_i >= THRES_REPROJ_ERROR)
        {
          mask_inlier[index] = false;
          ++count_invalid;
        }
      }

      // Solve H^-1*Jtr;
      FillLowerTriangle_6Dof(JtWJ);
      for (size_t i = 0; i < 6; ++i)
        JtWJ(i, i) *= (1.0f + lambda); // lambda

      const Eigen::Matrix<float, 6, 1> delta_xi = JtWJ.ldlt().solve(mJtWr);
      Eigen::Isometry3f delta_pose;
      se3Exp<float>(delta_xi, delta_pose);
      pose_camera_to_world_optimized = delta_pose * pose_camera_to_world_optimized;

      debug_poses_.push_back(pose_camera_to_world_optimized.inverse());

      err_curr *= (inverse_n_pts * 0.5f);
      const float delta_error = abs(err_curr - err_prev);

      if (delta_xi.norm() < THRES_DELTA_XI || delta_error < THRES_DELTA_ERROR) // Early convergence
      {
        is_converged = true;
        break;
      }
      if (iteration == MAX_ITERATION - 1)
        is_converged = false;

      // Go to next step
      err_prev = err_curr;

      const double iter_time = stopwatch.GetLapTimeFromLatest();
      if (summary != nullptr)
      {
        OptimizationInfo optimization_info;
        optimization_info.cost = err_curr;
        optimization_info.cost_change = abs(delta_error);
        optimization_info.abs_step = delta_xi.norm();
        optimization_info.abs_gradient = 0;
        optimization_info.iter_time = iter_time;

        summary->optimization_info_list_.push_back(optimization_info);
      }
    }

    const double total_time = stopwatch.GetLapTimeFromStart();
    if (summary != nullptr)
    {
      summary->convergence_status_ = is_converged;
      summary->total_time_in_millisecond_ = total_time;
    }

    if (!std::isnan(pose_camera_to_world_optimized.linear().norm()))
    {
      pose_world_to_current_frame = pose_camera_to_world_optimized.inverse();
    }
    else
    {
      std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
                << ", Tcw_optimized: \n"
                << pose_camera_to_world_optimized.linear() << " " << pose_camera_to_world_optimized.translation() << "\n";
      is_success = false; // if nan, do not update.
    }

    return is_success;
  }

  bool PoseOptimizer::SolveStereoPoseOnlyBundleAdjustment6Dof(
      const std::vector<Eigen::Vector3f> &world_position_list,
      const std::vector<Eigen::Vector2f> &matched_left_pixel_list,
      const std::vector<Eigen::Vector2f> &matched_right_pixel_list,
      const float fx_left, const float fy_left, const float cx_left, const float cy_left,
      const float fx_right, const float fy_right, const float cx_right, const float cy_right,
      const Eigen::Isometry3f &pose_left_to_right,
      Eigen::Isometry3f &pose_world_to_left_current_frame,
      std::vector<bool> &mask_inlier_left,
      std::vector<bool> &mask_inlier_right,
      Options options,
      Summary *summary)
  {
    timer::StopWatch stopwatch("SolveStereoPoseOnlyBundleAdjustment6Dof");
    const auto &max_iteration = options.iteration_handle.max_num_iterations;
    const auto &threshold_convergence_delta_error = options.convergence_handle.threshold_cost_change;
    const auto &threshold_convergence_delta_pose = options.convergence_handle.threshold_step_size;
    const auto &threshold_huber_loss = options.outlier_handle.threshold_huber_loss;
    const auto &threshold_outlier_reproj_error = options.outlier_handle.threshold_outlier_rejection;

    if (summary != nullptr)
    {
      summary->max_iteration_ = max_iteration;
      summary->threshold_cost_change_ = threshold_convergence_delta_error;
      summary->threshold_step_size_ = threshold_convergence_delta_pose;
      summary->convergence_status_ = true;
    }
    debug_poses_.resize(0);
    if (world_position_list.size() != matched_left_pixel_list.size())
      throw std::runtime_error("In PoseOptimizer::SolveStereoPoseOnlyBundleAdjustment6Dof(), world_position_list.size() != left_current_pixel_list.size()");
    if (world_position_list.size() != matched_right_pixel_list.size())
      throw std::runtime_error("In PoseOptimizer::SolveStereoPoseOnlyBundleAdjustment6Dof(), world_position_list.size() != right_current_pixel_list.size()");

    bool is_success = true;

    const size_t n_pts = world_position_list.size();
    mask_inlier_left.resize(n_pts, true);
    mask_inlier_right.resize(n_pts, true);

    const auto &MAX_ITERATION = max_iteration;
    const auto &THRES_HUBER = threshold_huber_loss; // pixels
    const auto &THRES_DELTA_XI = threshold_convergence_delta_pose;
    const auto &THRES_DELTA_ERROR = threshold_convergence_delta_error;
    const auto &THRES_REPROJ_ERROR = threshold_outlier_reproj_error; // pixels

    const Eigen::Isometry3f pose_right_to_left = pose_left_to_right.inverse();
    Eigen::Isometry3f pose_camera_to_world_optimized = pose_world_to_left_current_frame.inverse();

    stopwatch.Start();
    bool is_converged = true;
    float err_prev = 1e10f;
    float lambda = 1e-5f;
    std::vector<Eigen::Vector3f> warped_left_position_list;
    std::vector<Eigen::Vector3f> warped_right_position_list;
    for (int iter = 0; iter < MAX_ITERATION; ++iter)
    {
      Eigen::Matrix<float, 6, 6> JtWJ;
      Eigen::Matrix<float, 6, 1> mJtWr;
      JtWJ.setZero();
      mJtWr.setZero();

      const Eigen::Isometry3f pose_right_camera_to_world_optimized = pose_right_to_left * pose_camera_to_world_optimized;

      // Warp and project 3d point & calculate error
      WarpPointList(pose_camera_to_world_optimized, world_position_list,
                    warped_left_position_list);
      WarpPointList(pose_right_to_left, warped_left_position_list,
                    warped_right_position_list);

      float err_curr = 0.0f;
      size_t count_invalid = 0;
      size_t count_left_edge = 0;
      size_t count_right_edge = 0;
      for (size_t index = 0; index < n_pts; ++index)
      {
        const Eigen::Vector2f &matched_left_pixel = matched_left_pixel_list[index];
        const Eigen::Vector2f &matched_right_pixel = matched_right_pixel_list[index];
        const Eigen::Vector3f &left_local_position = warped_left_position_list[index];
        const Eigen::Vector3f &right_local_position = warped_right_position_list[index];

        // left
        ++count_left_edge;
        Eigen::Matrix<float, 6, 1> jacobian_transpose_u_left;
        Eigen::Matrix<float, 6, 1> jacobian_transpose_v_left;
        Eigen::Matrix<float, 2, 1> residual_vector_left;
        ComputeJacobianResidual_ReprojectionError_6Dof(
            left_local_position, matched_left_pixel, fx_left, fy_left, cx_left, cy_left,
            jacobian_transpose_u_left, jacobian_transpose_v_left, residual_vector_left);

        Eigen::Matrix<float, 6, 1> gradient_i_left;
        Eigen::Matrix<float, 6, 6> hessian_i_left;
        float error_i_left = 0;
        float error_nonweighted_i_left = 0;
        ComputeGradientHessian_ReprojectionError_6Dof(
            jacobian_transpose_u_left, jacobian_transpose_v_left, residual_vector_left,
            THRES_HUBER,
            gradient_i_left, hessian_i_left, error_i_left, error_nonweighted_i_left);

        // Add gradient and Hessian to original matrix
        mJtWr.noalias() -= gradient_i_left;
        AddHessianOnlyUpperTriangle_6Dof(hessian_i_left, JtWJ);
        err_curr += error_i_left;

        // Outlier rejection
        if (error_nonweighted_i_left >= THRES_REPROJ_ERROR)
        {
          mask_inlier_left[index] = false;
          ++count_invalid;
        }

        // Check right
        if (matched_right_pixel.x() < 0 || matched_right_pixel.y() < 0)
          continue;

        ++count_right_edge;
        Eigen::Matrix<float, 6, 1> jacobian_transpose_u_right;
        Eigen::Matrix<float, 6, 1> jacobian_transpose_v_right;
        Eigen::Matrix<float, 2, 1> residual_vector_right;
        ComputeJacobianResidual_ReprojectionError_6Dof(
            right_local_position, matched_right_pixel, fx_right, fy_right, cx_right, cy_right,
            jacobian_transpose_u_right, jacobian_transpose_v_right, residual_vector_right);

        // Huber weight calculation by the Manhattan distance
        Eigen::Matrix<float, 6, 6> hessian_i_right;
        Eigen::Matrix<float, 6, 1> gradient_i_right;
        float error_i_right = 0;
        float error_nonweighted_i_right = 0;
        ComputeGradientHessian_ReprojectionError_6Dof(
            jacobian_transpose_u_right, jacobian_transpose_v_right, residual_vector_right,
            THRES_HUBER,
            gradient_i_right, hessian_i_right, error_i_right, error_nonweighted_i_right);

        // Add gradient and Hessian to original matrix
        mJtWr.noalias() -= gradient_i_right;
        AddHessianOnlyUpperTriangle_6Dof(hessian_i_right, JtWJ);
        err_curr += error_i_right;

        // Outlier rejection
        if (error_nonweighted_i_right >= THRES_REPROJ_ERROR)
        {
          mask_inlier_right[index] = false;
          ++count_invalid;
        }
      }

      // Solve H^-1*Jtr;
      FillLowerTriangle_6Dof(JtWJ);
      for (size_t i = 0; i < 6; ++i)
        JtWJ(i, i) *= (1.0f + lambda); // lambda

      const Eigen::Matrix<float, 6, 1> delta_xi = JtWJ.ldlt().solve(mJtWr);
      Eigen::Isometry3f delta_pose;
      se3Exp<float>(delta_xi, delta_pose);
      pose_camera_to_world_optimized = delta_pose * pose_camera_to_world_optimized;

      debug_poses_.push_back(pose_camera_to_world_optimized.inverse());

      err_curr /= (count_left_edge + count_right_edge) * 0.5f;
      const float delta_error = abs(err_curr - err_prev);

      if (delta_xi.norm() < THRES_DELTA_XI || delta_error < THRES_DELTA_ERROR)
      {
        // Early convergence
        is_converged = true;
        break;
      }
      if (iter == MAX_ITERATION - 1)
        is_converged = false;

      err_prev = err_curr;

      const double iter_time = stopwatch.GetLapTimeFromLatest();
      if (summary != nullptr)
      {
        OptimizationInfo optimization_info;
        optimization_info.cost = err_curr;
        optimization_info.cost_change = abs(delta_error);
        optimization_info.abs_step = delta_xi.norm();
        optimization_info.abs_gradient = 0;
        optimization_info.iter_time = iter_time;

        summary->optimization_info_list_.push_back(optimization_info);
      }
    }

    const double total_time = stopwatch.GetLapTimeFromStart();
    if (summary != nullptr)
    {
      summary->convergence_status_ = is_converged;
      summary->total_time_in_millisecond_ = total_time;
    }

    if (!std::isnan(pose_camera_to_world_optimized.linear().norm()))
    {
      pose_world_to_left_current_frame = pose_camera_to_world_optimized.inverse();
    }
    else
    {
      std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
                << ", Tcw_optimized: \n"
                << pose_camera_to_world_optimized.linear() << " " << pose_camera_to_world_optimized.translation() << "\n";
      is_success = false; // if nan, do not update.
    }

    return is_success;
  }

  bool PoseOptimizer::SolveMonocularPoseOnlyBundleAdjustment3Dof(
      const std::vector<Eigen::Vector3f> &world_position_list,
      const std::vector<Eigen::Vector2f> &matched_pixel_list,
      const float fx, const float fy, const float cx, const float cy,
      const Eigen::Isometry3f &pose_base_to_camera,
      const Eigen::Isometry3f &pose_world_to_last,
      Eigen::Isometry3f &pose_world_to_current,
      std::vector<bool> &mask_inlier,
      Options options,
      Summary *summary)
  {
    timer::StopWatch stopwatch("SolveMonocularPoseOnlyBundleAdjustment3Dof");
    const auto &max_iteration = options.iteration_handle.max_num_iterations;
    const auto &threshold_convergence_delta_error = options.convergence_handle.threshold_cost_change;
    const auto &threshold_convergence_delta_pose = options.convergence_handle.threshold_step_size;
    const auto &threshold_huber_loss = options.outlier_handle.threshold_huber_loss;
    const auto &threshold_outlier_reproj_error = options.outlier_handle.threshold_outlier_rejection;
    if (summary != nullptr)
    {
      summary->max_iteration_ = max_iteration;
      summary->threshold_cost_change_ = threshold_convergence_delta_error;
      summary->threshold_step_size_ = threshold_convergence_delta_pose;
      summary->convergence_status_ = true;
    }
    debug_poses_.resize(0);
    if (world_position_list.size() != matched_pixel_list.size())
      throw std::runtime_error("In PoseOptimizer::SolveMonocularPoseOnlyBundleAdjustment3Dof(), world_position_list.size() != current_pixel_list.size()");

    bool is_success = true;

    const auto n_pts = world_position_list.size();
    const float inverse_n_pts = 1.0f / static_cast<float>(n_pts);
    mask_inlier.resize(n_pts, true);

    const auto &MAX_ITERATION = max_iteration;
    const auto &THRES_HUBER = threshold_huber_loss; // pixels
    const auto &THRES_DELTA_XI = threshold_convergence_delta_pose;
    const auto &THRES_DELTA_ERROR = threshold_convergence_delta_error;
    const auto &THRES_REPROJ_ERROR = threshold_outlier_reproj_error; // pixels

    const Eigen::Isometry3f pose_camera_to_base = pose_base_to_camera.inverse();
    const Eigen::Matrix3f R_cb = pose_camera_to_base.linear();

    // Warp the world position to the last frame
    const Eigen::Isometry3f pose_c2c1_prior =
        pose_world_to_current.inverse() * pose_world_to_last;
    const Eigen::Isometry3f pose_b2b1 = pose_base_to_camera * pose_c2c1_prior * pose_camera_to_base;

    // calculate prior theta_21 = [x_21, y_21, psi_21]^T
    const auto R_b2b1 = pose_b2b1.linear();
    const auto i2 = R_b2b1.block<3, 1>(0, 0);
    const float x_21_initial_value = pose_b2b1.translation().x();
    const float y_21_initial_value = pose_b2b1.translation().y();
    const float psi_21_initial_value = atan2(i2(1), i2(0));

    Eigen::Isometry3f pose_world_to_current_optimized = pose_world_to_current;
    Eigen::Vector3f parameter_b2b1_optimized;
    parameter_b2b1_optimized(0) = x_21_initial_value;
    parameter_b2b1_optimized(1) = y_21_initial_value;
    parameter_b2b1_optimized(2) = psi_21_initial_value;

    Eigen::Isometry3f pose_b2b1_optimized;
    stopwatch.Start();
    bool is_converged = true;
    float err_prev = 1e10f;
    float lambda = 1e-5f;
    std::vector<Eigen::Vector3f> warped_local_position_list;
    for (int iteration = 0; iteration < MAX_ITERATION; ++iteration)
    {
      Eigen::Matrix<float, 3, 3> JtWJ;
      Eigen::Matrix<float, 3, 1> mJtWr;
      JtWJ.setZero();
      mJtWr.setZero();

      // Calculate currently optimized pose
      const auto &x_b2b1_optimized = parameter_b2b1_optimized(0);
      const auto &y_b2b1_optimized = parameter_b2b1_optimized(1);
      const auto &psi_b2b1_optimized = parameter_b2b1_optimized(2);
      const float cos_psi = cos(psi_b2b1_optimized);
      const float sin_psi = sin(psi_b2b1_optimized);

      pose_b2b1_optimized.linear() << cos_psi, -sin_psi, 0,
          sin_psi, cos_psi, 0,
          0, 0, 1;
      pose_b2b1_optimized.translation() << x_b2b1_optimized, y_b2b1_optimized, 0;
      const auto pose_c2b1_optimized = pose_camera_to_base * pose_b2b1_optimized;

      // Warp and project 3d point & calculate error
      WarpPointList(pose_c2b1_optimized, world_position_list,
                    warped_local_position_list);
      float err_curr = 0.0f;
      size_t count_invalid = 0;
      for (size_t index = 0; index < n_pts; ++index)
      {
        const Eigen::Vector2f &matched_pixel = matched_pixel_list[index];
        const Eigen::Vector3f &world_position = world_position_list[index];
        const Eigen::Vector3f &local_position = warped_local_position_list[index];

        Eigen::Matrix<float, 3, 1> jacobian_transpose_u;
        Eigen::Matrix<float, 3, 1> jacobian_transpose_v;
        Eigen::Matrix<float, 2, 1> residual_i;
        ComputeJacobianResidual_ReprojectionError_Planar3Dof(
            local_position, world_position,
            matched_pixel, fx, fy, cx, cy,
            R_cb, cos_psi, sin_psi,
            jacobian_transpose_u, jacobian_transpose_v, residual_i);

        Eigen::Matrix<float, 3, 3> hessian_i;
        Eigen::Matrix<float, 3, 1> gradient_i;
        float error_i = 0;
        float error_nonweighted_i = 0;
        ComputeGradientHessian_ReprojectionError_Planar3Dof(
            jacobian_transpose_u, jacobian_transpose_v, residual_i,
            THRES_HUBER,
            gradient_i, hessian_i, error_i, error_nonweighted_i);

        // Add gradient and Hessian to original matrix
        mJtWr.noalias() -= gradient_i;
        AddHessianOnlyUpperTriangle_3Dof(hessian_i, JtWJ);
        err_curr += error_i;

        // Outlier rejection
        if (error_nonweighted_i >= THRES_REPROJ_ERROR)
        {
          mask_inlier[index] = false;
          ++count_invalid;
        }
      }

      // Solve H^-1*Jtr;
      FillLowerTriangleByUpperTriangle_3Dof(JtWJ);
      for (size_t i = 0; i < 3; ++i)
        JtWJ(i, i) *= (1.0f + lambda); // lambda

      const Eigen::Matrix<float, 3, 1> delta_param = (JtWJ.ldlt().solve(mJtWr));

      Eigen::Isometry3f delta_pose;
      const auto &delta_x = delta_param(0);
      const auto &delta_y = delta_param(1);
      const auto &delta_psi = delta_param(2);
      delta_pose.linear() << cos(delta_psi), -sin(delta_psi), 0, sin(delta_psi), cos(delta_psi), 0, 0, 0, 1;
      delta_pose.translation() << delta_x, delta_y, 0;
      pose_b2b1_optimized = delta_pose * pose_b2b1_optimized;

      parameter_b2b1_optimized(0) = pose_b2b1_optimized.translation().x();
      parameter_b2b1_optimized(1) = pose_b2b1_optimized.translation().y();
      parameter_b2b1_optimized(2) += delta_psi;

      pose_world_to_current_optimized = pose_b2b1_optimized.inverse() * pose_base_to_camera;
      debug_poses_.push_back(pose_world_to_current_optimized);

      err_curr *= (inverse_n_pts * 0.5f);
      const float delta_error = abs(err_curr - err_prev);

      if (delta_param.norm() < THRES_DELTA_XI || delta_error < THRES_DELTA_ERROR)
      {
        is_converged = true;
        // std::cout << "    Early convergence. stops at iteration " << iter << ", error: " << err_curr << ", delta_error: " << delta_error << ", step_size: " << delta_xi.norm() << ", # invalid: " << count_invalid << "\n";
        break;
      }
      if (iteration == MAX_ITERATION - 1)
      {
        is_converged = false;
      }
      err_prev = err_curr;

      const double iter_time = stopwatch.GetLapTimeFromLatest();
      if (summary != nullptr)
      {
        OptimizationInfo optimization_info;
        optimization_info.cost = err_curr;
        optimization_info.cost_change = abs(delta_error);
        optimization_info.abs_step = delta_param.norm();
        optimization_info.abs_gradient = 0;
        optimization_info.iter_time = iter_time;

        summary->optimization_info_list_.push_back(optimization_info);
      }
    }

    const double total_time = stopwatch.GetLapTimeFromStart();
    if (summary != nullptr)
    {
      summary->convergence_status_ = is_converged;
      summary->total_time_in_millisecond_ = total_time;
    }

    if (!std::isnan(pose_b2b1_optimized.linear().norm()))
    {
      pose_world_to_current = pose_world_to_current_optimized;
    }
    else
    {
      std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
                << ", pose_21_optimized: \n"
                << pose_b2b1_optimized.linear() << " " << pose_b2b1_optimized.translation() << "\n";
      is_success = false; // if nan, do not update.
    }

    return is_success;
  }

  bool PoseOptimizer::SolveStereoPoseOnlyBundleAdjustment3Dof(
      const std::vector<Eigen::Vector3f> &world_position_list,
      const std::vector<Eigen::Vector2f> &matched_left_pixel_list,
      const std::vector<Eigen::Vector2f> &matched_right_pixel_list,
      const float fx_left, const float fy_left, const float cx_left, const float cy_left,
      const float fx_right, const float fy_right, const float cx_right, const float cy_right,
      const Eigen::Isometry3f &pose_base_to_camera,
      const Eigen::Isometry3f &pose_left_to_right,
      const Eigen::Isometry3f &pose_world_to_last,
      Eigen::Isometry3f &pose_world_to_current,
      std::vector<bool> &mask_inlier_left,
      std::vector<bool> &mask_inlier_right,
      Options options,
      Summary *summary)
  {
    timer::StopWatch stopwatch("SolveMonocularPoseOnlyBundleAdjustment3Dof");
    const auto &max_iteration = options.iteration_handle.max_num_iterations;
    const auto &threshold_convergence_delta_error = options.convergence_handle.threshold_cost_change;
    const auto &threshold_convergence_delta_pose = options.convergence_handle.threshold_step_size;
    const auto &threshold_huber_loss = options.outlier_handle.threshold_huber_loss;
    const auto &threshold_outlier_reproj_error = options.outlier_handle.threshold_outlier_rejection;
    if (summary != nullptr)
    {
      summary->max_iteration_ = max_iteration;
      summary->threshold_cost_change_ = threshold_convergence_delta_error;
      summary->threshold_step_size_ = threshold_convergence_delta_pose;
      summary->convergence_status_ = true;
    }
    debug_poses_.resize(0);
    if (world_position_list.size() != matched_left_pixel_list.size())
      throw std::runtime_error("In PoseOptimizer::SolveMonocularPoseOnlyBundleAdjustment3Dof(), world_position_list.size() != left_current_pixel_list.size()");
    if (world_position_list.size() != matched_right_pixel_list.size())
      throw std::runtime_error("In PoseOptimizer::SolveMonocularPoseOnlyBundleAdjustment3Dof(), world_position_list.size() != right_current_pixel_list.size()");

    bool is_success = true;

    const size_t n_pts = world_position_list.size();
    mask_inlier_left.resize(n_pts, true);
    mask_inlier_right.resize(n_pts, true);

    const auto &MAX_ITERATION = max_iteration;
    const auto &THRES_HUBER = threshold_huber_loss; // pixels
    const auto &THRES_DELTA_XI = threshold_convergence_delta_pose;
    const auto &THRES_DELTA_ERROR = threshold_convergence_delta_error;
    const auto &THRES_REPROJ_ERROR = threshold_outlier_reproj_error; // pixels

    const Eigen::Isometry3f pose_right_to_left = pose_left_to_right.inverse();
    const Eigen::Isometry3f pose_camera_to_base = pose_base_to_camera.inverse();
    const Eigen::Matrix3f rotation_left_camera_to_base = pose_camera_to_base.linear();
    const Eigen::Matrix3f rotation_right_camera_to_base = pose_right_to_left.linear() * pose_camera_to_base.linear();

    // Warp the world position to the last frame
    const Eigen::Isometry3f pose_c2c1_prior =
        pose_world_to_current.inverse() * pose_world_to_last;
    const Eigen::Isometry3f pose_b2b1 = pose_base_to_camera * pose_c2c1_prior * pose_camera_to_base;

    // calculate prior theta_21 = [x_21, y_21, psi_21]^T
    const auto &R_b2b1 = pose_b2b1.linear();
    const auto &i2 = R_b2b1.block<3, 1>(0, 0);
    const float x_21_initial_value = pose_b2b1.translation().x();
    const float y_21_initial_value = pose_b2b1.translation().y();
    const float psi_21_initial_value = atan2(i2(1), i2(0));

    Eigen::Isometry3f pose_world_to_current_optimized = pose_world_to_current;
    Eigen::Vector3f parameter_b2b1_optimized;
    parameter_b2b1_optimized(0) = x_21_initial_value;
    parameter_b2b1_optimized(1) = y_21_initial_value;
    parameter_b2b1_optimized(2) = psi_21_initial_value;

    Eigen::Isometry3f pose_b2b1_optimized;

    stopwatch.Start();
    bool is_converged = true;
    float err_prev = 1e10f;
    float lambda = 1e-5f;
    std::vector<Eigen::Vector3f> warped_left_position_list;
    std::vector<Eigen::Vector3f> warped_right_position_list;
    for (int iteration = 0; iteration < MAX_ITERATION; ++iteration)
    {
      Eigen::Matrix<float, 3, 3> JtWJ;
      Eigen::Matrix<float, 3, 1> mJtWr;
      JtWJ.setZero();
      mJtWr.setZero();

      // Calculate currently optimized pose
      const float &x_b2b1_optimized = parameter_b2b1_optimized(0);
      const float &y_b2b1_optimized = parameter_b2b1_optimized(1);
      const float &psi_b2b1_optimized = parameter_b2b1_optimized(2);
      const float cos_psi = cos(psi_b2b1_optimized);
      const float sin_psi = sin(psi_b2b1_optimized);

      pose_b2b1_optimized.linear() << cos_psi, -sin_psi, 0,
          sin_psi, cos_psi, 0,
          0, 0, 1;
      pose_b2b1_optimized.translation() << x_b2b1_optimized, y_b2b1_optimized, 0;
      const auto pose_left_c2b1_optimized = pose_camera_to_base * pose_b2b1_optimized;
      const auto pose_right_c2b1_optimized = pose_right_to_left * pose_left_c2b1_optimized;

      // Warp and project 3d point & calculate error
      WarpPointList(pose_left_c2b1_optimized, world_position_list,
                    warped_left_position_list);
      WarpPointList(pose_right_c2b1_optimized, world_position_list,
                    warped_right_position_list);

      float err_curr = 0.0f;
      size_t count_invalid = 0;
      size_t count_left_edge = 0;
      size_t count_right_edge = 0;
      for (size_t index = 0; index < n_pts; ++index)
      {
        const Eigen::Vector2f &matched_left_pixel = matched_left_pixel_list[index];
        const Eigen::Vector2f &matched_right_pixel = matched_right_pixel_list[index];
        const Eigen::Vector3f &world_position = world_position_list[index];

        const Eigen::Vector3f &X_b = world_position;
        const Eigen::Vector3f X_b2 = pose_b2b1_optimized * X_b;
        const Eigen::Vector3f &left_local_position = warped_left_position_list[index];
        const Eigen::Vector3f &right_local_position = warped_right_position_list[index];

        // left
        ++count_left_edge;
        Eigen::Matrix<float, 3, 1> jacobian_transpose_u_left;
        Eigen::Matrix<float, 3, 1> jacobian_transpose_v_left;
        Eigen::Matrix<float, 2, 1> residual_vector_left;
        ComputeJacobianResidual_ReprojectionError_Planar3Dof(
            left_local_position, world_position, matched_left_pixel, fx_left, fy_left, cx_left, cy_left,
            rotation_left_camera_to_base, cos_psi, sin_psi,
            jacobian_transpose_u_left, jacobian_transpose_v_left, residual_vector_left);

        Eigen::Matrix<float, 3, 1> gradient_i_left;
        Eigen::Matrix<float, 3, 3> hessian_i_left;
        float error_i_left = 0;
        float error_nonweighted_i_left = 0;
        ComputeGradientHessian_ReprojectionError_Planar3Dof(
            jacobian_transpose_u_left, jacobian_transpose_v_left, residual_vector_left,
            THRES_HUBER,
            gradient_i_left, hessian_i_left, error_i_left, error_nonweighted_i_left);

        // Add gradient and Hessian to original matrix
        mJtWr.noalias() -= gradient_i_left;
        AddHessianOnlyUpperTriangle_3Dof(hessian_i_left, JtWJ);
        err_curr += error_i_left;

        // Outlier rejection
        if (error_nonweighted_i_left >= THRES_REPROJ_ERROR)
        {
          mask_inlier_left[index] = false;
          ++count_invalid;
        }

        // Check right
        if (matched_right_pixel.x() < 0 || matched_right_pixel.y() < 0)
          continue;

        ++count_right_edge;
        Eigen::Matrix<float, 3, 1> jacobian_transpose_u_right;
        Eigen::Matrix<float, 3, 1> jacobian_transpose_v_right;
        Eigen::Matrix<float, 2, 1> residual_vector_right;
        ComputeJacobianResidual_ReprojectionError_Planar3Dof(
            right_local_position, world_position, matched_right_pixel,
            fx_right, fy_right, cx_right, cy_right,
            rotation_right_camera_to_base, cos_psi, sin_psi,
            jacobian_transpose_u_right, jacobian_transpose_v_right, residual_vector_right);

        // Huber weight calculation by the Manhattan distance
        Eigen::Matrix<float, 3, 3> hessian_i_right;
        Eigen::Matrix<float, 3, 1> gradient_i_right;
        float error_i_right = 0;
        float error_nonweighted_i_right = 0;
        ComputeGradientHessian_ReprojectionError_Planar3Dof(
            jacobian_transpose_u_right, jacobian_transpose_v_right, residual_vector_right,
            THRES_HUBER,
            gradient_i_right, hessian_i_right, error_i_right, error_nonweighted_i_right);

        // Add gradient and Hessian to original matrix
        mJtWr.noalias() -= gradient_i_right;
        AddHessianOnlyUpperTriangle_3Dof(hessian_i_right, JtWJ);
        err_curr += error_i_right;

        // Outlier rejection
        if (error_nonweighted_i_right >= THRES_REPROJ_ERROR)
        {
          mask_inlier_right[index] = false;
          ++count_invalid;
        }
      }

      // Solve H^-1*Jtr;
      FillLowerTriangleByUpperTriangle_3Dof(JtWJ);
      for (size_t i = 0; i < 3; ++i)
        JtWJ(i, i) *= (1.0f + lambda); // lambda

      const Eigen::Matrix<float, 3, 1> delta_param = (JtWJ.ldlt().solve(mJtWr));

      Eigen::Isometry3f delta_pose;
      const float &delta_x = delta_param(0);
      const float &delta_y = delta_param(1);
      const float &delta_psi = delta_param(2);
      delta_pose.linear() << cos(delta_psi), -sin(delta_psi), 0, sin(delta_psi), cos(delta_psi), 0, 0, 0, 1;
      delta_pose.translation() << delta_x, delta_y, 0;
      pose_b2b1_optimized = delta_pose * pose_b2b1_optimized;

      parameter_b2b1_optimized(0) = pose_b2b1_optimized.translation().x();
      parameter_b2b1_optimized(1) = pose_b2b1_optimized.translation().y();
      parameter_b2b1_optimized(2) += delta_psi;

      pose_world_to_current_optimized = pose_b2b1_optimized.inverse() * pose_base_to_camera;
      debug_poses_.push_back(pose_world_to_current_optimized);

      err_curr /= (count_left_edge + count_right_edge) * 0.5f;
      const float delta_error = abs(err_curr - err_prev);

      if (delta_param.norm() < THRES_DELTA_XI || delta_error < THRES_DELTA_ERROR)
      {
        // Early convergence.
        is_converged = true;
        break;
      }
      if (iteration == MAX_ITERATION - 1)
      {
        is_converged = false;
      }
      err_prev = err_curr;

      const double iter_time = stopwatch.GetLapTimeFromLatest();
      if (summary != nullptr)
      {
        OptimizationInfo optimization_info;
        optimization_info.cost = err_curr;
        optimization_info.cost_change = abs(delta_error);
        optimization_info.abs_step = delta_param.norm();
        optimization_info.abs_gradient = 0;
        optimization_info.iter_time = iter_time;

        summary->optimization_info_list_.push_back(optimization_info);
      }
    }

    const double total_time = stopwatch.GetLapTimeFromStart();
    if (summary != nullptr)
    {
      summary->convergence_status_ = is_converged;
      summary->total_time_in_millisecond_ = total_time;
    }

    if (!std::isnan(pose_b2b1_optimized.linear().norm()))
    {
      pose_world_to_current = pose_world_to_current_optimized;
    }
    else
    {
      std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
                << ", pose_21_optimized: \n"
                << pose_b2b1_optimized.linear() << " " << pose_b2b1_optimized.translation() << "\n";
      is_success = false; // if nan, do not update.
    }
    return is_success;
  }

  // NDT minimization
  bool PoseOptimizer::SolveNdtPoseOptimization6Dof(
      const std::vector<Eigen::Vector3f> &query_position_list,
      const std::vector<Eigen::Vector3f> &reference_cell_mu,
      const std::vector<Eigen::Matrix3f> &reference_cell_covariance,
      Eigen::Isometry3f &pose_reference_to_query,
      Summary *summary,
      const int max_iteration,
      const float threshold_outlier_reproj_error,
      const float threshold_huber_loss,
      const float threshold_convergence_delta_pose, const float threshold_convergence_delta_error)
  {
    if (query_position_list.size() != reference_cell_mu.size())
      throw std::runtime_error("query_position_list.size() != reference_cell_mu.size()");
    if (query_position_list.size() != reference_cell_covariance.size())
      throw std::runtime_error("query_position_list.size() != reference_cell_covariance.size()");

    timer::StopWatch stopwatch("SolveNdtPoseOptimization6Dof");
    if (summary != nullptr)
    {
      summary->max_iteration_ = max_iteration;
      summary->threshold_cost_change_ = threshold_convergence_delta_error;
      summary->threshold_step_size_ = threshold_convergence_delta_pose;
      summary->convergence_status_ = true;
    }
    debug_poses_.resize(0);

    bool is_success = true;

    const size_t num_query_points = query_position_list.size();
    const float inverse_num_query_points = 1.0 / num_query_points;

    const size_t MAX_ITER = max_iteration;
    const float THRES_HUBER = threshold_huber_loss; // pixels
    const float THRES_DELTA_XI = threshold_convergence_delta_pose;
    const float THRES_DELTA_ERROR = threshold_convergence_delta_error;
    const float THRES_REPROJ_ERROR = threshold_outlier_reproj_error; // pixels

    Eigen::Isometry3f pose_reference_to_query_optimized = pose_reference_to_query;

    stopwatch.Start();
    bool is_converged = true;
    float err_prev = 1e10f;
    float lambda = 1e-5f;

    Eigen::Matrix<float, 6, 6> JtSJ;
    Eigen::Matrix<float, 6, 1> mJtSr;
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
      JtSJ.setZero();
      mJtSr.setZero();

      // Warp and project 3d point & calculate error
      float err_curr = 0.0f;
      size_t count_invalid = 0;
      for (size_t index = 0; index < num_query_points; ++index)
      {
        const Eigen::Vector3f &query_point = query_position_list[index];

        // Warp the point
        const Eigen::Vector3f rotated_query_point = pose_reference_to_query_optimized.linear() * query_point;
        const Eigen::Vector3f warped_query_point = rotated_query_point + pose_reference_to_query_optimized.translation();

        //  center, information matrix
        const Eigen::Vector3f &cell_center_position = reference_cell_mu[index];
        const Eigen::Matrix3f &cell_inverse_covariance_matrix = reference_cell_covariance[index].inverse();

        // Calculate residual & jacobian
        // residual
        const Eigen::Vector3f d_i = warped_query_point - cell_center_position;
        const Eigen::Vector3f Sd_i = cell_inverse_covariance_matrix * d_i;

        // Jacobian matrix
        Eigen::Matrix<float, 3, 6> J;
        // jacobian_matrix << eye(3,3) , -skew(rotated_query_point)
        J.setZero();
        J(0, 0) = 1;
        J(1, 1) = 1;
        J(2, 2) = 1;
        J(0, 4) = rotated_query_point(2);
        J(0, 5) = -rotated_query_point(1);
        J(1, 5) = rotated_query_point(0);

        J(1, 3) = -rotated_query_point(2);
        J(2, 3) = rotated_query_point(1);
        J(2, 4) = -rotated_query_point(0);

        // Huber weight calculation by the Manhattan distance
        float weight = 1.0f;
        float abs_residual = std::sqrt(d_i.transpose() * Sd_i);
        if (abs_residual >= THRES_HUBER)
        {
          weight = THRES_HUBER / abs_residual;

          Eigen::Matrix<float, 6, 6> JtSJ_tmp;
          const Eigen::Matrix<float, 3, 6> wJ = weight * J;
          this->CalcJtSJ_6Dof(wJ, cell_inverse_covariance_matrix, JtSJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
          AddHessianOnlyUpperTriangle_6Dof(JtSJ_tmp, JtSJ);
          mJtSr.noalias() -= wJ.transpose() * Sd_i;
          err_curr += abs_residual;
        }
        else
        {
          Eigen::Matrix<float, 6, 6> JtSJ_tmp;
          this->CalcJtSJ_6Dof(J, cell_inverse_covariance_matrix, JtSJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
          AddHessianOnlyUpperTriangle_6Dof(JtSJ_tmp, JtSJ);
          mJtSr.noalias() -= J.transpose() * Sd_i;
          err_curr += abs_residual;
        }

        if (abs_residual >= THRES_REPROJ_ERROR)
        {
          // mask_inlier[index] = false;
          ++count_invalid;
        }
      }

      // Solve H^-1*Jtr;
      FillLowerTriangle_6Dof(JtSJ);
      for (size_t i = 0; i < 6; ++i)
        JtSJ(i, i) *= (1.0f + lambda); // lambda

      const Eigen::Matrix<float, 6, 1> delta_param = JtSJ.ldlt().solve(mJtSr); // tx ty tz wx wy wz
      Eigen::Matrix<float, 3, 3> delta_R;
      so3Exp<float>(delta_param.block<3, 1>(3, 0), delta_R);
      Eigen::Isometry3f delta_pose;
      delta_pose.linear() = delta_R;
      delta_pose.translation() = delta_param.block<3, 1>(0, 0);

      // update pose
      pose_reference_to_query_optimized = delta_pose * pose_reference_to_query_optimized;

      debug_poses_.push_back(pose_reference_to_query_optimized);

      err_curr *= inverse_num_query_points;
      const float delta_error = abs(err_curr - err_prev);
      if (delta_param.norm() < THRES_DELTA_XI || delta_error < THRES_DELTA_ERROR)
      {
        is_converged = true;
        // std::cout << "    Early convergence. stops at iteration " << iter << ", error: " << err_curr << ", delta_error: " << delta_error << ", step_size: " << delta_xi.norm() << ", # invalid: " << count_invalid << "\n";
        break;
      }
      if (iter == MAX_ITER - 1)
      {
        is_converged = false;
      }
      err_prev = err_curr;

      const double iter_time = stopwatch.GetLapTimeFromLatest();
      if (summary != nullptr)
      {
        OptimizationInfo optimization_info;
        optimization_info.cost = err_curr;
        optimization_info.cost_change = abs(delta_error);
        optimization_info.abs_step = delta_param.norm();
        optimization_info.abs_gradient = 0;
        optimization_info.iter_time = iter_time;

        summary->optimization_info_list_.push_back(optimization_info);
      }
    }

    const double total_time = stopwatch.GetLapTimeFromStart();
    if (summary != nullptr)
    {
      summary->convergence_status_ = is_converged;
      summary->total_time_in_millisecond_ = total_time;
    }

    if (!std::isnan(pose_reference_to_query_optimized.linear().norm()))
    {
      pose_reference_to_query = pose_reference_to_query_optimized.inverse();
    }
    else
    {
      std::cout << "!! WARNING !! poseonly BA yields NAN value!!"
                << ", pose_reference_to_query_optimized: \n"
                << pose_reference_to_query_optimized.linear() << " " << pose_reference_to_query_optimized.translation() << "\n";
      is_success = false; // if nan, do not update.
    }

    return is_success;
  }

  std::vector<Eigen::Isometry3f> PoseOptimizer::GetDebugPoses()
  {
    return debug_poses_;
  }

  inline void PoseOptimizer::CalcJtJ_x_6Dof(const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp)
  {
    JtJ_tmp.setZero();

    // Product
    // Original : 36 mult
    // Reduced  : 15 mult + 10 insert
    JtJ_tmp(0, 0) = Jt(0) * Jt(0);
    // JtJ_tmp(0,1) = Jt(0)*Jt(1);
    JtJ_tmp(0, 2) = Jt(0) * Jt(2);
    JtJ_tmp(0, 3) = Jt(0) * Jt(3);
    JtJ_tmp(0, 4) = Jt(0) * Jt(4);
    JtJ_tmp(0, 5) = Jt(0) * Jt(5);

    // JtJ_tmp(1,1) = Jt(1)*Jt(1);
    // JtJ_tmp(1,2) = Jt(1)*Jt(2);
    // JtJ_tmp(1,3) = Jt(1)*Jt(3);
    // JtJ_tmp(1,4) = Jt(1)*Jt(4);
    // JtJ_tmp(1,5) = Jt(1)*Jt(5);

    JtJ_tmp(2, 2) = Jt(2) * Jt(2);
    JtJ_tmp(2, 3) = Jt(2) * Jt(3);
    JtJ_tmp(2, 4) = Jt(2) * Jt(4);
    JtJ_tmp(2, 5) = Jt(2) * Jt(5);

    JtJ_tmp(3, 3) = Jt(3) * Jt(3);
    JtJ_tmp(3, 4) = Jt(3) * Jt(4);
    JtJ_tmp(3, 5) = Jt(3) * Jt(5);

    JtJ_tmp(4, 4) = Jt(4) * Jt(4);
    JtJ_tmp(4, 5) = Jt(4) * Jt(5);

    JtJ_tmp(5, 5) = Jt(5) * Jt(5);

    // Filling symmetric elements
    // JtJ_tmp(1,0) = JtJ_tmp(0,1);
    JtJ_tmp(2, 0) = JtJ_tmp(0, 2);
    JtJ_tmp(3, 0) = JtJ_tmp(0, 3);
    JtJ_tmp(4, 0) = JtJ_tmp(0, 4);
    JtJ_tmp(5, 0) = JtJ_tmp(0, 5);

    // JtJ_tmp(2,1) = JtJ_tmp(1,2);
    // JtJ_tmp(3,1) = JtJ_tmp(1,3);
    // JtJ_tmp(4,1) = JtJ_tmp(1,4);
    // JtJ_tmp(5,1) = JtJ_tmp(1,5);

    JtJ_tmp(3, 2) = JtJ_tmp(2, 3);
    JtJ_tmp(4, 2) = JtJ_tmp(2, 4);
    JtJ_tmp(5, 2) = JtJ_tmp(2, 5);

    JtJ_tmp(4, 3) = JtJ_tmp(3, 4);
    JtJ_tmp(5, 3) = JtJ_tmp(3, 5);

    JtJ_tmp(5, 4) = JtJ_tmp(4, 5);
  }

  inline void PoseOptimizer::CalcJtJ_y_6Dof(const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp)
  {
    JtJ_tmp.setZero();

    // Product
    // Original : 36 + 36 mult
    // Reduced  : 6 + 15 mult + 10 insert
    // JtJ_tmp(0,0) = Jt(0)*Jt(0);
    // JtJ_tmp(0,1) = Jt(0)*Jt(1);
    // JtJ_tmp(0,2) = Jt(0)*Jt(2);
    // JtJ_tmp(0,3) = Jt(0)*Jt(3);
    // JtJ_tmp(0,4) = Jt(0)*Jt(4);
    // JtJ_tmp(0,5) = Jt(0)*Jt(5);

    JtJ_tmp(1, 1) = Jt(1) * Jt(1);
    JtJ_tmp(1, 2) = Jt(1) * Jt(2);
    JtJ_tmp(1, 3) = Jt(1) * Jt(3);
    JtJ_tmp(1, 4) = Jt(1) * Jt(4);
    JtJ_tmp(1, 5) = Jt(1) * Jt(5);

    JtJ_tmp(2, 2) = Jt(2) * Jt(2);
    JtJ_tmp(2, 3) = Jt(2) * Jt(3);
    JtJ_tmp(2, 4) = Jt(2) * Jt(4);
    JtJ_tmp(2, 5) = Jt(2) * Jt(5);

    JtJ_tmp(3, 3) = Jt(3) * Jt(3);
    JtJ_tmp(3, 4) = Jt(3) * Jt(4);
    JtJ_tmp(3, 5) = Jt(3) * Jt(5);

    JtJ_tmp(4, 4) = Jt(4) * Jt(4);
    JtJ_tmp(4, 5) = Jt(4) * Jt(5);

    JtJ_tmp(5, 5) = Jt(5) * Jt(5);

    // Filling symmetric elements
    // JtJ_tmp(1,0) = JtJ_tmp(0,1);
    // JtJ_tmp(2,0) = JtJ_tmp(0,2);
    // JtJ_tmp(3,0) = JtJ_tmp(0,3);
    // JtJ_tmp(4,0) = JtJ_tmp(0,4);
    // JtJ_tmp(5,0) = JtJ_tmp(0,5);

    JtJ_tmp(2, 1) = JtJ_tmp(1, 2);
    JtJ_tmp(3, 1) = JtJ_tmp(1, 3);
    JtJ_tmp(4, 1) = JtJ_tmp(1, 4);
    JtJ_tmp(5, 1) = JtJ_tmp(1, 5);

    JtJ_tmp(3, 2) = JtJ_tmp(2, 3);
    JtJ_tmp(4, 2) = JtJ_tmp(2, 4);
    JtJ_tmp(5, 2) = JtJ_tmp(2, 5);

    JtJ_tmp(4, 3) = JtJ_tmp(3, 4);
    JtJ_tmp(5, 3) = JtJ_tmp(3, 5);

    JtJ_tmp(5, 4) = JtJ_tmp(4, 5);
  }

  inline void PoseOptimizer::CalcJtWJ_x_6Dof(const float weight, const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp)
  {
    JtJ_tmp.setZero();

    Eigen::Matrix<float, 6, 1> wJt;
    wJt.setZero();

    // Product the weight
    wJt << weight * Jt(0), weight * Jt(1), weight * Jt(2), weight * Jt(3), weight * Jt(4), weight * Jt(5);

    // Product
    // Original : 36 + 36 mult
    // Reduced  : 6 + 15 mult + 10 insert
    JtJ_tmp(0, 0) = wJt(0) * Jt(0);
    // JtJ_tmp(0,1) = weight*Jt(0)*Jt(1);
    JtJ_tmp(0, 2) = wJt(0) * Jt(2);
    JtJ_tmp(0, 3) = wJt(0) * Jt(3);
    JtJ_tmp(0, 4) = wJt(0) * Jt(4);
    JtJ_tmp(0, 5) = wJt(0) * Jt(5);

    // JtJ_tmp(1,1) = weight*Jt(1)*Jt(1);
    // JtJ_tmp(1,2) = weight*Jt(1)*Jt(2);
    // JtJ_tmp(1,3) = weight*Jt(1)*Jt(3);
    // JtJ_tmp(1,4) = weight*Jt(1)*Jt(4);
    // JtJ_tmp(1,5) = weight*Jt(1)*Jt(5);

    JtJ_tmp(2, 2) = wJt(2) * Jt(2);
    JtJ_tmp(2, 3) = wJt(2) * Jt(3);
    JtJ_tmp(2, 4) = wJt(2) * Jt(4);
    JtJ_tmp(2, 5) = wJt(2) * Jt(5);

    JtJ_tmp(3, 3) = wJt(3) * Jt(3);
    JtJ_tmp(3, 4) = wJt(3) * Jt(4);
    JtJ_tmp(3, 5) = wJt(3) * Jt(5);

    JtJ_tmp(4, 4) = wJt(4) * Jt(4);
    JtJ_tmp(4, 5) = wJt(4) * Jt(5);

    JtJ_tmp(5, 5) = wJt(5) * Jt(5);

    // Filling symmetric elements
    // JtJ_tmp(1,0) = JtJ_tmp(0,1);
    JtJ_tmp(2, 0) = JtJ_tmp(0, 2);
    JtJ_tmp(3, 0) = JtJ_tmp(0, 3);
    JtJ_tmp(4, 0) = JtJ_tmp(0, 4);
    JtJ_tmp(5, 0) = JtJ_tmp(0, 5);

    // JtJ_tmp(2,1) = JtJ_tmp(1,2);
    // JtJ_tmp(3,1) = JtJ_tmp(1,3);
    // JtJ_tmp(4,1) = JtJ_tmp(1,4);
    // JtJ_tmp(5,1) = JtJ_tmp(1,5);

    JtJ_tmp(3, 2) = JtJ_tmp(2, 3);
    JtJ_tmp(4, 2) = JtJ_tmp(2, 4);
    JtJ_tmp(5, 2) = JtJ_tmp(2, 5);

    JtJ_tmp(4, 3) = JtJ_tmp(3, 4);
    JtJ_tmp(5, 3) = JtJ_tmp(3, 5);

    JtJ_tmp(5, 4) = JtJ_tmp(4, 5);
  }

  inline void PoseOptimizer::CalcJtWJ_y_6Dof(const float weight, const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp)
  {
    JtJ_tmp.setZero();

    Eigen::Matrix<float, 6, 1> wJt;
    wJt.setZero();

    // Product the weight
    wJt << weight * Jt(0), weight * Jt(1), weight * Jt(2), weight * Jt(3), weight * Jt(4), weight * Jt(5);

    // Product
    // Original : 36 + 36 mult
    // Reduced  : 6 + 15 mult + 10 insert
    // JtJ_tmp(0,0) = wJt(0)*Jt(0);
    // JtJ_tmp(0,1) = wJt(0)*Jt(1);
    // JtJ_tmp(0,2) = wJt(0)*Jt(2);
    // JtJ_tmp(0,3) = wJt(0)*Jt(3);
    // JtJ_tmp(0,4) = wJt(0)*Jt(4);
    // JtJ_tmp(0,5) = wJt(0)*Jt(5);

    JtJ_tmp(1, 1) = wJt(1) * Jt(1);
    JtJ_tmp(1, 2) = wJt(1) * Jt(2);
    JtJ_tmp(1, 3) = wJt(1) * Jt(3);
    JtJ_tmp(1, 4) = wJt(1) * Jt(4);
    JtJ_tmp(1, 5) = wJt(1) * Jt(5);

    JtJ_tmp(2, 2) = wJt(2) * Jt(2);
    JtJ_tmp(2, 3) = wJt(2) * Jt(3);
    JtJ_tmp(2, 4) = wJt(2) * Jt(4);
    JtJ_tmp(2, 5) = wJt(2) * Jt(5);

    JtJ_tmp(3, 3) = wJt(3) * Jt(3);
    JtJ_tmp(3, 4) = wJt(3) * Jt(4);
    JtJ_tmp(3, 5) = wJt(3) * Jt(5);

    JtJ_tmp(4, 4) = wJt(4) * Jt(4);
    JtJ_tmp(4, 5) = wJt(4) * Jt(5);

    JtJ_tmp(5, 5) = wJt(5) * Jt(5);

    // Filling symmetric elements
    // JtJ_tmp(1,0) = JtJ_tmp(0,1);
    // JtJ_tmp(2,0) = JtJ_tmp(0,2);
    // JtJ_tmp(3,0) = JtJ_tmp(0,3);
    // JtJ_tmp(4,0) = JtJ_tmp(0,4);
    // JtJ_tmp(5,0) = JtJ_tmp(0,5);

    JtJ_tmp(2, 1) = JtJ_tmp(1, 2);
    JtJ_tmp(3, 1) = JtJ_tmp(1, 3);
    JtJ_tmp(4, 1) = JtJ_tmp(1, 4);
    JtJ_tmp(5, 1) = JtJ_tmp(1, 5);

    JtJ_tmp(3, 2) = JtJ_tmp(2, 3);
    JtJ_tmp(4, 2) = JtJ_tmp(2, 4);
    JtJ_tmp(5, 2) = JtJ_tmp(2, 5);

    JtJ_tmp(4, 3) = JtJ_tmp(3, 4);
    JtJ_tmp(5, 3) = JtJ_tmp(3, 5);

    JtJ_tmp(5, 4) = JtJ_tmp(4, 5);
  }

  inline void PoseOptimizer::AddHessianOnlyUpperTriangle_6Dof(const Eigen::Matrix<float, 6, 6> &JtJ_tmp, Eigen::Matrix<float, 6, 6> &JtJ)
  {
    JtJ(0, 0) += JtJ_tmp(0, 0);
    JtJ(0, 1) += JtJ_tmp(0, 1);
    JtJ(0, 2) += JtJ_tmp(0, 2);
    JtJ(0, 3) += JtJ_tmp(0, 3);
    JtJ(0, 4) += JtJ_tmp(0, 4);
    JtJ(0, 5) += JtJ_tmp(0, 5);

    JtJ(1, 1) += JtJ_tmp(1, 1);
    JtJ(1, 2) += JtJ_tmp(1, 2);
    JtJ(1, 3) += JtJ_tmp(1, 3);
    JtJ(1, 4) += JtJ_tmp(1, 4);
    JtJ(1, 5) += JtJ_tmp(1, 5);

    JtJ(2, 2) += JtJ_tmp(2, 2);
    JtJ(2, 3) += JtJ_tmp(2, 3);
    JtJ(2, 4) += JtJ_tmp(2, 4);
    JtJ(2, 5) += JtJ_tmp(2, 5);

    JtJ(3, 3) += JtJ_tmp(3, 3);
    JtJ(3, 4) += JtJ_tmp(3, 4);
    JtJ(3, 5) += JtJ_tmp(3, 5);

    JtJ(4, 4) += JtJ_tmp(4, 4);
    JtJ(4, 5) += JtJ_tmp(4, 5);

    JtJ(5, 5) += JtJ_tmp(5, 5);
  }

  inline void PoseOptimizer::FillLowerTriangle_6Dof(Eigen::Matrix<float, 6, 6> &JtJ)
  {
    JtJ(1, 0) = JtJ(0, 1);
    JtJ(2, 0) = JtJ(0, 2);
    JtJ(3, 0) = JtJ(0, 3);
    JtJ(4, 0) = JtJ(0, 4);
    JtJ(5, 0) = JtJ(0, 5);

    JtJ(2, 1) = JtJ(1, 2);
    JtJ(3, 1) = JtJ(1, 3);
    JtJ(4, 1) = JtJ(1, 4);
    JtJ(5, 1) = JtJ(1, 5);

    JtJ(3, 2) = JtJ(2, 3);
    JtJ(4, 2) = JtJ(2, 4);
    JtJ(5, 2) = JtJ(2, 5);

    JtJ(4, 3) = JtJ(3, 4);
    JtJ(5, 3) = JtJ(3, 5);

    JtJ(5, 4) = JtJ(4, 5);
  }

  inline void PoseOptimizer::CalcJtJ_3Dof(const Eigen::Matrix<float, 3, 1> &Jt, Eigen::Matrix<float, 3, 3> &JtJ_tmp)
  {
    JtJ_tmp.setZero();

    // Product
    // Original : 36 mult
    // Reduced  : 15 mult + 10 insert
    JtJ_tmp(0, 0) = Jt(0) * Jt(0);
    JtJ_tmp(0, 1) = Jt(0) * Jt(1);
    JtJ_tmp(0, 2) = Jt(0) * Jt(2);

    JtJ_tmp(1, 1) = Jt(1) * Jt(1);
    JtJ_tmp(1, 2) = Jt(1) * Jt(2);

    JtJ_tmp(2, 2) = Jt(2) * Jt(2);

    // Filling symmetric elements
    JtJ_tmp(1, 0) = JtJ_tmp(0, 1);

    JtJ_tmp(2, 0) = JtJ_tmp(0, 2);
    JtJ_tmp(2, 1) = JtJ_tmp(1, 2);
  }

  inline void PoseOptimizer::CalcJtWJ_3Dof(const float weight, const Eigen::Matrix<float, 3, 1> &Jt, Eigen::Matrix<float, 3, 3> &JtJ_tmp)
  {
    JtJ_tmp.setZero();

    Eigen::Matrix<float, 3, 1> wJt;
    wJt.setZero();

    // Product the weight
    wJt << weight * Jt(0), weight * Jt(1), weight * Jt(2);

    // Product
    // Original : 36 + 36 mult
    // Reduced  : 6 + 15 mult + 10 insert
    JtJ_tmp(0, 0) = wJt(0) * Jt(0);
    JtJ_tmp(0, 1) = wJt(0) * Jt(1);
    JtJ_tmp(0, 2) = wJt(0) * Jt(2);

    JtJ_tmp(1, 1) = wJt(1) * Jt(1);
    JtJ_tmp(1, 2) = wJt(1) * Jt(2);

    JtJ_tmp(2, 2) = wJt(2) * Jt(2);

    // Filling symmetric elements
    JtJ_tmp(1, 0) = JtJ_tmp(0, 1);
    JtJ_tmp(2, 0) = JtJ_tmp(0, 2);

    JtJ_tmp(2, 1) = JtJ_tmp(1, 2);
  }

  inline void PoseOptimizer::AddHessianOnlyUpperTriangle_3Dof(const Eigen::Matrix<float, 3, 3> &JtJ_tmp, Eigen::Matrix<float, 3, 3> &JtJ)
  {
    JtJ(0, 0) += JtJ_tmp(0, 0);
    JtJ(0, 1) += JtJ_tmp(0, 1);
    JtJ(0, 2) += JtJ_tmp(0, 2);

    JtJ(1, 1) += JtJ_tmp(1, 1);
    JtJ(1, 2) += JtJ_tmp(1, 2);

    JtJ(2, 2) += JtJ_tmp(2, 2);

    FillLowerTriangleByUpperTriangle_3Dof(JtJ);
  }

  inline void PoseOptimizer::FillLowerTriangleByUpperTriangle_3Dof(Eigen::Matrix<float, 3, 3> &JtJ)
  {
    JtJ(1, 0) = JtJ(0, 1);
    JtJ(2, 0) = JtJ(0, 2);

    JtJ(2, 1) = JtJ(1, 2);
  }

  inline void PoseOptimizer::CalcJtSJ_6Dof(const Eigen::Matrix<float, 3, 6> &J, const Eigen::Matrix<float, 3, 3> &S, Eigen::Matrix<float, 6, 6> &JtSJ)
  {
    // JtSJ.setZero();
    Eigen::Matrix<float, 3, 6> SJ;
    // SJ.setZero();
    SJ = S * J; // 54 mult, 36 add

    // Product
    // Original : 108 mult, 72 add
    // Reduced  : 63 mult + 42 add + 16 insert
#define Jti_mult_SJj(J, SJ, i, j) (J(i, 0) * SJ(0, j) + J(i, 1) * SJ(1, j) + J(i, 2) * SJ(2, j))
    JtSJ(0, 0) = Jti_mult_SJj(J, SJ, 0, 0);
    JtSJ(0, 1) = Jti_mult_SJj(J, SJ, 0, 1);
    JtSJ(0, 2) = Jti_mult_SJj(J, SJ, 0, 2);
    JtSJ(0, 3) = Jti_mult_SJj(J, SJ, 0, 3);
    JtSJ(0, 4) = Jti_mult_SJj(J, SJ, 0, 4);
    JtSJ(0, 5) = Jti_mult_SJj(J, SJ, 0, 5);

    JtSJ(1, 1) = Jti_mult_SJj(J, SJ, 1, 1);
    JtSJ(1, 2) = Jti_mult_SJj(J, SJ, 1, 2);
    JtSJ(1, 3) = Jti_mult_SJj(J, SJ, 1, 3);
    JtSJ(1, 4) = Jti_mult_SJj(J, SJ, 1, 4);
    JtSJ(1, 5) = Jti_mult_SJj(J, SJ, 1, 5);

    JtSJ(2, 2) = Jti_mult_SJj(J, SJ, 2, 2);
    JtSJ(2, 3) = Jti_mult_SJj(J, SJ, 2, 3);
    JtSJ(2, 4) = Jti_mult_SJj(J, SJ, 2, 4);
    JtSJ(2, 5) = Jti_mult_SJj(J, SJ, 2, 5);

    JtSJ(3, 3) = Jti_mult_SJj(J, SJ, 3, 3);
    JtSJ(3, 4) = Jti_mult_SJj(J, SJ, 3, 4);
    JtSJ(3, 5) = Jti_mult_SJj(J, SJ, 3, 5);

    JtSJ(4, 4) = Jti_mult_SJj(J, SJ, 4, 4);
    JtSJ(4, 5) = Jti_mult_SJj(J, SJ, 4, 5);

    JtSJ(5, 5) = Jti_mult_SJj(J, SJ, 5, 5);
#undef Jti_mult_SJj

    // JtSJ(0,0) = J.block<3,1>(0,0).transpose()*SJ.block<3,1>(0,0);
    // JtSJ(0,1) = J.block<3,1>(0,0).transpose()*SJ.block<3,1>(0,1);
    // JtSJ(0,2) = J.block<3,1>(0,0).transpose()*SJ.block<3,1>(0,2);
    // JtSJ(0,3) = J.block<3,1>(0,0).transpose()*SJ.block<3,1>(0,3);
    // JtSJ(0,4) = J.block<3,1>(0,0).transpose()*SJ.block<3,1>(0,4);
    // JtSJ(0,5) = J.block<3,1>(0,0).transpose()*SJ.block<3,1>(0,5);

    // JtSJ(1,1) = J.block<3,1>(0,1).transpose()*SJ.block<3,1>(0,1);
    // JtSJ(1,2) = J.block<3,1>(0,1).transpose()*SJ.block<3,1>(0,2);
    // JtSJ(1,3) = J.block<3,1>(0,1).transpose()*SJ.block<3,1>(0,3);
    // JtSJ(1,4) = J.block<3,1>(0,1).transpose()*SJ.block<3,1>(0,4);
    // JtSJ(1,5) = J.block<3,1>(0,1).transpose()*SJ.block<3,1>(0,5);

    // JtSJ(2,2) = J.block<3,1>(0,2).transpose()*SJ.block<3,1>(0,2);
    // JtSJ(2,3) = J.block<3,1>(0,2).transpose()*SJ.block<3,1>(0,3);
    // JtSJ(2,4) = J.block<3,1>(0,2).transpose()*SJ.block<3,1>(0,4);
    // JtSJ(2,5) = J.block<3,1>(0,2).transpose()*SJ.block<3,1>(0,5);

    // JtSJ(3,3) = J.block<3,1>(0,3).transpose()*SJ.block<3,1>(0,3);
    // JtSJ(3,4) = J.block<3,1>(0,3).transpose()*SJ.block<3,1>(0,4);
    // JtSJ(3,5) = J.block<3,1>(0,3).transpose()*SJ.block<3,1>(0,5);

    // JtSJ(4,4) = J.block<3,1>(0,4).transpose()*SJ.block<3,1>(0,4);
    // JtSJ(4,5) = J.block<3,1>(0,4).transpose()*SJ.block<3,1>(0,5);

    // JtSJ(5,5) = J.block<3,1>(0,5).transpose()*SJ.block<3,1>(0,5);

    // Filling symmetric elements
    JtSJ(1, 0) = JtSJ(0, 1);
    JtSJ(2, 0) = JtSJ(0, 2);
    JtSJ(3, 0) = JtSJ(0, 3);
    JtSJ(4, 0) = JtSJ(0, 4);
    JtSJ(5, 0) = JtSJ(0, 5);

    JtSJ(2, 1) = JtSJ(1, 2);
    JtSJ(3, 1) = JtSJ(1, 3);
    JtSJ(4, 1) = JtSJ(1, 4);
    JtSJ(5, 1) = JtSJ(1, 5);

    JtSJ(3, 2) = JtSJ(2, 3);
    JtSJ(4, 2) = JtSJ(2, 4);
    JtSJ(5, 2) = JtSJ(2, 5);

    JtSJ(4, 3) = JtSJ(3, 4);
    JtSJ(5, 3) = JtSJ(3, 5);

    JtSJ(5, 4) = JtSJ(4, 5);
  }

  template <typename T>
  void PoseOptimizer::se3Exp(const Eigen::Matrix<T, 6, 1> &xi, Eigen::Transform<T, 3, 1> &pose)
  {
    // initialize variables
    T theta = 0.0;
    Eigen::Matrix<T, 3, 1> v, w;
    Eigen::Matrix<T, 3, 3> wx, R, V;
    Eigen::Matrix<T, 3, 1> t;

    v(0) = xi(0);
    v(1) = xi(1);
    v(2) = xi(2);

    w(0) = xi(3);
    w(1) = xi(4);
    w(2) = xi(5);

    theta = std::sqrt(w.transpose() * w);
    wx << 0, -w(2), w(1),
        w(2), 0, -w(0),
        -w(1), w(0), 0;

    if (theta < 1e-7)
    {
      R = Eigen::Matrix<T, 3, 3>::Identity() + wx + 0.5 * wx * wx;
      V = Eigen::Matrix<T, 3, 3>::Identity() + 0.5 * wx + wx * wx * 0.33333333333333333333333333;
    }
    else
    {
      R = Eigen::Matrix<T, 3, 3>::Identity() + (sin(theta) / theta) * wx + ((1 - cos(theta)) / (theta * theta)) * (wx * wx);
      V = Eigen::Matrix<T, 3, 3>::Identity() + ((1 - cos(theta)) / (theta * theta)) * wx + ((theta - sin(theta)) / (theta * theta * theta)) * (wx * wx);
    }
    t = V * v;

    // assign rigid body transformation matrix (in SE(3))
    pose.linear() = R;
    pose.translation() = t;
  }

  template <typename T>
  void PoseOptimizer::so3Exp(const Eigen::Matrix<T, 3, 1> &w, Eigen::Matrix<T, 3, 3> &R)
  {
    // initialize variables
    T theta = 0.0;
    Eigen::Matrix<T, 3, 3> wx;

    theta = std::sqrt(w.transpose() * w);
    wx << 0, -w(2), w(1),
        w(2), 0, -w(0),
        -w(1), w(0), 0;

    Eigen::Matrix<T, 3, 3> wxwx = wx * wx;
    if (theta < 1e-9)
    {
      R = Eigen::Matrix<T, 3, 3>::Identity() + wx + 0.5 * wxwx;
    }
    else
    {
      double invtheta2 = 1.0 / (theta * theta);
      R = Eigen::Matrix<T, 3, 3>::Identity() + (sin(theta) / theta) * wx + ((1 - cos(theta)) * invtheta2) * wxwx;
    }
  }

  inline void PoseOptimizer::WarpPointList(
      const Eigen::Isometry3f &pose_target_to_initial,
      const std::vector<Eigen::Vector3f> &initial_position_list,
      std::vector<Eigen::Vector3f> &warped_target_position_list)
  {
    const size_t num_data = initial_position_list.size();
    warped_target_position_list.reserve(num_data);
    warped_target_position_list.resize(num_data);
    for (size_t index = 0; index < num_data; ++index)
    {
      const auto &initial_position = initial_position_list[index];
      warped_target_position_list[index] = pose_target_to_initial * initial_position;
    }
  }

  inline void PoseOptimizer::ComputeJacobianResidual_ReprojectionError_6Dof(
      const Eigen::Vector3f &local_position,
      const Eigen::Vector2f &matched_pixel,
      const float fx, const float fy, const float cx, const float cy,
      Eigen::Matrix<float, 6, 1> &jacobian_matrix_x_transpose,
      Eigen::Matrix<float, 6, 1> &jacobian_matrix_y_transpose,
      Eigen::Matrix<float, 2, 1> &residual_vector)
  {
    const auto inverse_z = 1.0f / local_position(2);
    const auto x_inverse_z = local_position(0) * inverse_z;
    const auto y_inverse_z = local_position(1) * inverse_z;
    const auto fx_x_inverse_z = fx * x_inverse_z;
    const auto fy_y_inverse_z = fy * y_inverse_z;

    const auto projected_u = fx_x_inverse_z + cx;
    const auto projected_v = fy_y_inverse_z + cy;

    // residual for u and v
    residual_vector(0) = projected_u - matched_pixel.x();
    residual_vector(1) = projected_v - matched_pixel.y();

    // JtWJ, JtWr for u and v
    jacobian_matrix_x_transpose(0) = fx * inverse_z;
    jacobian_matrix_x_transpose(1) = 0.0f;
    jacobian_matrix_x_transpose(2) = -fx_x_inverse_z * inverse_z;
    jacobian_matrix_x_transpose(3) = -fx_x_inverse_z * y_inverse_z;
    jacobian_matrix_x_transpose(4) = fx * (1.0f + x_inverse_z * x_inverse_z);
    jacobian_matrix_x_transpose(5) = -fx * y_inverse_z;

    jacobian_matrix_y_transpose(0) = 0.0f;
    jacobian_matrix_y_transpose(1) = fy * inverse_z;
    jacobian_matrix_y_transpose(2) = -fy_y_inverse_z * inverse_z;
    jacobian_matrix_y_transpose(3) = -fy * (1.0f + y_inverse_z * y_inverse_z);
    jacobian_matrix_y_transpose(4) = fy_y_inverse_z * x_inverse_z;
    jacobian_matrix_y_transpose(5) = fy * x_inverse_z;
  }

  inline void PoseOptimizer::ComputeGradientHessian_ReprojectionError_6Dof(
      const Eigen::Matrix<float, 6, 1> &jacobian_matrix_u_transpose,
      const Eigen::Matrix<float, 6, 1> &jacobian_matrix_v_transpose,
      const Eigen::Matrix<float, 2, 1> &residual_vector,
      const float &threshold_huber,
      Eigen::Matrix<float, 6, 1> &gradient_vector,
      Eigen::Matrix<float, 6, 6> &hessian_matrix,
      float &error,
      float &error_nonweighted)
  {
    const auto &Jt_u = jacobian_matrix_u_transpose;
    const auto &Jt_v = jacobian_matrix_v_transpose;
    auto &JtWr = gradient_vector;
    auto &JtWJ = hessian_matrix;
    JtWr.setZero();
    JtWJ.setZero();
    error = 0.0f;

    // Huber weight calculation by the Manhattan distance
    float weight = 1.0f;
    const auto abs_residual_sum = abs(residual_vector(0)) + abs(residual_vector(1));
    error_nonweighted = abs_residual_sum;
    const auto &residual_u = residual_vector(0);
    const auto &residual_v = residual_vector(1);
    if (abs_residual_sum >= threshold_huber)
    {
      weight = threshold_huber / abs_residual_sum;

      Eigen::Matrix<float, 6, 6> JtJ_tmp;
      JtJ_tmp.setZero();
      const auto weight_residual_u = weight * residual_u;
      const auto error_u = weight_residual_u * residual_u;
      this->CalcJtWJ_x_6Dof(weight, Jt_u, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_6Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += weight_residual_u * Jt_u;

      JtJ_tmp.setZero();
      const auto weight_residual_v = weight * residual_v;
      const auto error_v = weight_residual_v * residual_v;
      this->CalcJtWJ_y_6Dof(weight, Jt_v, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_6Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += weight_residual_v * Jt_v;
      error += error_u;
    }
    else
    {
      Eigen::Matrix<float, 6, 6> JtJ_tmp;
      JtJ_tmp.setZero();
      const auto error_u = residual_u * residual_u;
      this->CalcJtJ_x_6Dof(Jt_u, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_6Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += residual_u * Jt_u;

      JtJ_tmp.setZero();
      const auto error_v = residual_v * residual_v;
      this->CalcJtJ_y_6Dof(Jt_v, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_6Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += residual_v * Jt_v;
      error += error_v;
    }
  }

  inline void PoseOptimizer::ComputeJacobianResidual_ReprojectionError_Planar3Dof(
      const Eigen::Vector3f &target_local_position,
      const Eigen::Vector3f &world_position,
      const Eigen::Vector2f &matched_pixel,
      const float fx, const float fy, const float cx, const float cy,
      const Eigen::Matrix<float, 3, 3> &rotation_target_camera_to_base,
      const float cos_psi, const float sin_psi,
      Eigen::Matrix<float, 3, 1> &jacobian_matrix_u_transpose,
      Eigen::Matrix<float, 3, 1> &jacobian_matrix_v_transpose,
      Eigen::Matrix<float, 2, 1> &residual_vector)
  {
    const auto &r11 = rotation_target_camera_to_base(0, 0);
    const auto &r12 = rotation_target_camera_to_base(0, 1);
    const auto &r21 = rotation_target_camera_to_base(1, 0);
    const auto &r22 = rotation_target_camera_to_base(1, 1);
    const auto &r31 = rotation_target_camera_to_base(2, 0);
    const auto &r32 = rotation_target_camera_to_base(2, 1);

    const auto inverse_z = 1.0f / target_local_position(2);
    const auto x_inverse_z = target_local_position(0) * inverse_z;
    const auto y_inverse_z = target_local_position(1) * inverse_z;
    const auto fx_x_inverse_z = fx * x_inverse_z;
    const auto fy_y_inverse_z = fy * y_inverse_z;

    const auto projected_u = fx_x_inverse_z + cx;
    const auto projected_v = fy_y_inverse_z + cy;

    // residual for u and v
    residual_vector(0) = projected_u - matched_pixel.x();
    residual_vector(1) = projected_v - matched_pixel.y();

    // JtWJ, JtWr for u and v
    const auto alpha_1 = fx * inverse_z;
    const auto alpha_2 = -fx_x_inverse_z * inverse_z;
    const auto beta_1 = fy * inverse_z;
    const auto beta_2 = -fy_y_inverse_z * inverse_z;

    const auto &xb = world_position(0);
    const auto &yb = world_position(1);
    const auto A = -sin_psi * xb - cos_psi * yb;
    const auto B = cos_psi * xb - sin_psi * yb;

    const auto alpha_1_r11 = alpha_1 * r11;
    const auto alpha_2_r31 = alpha_2 * r31;
    const auto alpha_1_r12 = alpha_1 * r12;
    const auto alpha_2_r32 = alpha_2 * r32;

    const auto beta_1_r21 = beta_1 * r21;
    const auto beta_2_r31 = beta_2 * r31;
    const auto beta_1_r22 = beta_1 * r22;
    const auto beta_2_r32 = beta_2 * r32;

    jacobian_matrix_u_transpose(0) = alpha_1_r11 + alpha_2_r31;
    jacobian_matrix_u_transpose(1) = alpha_1_r12 + alpha_2_r32;
    jacobian_matrix_u_transpose(2) = jacobian_matrix_u_transpose(0) * A + jacobian_matrix_u_transpose(1) * B;

    jacobian_matrix_v_transpose(0) = beta_1_r21 + beta_2_r31;
    jacobian_matrix_v_transpose(1) = beta_1_r22 + beta_2_r32;
    jacobian_matrix_v_transpose(2) = jacobian_matrix_v_transpose(0) * A + jacobian_matrix_v_transpose(1) * B;
  }
  inline void PoseOptimizer::ComputeGradientHessian_ReprojectionError_Planar3Dof(
      const Eigen::Matrix<float, 3, 1> &jacobian_matrix_u_transpose,
      const Eigen::Matrix<float, 3, 1> &jacobian_matrix_v_transpose,
      const Eigen::Matrix<float, 2, 1> &residual_vector,
      const float &threshold_huber,
      Eigen::Matrix<float, 3, 1> &gradient_vector,
      Eigen::Matrix<float, 3, 3> &hessian_matrix,
      float &error,
      float &error_nonweighted)
  {
    const auto &Jt_u = jacobian_matrix_u_transpose;
    const auto &Jt_v = jacobian_matrix_v_transpose;
    auto &JtWr = gradient_vector;
    auto &JtWJ = hessian_matrix;
    JtWr.setZero();
    JtWJ.setZero();
    error = 0.0f;

    // Huber weight calculation by the Manhattan distance
    float weight = 1.0f;
    const auto abs_residual_sum = abs(residual_vector(0)) + abs(residual_vector(1));
    error_nonweighted = abs_residual_sum;
    const auto &residual_u = residual_vector(0);
    const auto &residual_v = residual_vector(1);
    if (abs_residual_sum >= threshold_huber)
    {
      weight = threshold_huber / abs_residual_sum;

      Eigen::Matrix<float, 3, 3> JtJ_tmp;
      JtJ_tmp.setZero();
      const auto weight_residual_u = weight * residual_u;
      const auto error_u = weight_residual_u * residual_u;
      this->CalcJtWJ_3Dof(weight, Jt_u, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_3Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += weight_residual_u * Jt_u;

      JtJ_tmp.setZero();
      const auto weight_residual_v = weight * residual_v;
      const auto error_v = weight_residual_v * residual_v;
      this->CalcJtWJ_3Dof(weight, Jt_v, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_3Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += weight_residual_v * Jt_v;
      error += error_u;
    }
    else
    {
      Eigen::Matrix<float, 3, 3> JtJ_tmp;
      JtJ_tmp.setZero();
      const auto error_u = residual_u * residual_u;
      this->CalcJtJ_3Dof(Jt_u, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_3Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += residual_u * Jt_u;

      JtJ_tmp.setZero();
      const auto error_v = residual_v * residual_v;
      this->CalcJtJ_3Dof(Jt_v, JtJ_tmp); // JtWJ.noalias()  += weight*(Jt*Jt.transpose());
      // JtWJ.noalias() += JtJ_tmp;
      AddHessianOnlyUpperTriangle_3Dof(JtJ_tmp, JtWJ);
      JtWr.noalias() += residual_v * Jt_v;
      error += error_v;
    }
  }
};
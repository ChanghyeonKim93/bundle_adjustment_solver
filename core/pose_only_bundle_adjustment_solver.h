#ifndef _POSE_ONLY_BUNDLE_ADJUSTMENT_H_
#define _POSE_ONLY_BUNDLE_ADJUSTMENT_H_

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ios>
#include <iomanip>
#include <limits>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

#include "../utility/timer.h"

namespace analytic_solver
{
  struct OptimizationInfo
  {
    double cost;
    double cost_change;
    double abs_gradient;
    double abs_step;
    double iter_time;
  };

  enum solver_type_enum
  {
    UNDEFINED = -1,
    GRADIENT_DESCENT = 0,
    GAUSS_NEWTON = 1,
    LEVENBERG_MARQUARDT = 2
  };
  class Options
  {
    friend class PoseOnlyBundleAdjustmentSolver;

  public:
    Options() {}
    ~Options() {}

    solver_type_enum solver_type{solver_type_enum::GAUSS_NEWTON};
    struct
    {
      float threshold_step_size{1e-5};
      float threshold_cost_change{1e-5};
    } convergence_handle;
    struct
    {
      float threshold_huber_loss{1.5};
      float threshold_outlier_rejection{2.5};
    } outlier_handle;
    struct
    {
      int max_num_iterations{50};
    } iteration_handle;
  };

  class Summary
  {
    friend class PoseOnlyBundleAdjustmentSolver;

  public:
    Summary();
    ~Summary();
    std::string BriefReport();
    std::string FullReport();
    const double GetTotalTimeInSecond() const;

  protected:
    std::vector<OptimizationInfo> optimization_info_list_;
    int max_iteration_;
    double total_time_in_millisecond_;
    double threshold_step_size_;
    double threshold_cost_change_;
    bool convergence_status_;
  };

  class PoseOnlyBundleAdjustmentSolver
  {
  public:
    PoseOnlyBundleAdjustmentSolver();
    ~PoseOnlyBundleAdjustmentSolver();

    bool SolveMonocular_Planar3Dof(
        const std::vector<Eigen::Vector3f> &world_position_list,
        const std::vector<Eigen::Vector2f> &matched_pixel_list,
        const float fx, const float fy, const float cx, const float cy,
        const Eigen::Isometry3f &pose_base_to_camera,
        const Eigen::Isometry3f &pose_world_to_last,
        Eigen::Isometry3f &pose_world_to_current,
        std::vector<bool> &mask_inlier,
        Options options,
        Summary *summary = nullptr);
    bool SolveStereo_Planar3Dof(
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
        Summary *summary = nullptr);

    bool SolveMonocular_6Dof(
        const std::vector<Eigen::Vector3f> &world_position_list,
        const std::vector<Eigen::Vector2f> &matched_pixel_list,
        const float fx, const float fy, const float cx, const float cy,
        Eigen::Isometry3f &pose_world_to_current_frame,
        std::vector<bool> &mask_inlier,
        Options options,
        Summary *summary = nullptr);
    bool SolveStereo_6Dof(
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
        Summary *summary = nullptr);

    std::vector<Eigen::Isometry3f> GetDebugPoses();

  private:
    inline void WarpPositionList(
        const Eigen::Isometry3f &pose_target_to_initial,
        const std::vector<Eigen::Vector3f> &initial_position_list,
        std::vector<Eigen::Vector3f> &warped_position_list);

    inline void ComputeJacobianResidual_ReprojectionError_6Dof(
        const Eigen::Vector3f &local_position,
        const Eigen::Vector2f &matched_pixel,
        const float fx, const float fy, const float cx, const float cy,
        Eigen::Matrix<float, 6, 1> &jacobian_matrix_x_transpose,
        Eigen::Matrix<float, 6, 1> &jacobian_matrix_y_transpose,
        Eigen::Matrix<float, 2, 1> &residual_vector);
    inline void ComputeGradientHessian_ReprojectionError_6Dof(
        const Eigen::Matrix<float, 6, 1> &jacobian_matrix_x_transpose,
        const Eigen::Matrix<float, 6, 1> &jacobian_matrix_y_transpose,
        const Eigen::Matrix<float, 2, 1> &residual_vector,
        const float &threshold_huber,
        Eigen::Matrix<float, 6, 1> &gradient_vector,
        Eigen::Matrix<float, 6, 6> &hessian_matrix,
        float &error,
        float &error_nonweighted);

    inline void ComputeJacobianResidual_ReprojectionError_Planar3Dof(
        const Eigen::Vector3f &target_local_position,
        const Eigen::Vector3f &world_position,
        const Eigen::Vector2f &matched_pixel,
        const float fx, const float fy, const float cx, const float cy,
        const Eigen::Matrix<float, 3, 3> &rotation_target_camera_to_base,
        const float cos_psi, const float sin_psi,
        Eigen::Matrix<float, 3, 1> &jacobian_matrix_u_transpose,
        Eigen::Matrix<float, 3, 1> &jacobian_matrix_v_transpose,
        Eigen::Matrix<float, 2, 1> &residual_vector);
    inline void ComputeGradientHessian_ReprojectionError_Planar3Dof(
        const Eigen::Matrix<float, 3, 1> &jacobian_matrix_u_transpose,
        const Eigen::Matrix<float, 3, 1> &jacobian_matrix_v_transpose,
        const Eigen::Matrix<float, 2, 1> &residual_vector,
        const float &threshold_huber,
        Eigen::Matrix<float, 3, 1> &gradient_vector,
        Eigen::Matrix<float, 3, 3> &hessian_matrix,
        float &error,
        float &error_nonweighted);

  private:
    inline void CalcJtJ_x_6Dof(const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp);
    inline void CalcJtJ_y_6Dof(const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp);
    inline void CalcJtWJ_x_6Dof(const float weight, const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp);
    inline void CalcJtWJ_y_6Dof(const float weight, const Eigen::Matrix<float, 6, 1> &Jt, Eigen::Matrix<float, 6, 6> &JtJ_tmp);
    inline void AddHessian_OnlyUpperTriangle_6Dof(const Eigen::Matrix<float, 6, 6> &JtJ_tmp, Eigen::Matrix<float, 6, 6> &JtJ);
    inline void FillLowerTriangle_6Dof(Eigen::Matrix<float, 6, 6> &JtJ);

    inline void CalcJtJ_Planar3Dof(const Eigen::Matrix<float, 3, 1> &Jt, Eigen::Matrix<float, 3, 3> &JtJ_tmp);
    inline void CalcJtWJ_Planar3Dof(const float weight, const Eigen::Matrix<float, 3, 1> &Jt, Eigen::Matrix<float, 3, 3> &JtJ_tmp);
    inline void AddHessian_OnlyUpperTriangle_Planar3Dof(const Eigen::Matrix<float, 3, 3> &JtJ_tmp, Eigen::Matrix<float, 3, 3> &JtJ);
    inline void FillLowerTriangleByUpperTriangle_Planar3Dof(Eigen::Matrix<float, 3, 3> &JtJ);

  private:
    template <typename T>
    void CalculateMatrixExpoenetial_se3(const Eigen::Matrix<T, 6, 1> &xi, Eigen::Transform<T, 3, 1> &pose);
    template <typename T>
    void CalculateMatrixExpoenetial_so3(const Eigen::Matrix<T, 3, 1> &w, Eigen::Matrix<T, 3, 3> &R);

  private:
    std::vector<Eigen::Isometry3f> debug_poses_;
  };
};

#endif
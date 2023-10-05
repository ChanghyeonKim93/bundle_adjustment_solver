#ifndef _FULL_BUNDLE_ADJUSTMENT_SOLVER_REFACTOR_H_
#define _FULL_BUNDLE_ADJUSTMENT_SOLVER_REFACTOR_H_

#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
/*
    <Problem to solve>
        H*delta_theta = J.transpose()*r;
    where
    - Hessian
            H = [A_,  B_;
                 Bt_, C_];
    - Jacobian multiplied by residual vector
            J.transpose()*r = [a;b];
    - Update parameters
            delta_theta = [x;y] = -H.inverse() * J.transpose() * r;

*/

#include "../utility/timer.h"
#include "core/solver_option_and_summary.h"
#include "eigen3/Eigen/Dense"

namespace visual_navigation {
namespace analytic_solver {

using SolverNumeric = double;

using Index = int;
using Pixel = Eigen::Matrix<SolverNumeric, 2, 1>;
using Point = Eigen::Matrix<SolverNumeric, 3, 1>;
using Rotation3D = Eigen::Matrix<SolverNumeric, 3, 3>;
using Translation3D = Eigen::Matrix<SolverNumeric, 3, 1>;
using Pose = Eigen::Transform<SolverNumeric, 3, 1>;

using ErrorList = std::vector<SolverNumeric>;
using IndexList = std::vector<Index>;
using PixelList = std::vector<Pixel>;
using PointList = std::vector<Point>;

struct OptimizerCamera {
  OptimizerCamera() {}
  OptimizerCamera(const OptimizerCamera &camera) {
    fx = camera.fx;
    fy = camera.fy;
    cx = camera.cx;
    cy = camera.cy;
    camera_to_body_pose = camera.camera_to_body_pose;
  }

  SolverNumeric fx{0.0};
  SolverNumeric fy{0.0};
  SolverNumeric cx{0.0};
  SolverNumeric cy{0.0};
  Pose camera_to_body_pose;
};

struct PointObservation {
  int related_camera_id{-1};
  Pose *related_pose{nullptr};
  Point *related_point{nullptr};
  Pixel pixel{-1.0, -1.0};
};

/*
  주소 값이 아이디가 되는 것으로 할까 ?
  &mappoint 객체의 주소가 key가 된다.
  &pose 객체의 주소가 key가 된다.
  All pose  (N = N_optimize + N_fixed + N_nouse)
  All point (M = M_optimize + M_fixed + M_nouse)
  주소 -> pose - 1) optimizable  (N_optimize)
              - 2) fixed        (N_fixed)
              - 3)
  최적화 하고나서 주소 접근해서 업데이트 해줘야하는데?
*/
class FullBundleAdjustmentSolverRefactor {
 private:
  using MatrixDynamic = Eigen::Matrix<SolverNumeric, -1, -1>;

  using Mat2x3 = Eigen::Matrix<SolverNumeric, 2, 3>;
  using Mat3x2 = Eigen::Matrix<SolverNumeric, 3, 2>;

  using Mat2x6 = Eigen::Matrix<SolverNumeric, 2, 6>;
  using Mat6x2 = Eigen::Matrix<SolverNumeric, 6, 2>;

  using Mat3x1 = Eigen::Matrix<SolverNumeric, 3, 1>;
  using Mat1x3 = Eigen::Matrix<SolverNumeric, 1, 3>;
  using Mat3x3 = Eigen::Matrix<SolverNumeric, 3, 3>;

  using Mat3x6 = Eigen::Matrix<SolverNumeric, 3, 6>;
  using Mat6x3 = Eigen::Matrix<SolverNumeric, 6, 3>;

  using Mat6x6 = Eigen::Matrix<SolverNumeric, 6, 6>;

  using Vec2 = Eigen::Matrix<SolverNumeric, 2, 1>;
  using Vec3 = Eigen::Matrix<SolverNumeric, 3, 1>;
  using Vec6 = Eigen::Matrix<SolverNumeric, 6, 1>;

  using DiagBlockMat3x3 = std::vector<Mat3x3>;
  using DiagBlockMat6x6 = std::vector<Mat6x6>;

  using FullBlockMat3x3 = std::vector<std::vector<Mat3x3>>;
  using FullBlockMat6x3 = std::vector<std::vector<Mat6x3>>;
  using FullBlockMat3x6 = std::vector<std::vector<Mat3x6>>;
  using FullBlockMat6x6 = std::vector<std::vector<Mat6x6>>;

  using BlockVec3 = std::vector<Vec3>;
  using BlockVec6 = std::vector<Vec6>;

 public:
  FullBundleAdjustmentSolverRefactor();  // just reserve the memory

  void Reset();

  void RegisterCamera(const Index camera_id, const OptimizerCamera &camera);
  void RegisterWorldToBodyPose(Pose *original_pose);
  void RegisterWorldPoint(Point *original_point);

  void MakePoseFixed(Pose *original_pose);
  void MakePointFixed(Point *original_point);

  bool Solve(Options options, Summary *summary = nullptr);
  bool SolveByGradientDescent(Options options, Summary *summary = nullptr);

  std::string GetSolverStatistics() const;

 public:
  void AddObservation(const Index camera_id, Pose *related_pose,
                      Point *related_point, const Pixel &pixel);

 private:  // Solve related
  void FinalizeParameters();
  void SetProblemSize();
  void CheckPoseAndPointConnectivity();

  void ResetStorageMatrices();
  double EvaluateCurrentCost();
  double EvaluateCostChangeByQuadraticModel();
  void ReserveCurrentOptimizationParameters();
  void RevertToReservedOptimizationParameters();
  void UpdateOptimizationParameters(
      const std::vector<Vec6> &update_of_optimization_pose_parameter_list,
      const std::vector<Vec3> &update_of_optimization_point_parameter_list);

 private:
  inline bool IsFixedPose(Pose *original_pose);
  inline bool IsFixedPoint(Point *original_point);

 private:  // For fast calculations for symmetric matrices
  // Rij: jacobian_matrix_by_world_point (2x3)
  // Qij: jacobian_matrix_by_pose (2x6)
  // Rij_t_Rij: hessian_matrix_by_world_point (3x3)
  // Qij_t_Qij: hessian_matrix_by_pose (6x6)
  inline void CalculatePointHessianOnlyUpperTriangle(const SolverNumeric weight,
                                                     const Mat2x3 &Rij,
                                                     Mat3x3 *Rij_t_Rij);
  inline void CalculatePoseHessianOnlyUpperTriangle(const SolverNumeric weight,
                                                    const Mat2x6 &Qij,
                                                    Mat6x6 *Qij_t_Qij);
  inline void AccumulatePointHessianOnlyUpperTriangle(
      const Mat3x3 &Rij_t_Rij_upper, Mat3x3 *C);
  inline void AccumulatePoseHessianOnlyUpperTriangle(
      const Mat6x6 &Qij_t_Qij_upper, Mat6x6 *A);
  inline void FillLowerTriangleByUpperTriangle(Mat3x3 *C);
  inline void FillLowerTriangleByUpperTriangle(Mat6x6 *A);

 private:
  template <typename T>
  void se3Exp(const Eigen::Matrix<T, 6, 1> &xi,
              Eigen::Transform<T, 3, 1> &pose);
  template <typename T>
  void so3Exp(const Eigen::Matrix<T, 3, 1> &w, Eigen::Matrix<T, 3, 3> &R);

 private:  // Problem sizes
  int num_total_poses_;
  int num_total_points_;
  int num_optimization_poses_;
  int num_optimization_points_;
  int num_fixed_poses_;
  int num_fixed_points_;
  int num_total_observations_;

 private:
  bool is_parameter_finalized_{false};

 private:
  static constexpr SolverNumeric kTranslationScaler = 0.01;
  static constexpr SolverNumeric kInverseTranslationScaler =
      1.0 / kTranslationScaler;

 private:  // Camera list
  std::unordered_map<Index, OptimizerCamera> camera_id_to_camera_map_;

  std::unordered_map<Pose *, Pose>
      original_pose_to_inverse_optimized_pose_map_;                    // map
  std::unordered_set<Pose *> fixed_original_pose_set_;                 // set
  std::unordered_map<Pose *, Index> original_pose_to_pose_index_map_;  // map
  std::vector<Pose *> pose_index_to_original_pose_map_;                // vector
  std::vector<Pose> reserved_optimized_poses_;

  std::unordered_map<Point *, Point>
      original_point_to_optimized_point_map_;                             // map
  std::unordered_set<Point *> fixed_original_point_set_;                  // set
  std::unordered_map<Point *, Index> original_point_to_point_index_map_;  // map
  std::vector<Point *> point_index_to_original_point_map_;  // vector
  std::vector<Point> reserved_optimized_points_;

  std::vector<PointObservation> point_observation_list_;

 private:  // related to connectivity
  std::vector<std::unordered_set<Index>> i_opt_to_j_opt_;
  std::vector<std::unordered_set<Index>> j_opt_to_i_opt_;

  std::vector<std::unordered_set<Pose *>> i_opt_to_all_pose_;
  std::vector<std::unordered_set<Point *>> j_opt_to_all_point_;

 private:               // Storages to solve Schur Complement
  DiagBlockMat6x6 A_;   // N_opt (6x6) block diagonal part for poses
  FullBlockMat6x3 B_;   // N_opt x M_opt (6x3) block part (side)
  FullBlockMat3x6 Bt_;  // M_opt x N_opt (3x6) block part (side, transposed)
  DiagBlockMat3x3 C_;   // M_opt (3x3) block diagonal part for  3D points

  BlockVec6 a_;  // num_poses x 1 (6x1)
  BlockVec3 b_;  // num_points x 1 (3x1)

  BlockVec6 x_;  // num_poses (6x1)
  BlockVec3 y_;  // num_points (3x1)

  DiagBlockMat3x3 Cinv_;   // M_opt (3x3) blk diag. part for 3D points (inverse)
  FullBlockMat6x3 BCinv_;  // N_opt X M_opt  (6x3)
  FullBlockMat3x6 CinvBt_;   // M_opt x N_opt (3x6)
  FullBlockMat6x6 BCinvBt_;  // N_opt x N_opt (6x6)
  BlockVec6 BCinv_b_;        // N_opt (6x1)
  BlockVec3 Bt_x_;           // M_opt (3x1)
  BlockVec3 Cinv_b_;         // M_opt (3x1)

  FullBlockMat6x6 Am_BCinvBt_;  // N_opt x N_opt (6x6)
  BlockVec6 am_BCinv_b_;        // N_opt (6x1)
  BlockVec3 CinvBt_x_;          // M_opt (3x1)

  // Dynamic matrix
  MatrixDynamic Am_BCinvBt_mat_;  // 6*N_opt x 6*N_opt
  MatrixDynamic am_BCinv_b_mat_;  // 6*N_opt x 1

  MatrixDynamic x_mat_;  // 6*N_opt x 1
};

}  // namespace analytic_solver
}  // namespace visual_navigation

#endif
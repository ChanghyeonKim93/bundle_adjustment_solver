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

namespace analytic_solver {

using SolverNumeric = double;

using MatrixDynamic = Eigen::Matrix<SolverNumeric, -1, -1>;

using Matrix2x3 = Eigen::Matrix<SolverNumeric, 2, 3>;
using Matrix3x2 = Eigen::Matrix<SolverNumeric, 3, 2>;

using Matrix2x6 = Eigen::Matrix<SolverNumeric, 2, 6>;
using Matrix6x2 = Eigen::Matrix<SolverNumeric, 6, 2>;

using Matrix3x1 = Eigen::Matrix<SolverNumeric, 3, 1>;
using Matrix1x3 = Eigen::Matrix<SolverNumeric, 1, 3>;
using Matrix3x3 = Eigen::Matrix<SolverNumeric, 3, 3>;

using Matrix3x6 = Eigen::Matrix<SolverNumeric, 3, 6>;
using Matrix6x3 = Eigen::Matrix<SolverNumeric, 6, 3>;

using Matrix6x6 = Eigen::Matrix<SolverNumeric, 6, 6>;

using Vector2 = Eigen::Matrix<SolverNumeric, 2, 1>;
using Vector3 = Eigen::Matrix<SolverNumeric, 3, 1>;
using Vector6 = Eigen::Matrix<SolverNumeric, 6, 1>;

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

using DiagBlockMat33 = std::vector<Matrix3x3>;
using DiagBlockMat66 = std::vector<Matrix6x6>;

using FullBlockMat3x3 = std::vector<std::vector<Matrix3x3>>;
using FullBlockMat6x3 = std::vector<std::vector<Matrix6x3>>;
using FullBlockMat3x6 = std::vector<std::vector<Matrix3x6>>;
using FullBlockMat6x6 = std::vector<std::vector<Matrix6x6>>;

using BlockVec3 = std::vector<Vector3>;
using BlockVec6 = std::vector<Vector6>;

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

struct BundleAdjustmentObservation {
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
class FullBundleAdjustmentSolver {
 public:
  FullBundleAdjustmentSolver();  // just reserve the memory
  ~FullBundleAdjustmentSolver();

  void Reset();

  void AddCamera(const Index camera_id, const OptimizerCamera &camera);
  void AddPose(Pose *original_pose);
  void AddPoint(Point *original_point);

  void AddObservation(const Index camera_id, Pose *related_pose,
                      Point *related_point, const Pixel &pixel);

  void MakePoseFixed(Pose *original_pose_to_be_fixed);
  void MakePointFixed(Point *original_point_to_be_fixed);

  bool Solve(Options options, Summary *summary = nullptr);

  std::string GetSolverStatistics() const;

 private:
  SolverNumeric scaler_;
  SolverNumeric inverse_scaler_;

 private:
  void FinalizeParameters();
  void SetProblemSize();

 private:  // Solve related
  void CheckPoseAndPointConnectivity();

  void ZeroizeStorageMatrices();
  double EvaluateCurrentError();

  double EvaluateErrorChangeByQuadraticModel();

  void ReserveCurrentParameters();  // reserved_notupdated_opt_poses_,
                                    // reserved_notupdated_opt_points_;
  void RevertToReservedParameters();
  void UpdateParameters(const std::vector<Vector6> &pose_update_list,
                        const std::vector<Vector3> &point_update_list);

 private:  // For fast calculations for symmetric matrices
  // Rij: jacobian_matrix_by_world_point (2x3)
  // Qij: jacobian_matrix_by_pose (2x6)
  // Rij_t_Rij: hessian_matrix_by_world_point (3x3)
  // Qij_t_Qij: hessian_matrix_by_pose (6x6)
  inline void CalculatePointHessianOnlyUpperTriangle(const Matrix2x3 &Rij,
                                                     Matrix3x3 &Rij_t_Rij);
  inline void CalculatePointHessianOnlyUpperTriangleWithWeight(
      const SolverNumeric weight, const Matrix2x3 &Rij, Matrix3x3 &Rij_t_Rij);

  inline void CalculatePoseHessianOnlyUpperTriangle(const Matrix2x6 &Qij,
                                                    Matrix6x6 &Qij_t_Qij);
  inline void CalculatePoseHessianOnlyUpperTriangleWithWeight(
      const SolverNumeric weight, const Matrix2x6 &Qij, Matrix6x6 &Qij_t_Qij);

  inline void AccumulatePointHessianOnlyUpperTriangle(
      Matrix3x3 &C, Matrix3x3 &Rij_t_Rij_upper);
  inline void AccumulatePoseHessianOnlyUpperTriangle(
      Matrix6x6 &A, Matrix6x6 &Qij_t_Qij_upper);

  inline void FillLowerTriangleByUpperTriangle(Matrix3x3 &C);
  inline void FillLowerTriangleByUpperTriangle(Matrix6x6 &A);

 private:
  template <typename T>
  void se3Exp(const Eigen::Matrix<T, 6, 1> &xi,
              Eigen::Transform<T, 3, 1> &pose);
  template <typename T>
  void so3Exp(const Eigen::Matrix<T, 3, 1> &w, Eigen::Matrix<T, 3, 3> &R);

 private:                       // Problem sizes
  int num_total_poses_;         // # of all inserted poses
  int num_total_points_;        // # of all insertedmappoints
  int num_opt_poses_;           // # of optimization poses
  int num_opt_points_;          // # of optimization mappoints
  int num_fixed_poses_;         // # of fixed poses
  int num_fixed_points_;        // # of fixed points
  int num_total_observations_;  // # of total observations

 private:
  bool is_parameter_finalized_{false};

 private:            // Camera list
  int num_cameras_;  // # of rigidly fixed cameras (number 0 is the major
                     // camera)
  std::unordered_map<Index, OptimizerCamera> camera_id_to_camera_map_;

  std::unordered_map<Pose *, Pose> original_pose_to_T_jw_map_;    // map
  std::unordered_set<Pose *> fixed_original_pose_set_;            // set
  std::unordered_map<Pose *, Index> original_pose_to_j_opt_map_;  // map
  std::vector<Pose *> j_opt_to_original_pose_map_;                // vector
  std::vector<Pose> reserved_opt_poses_;

  std::unordered_map<Point *, Point> original_point_to_Xi_map_;     // map
  std::unordered_set<Point *> fixed_original_point_set_;            // set
  std::unordered_map<Point *, Index> original_point_to_i_opt_map_;  // map
  std::vector<Point *> i_opt_to_original_point_map_;                // vector
  std::vector<Point> reserved_opt_points_;

  std::vector<BundleAdjustmentObservation> observation_list_;

 private:  // related to connectivity
  std::vector<std::unordered_set<Index>> i_opt_to_j_opt_;
  std::vector<std::unordered_set<Index>> j_opt_to_i_opt_;

  std::vector<std::unordered_set<Pose *>> i_opt_to_all_pose_;
  std::vector<std::unordered_set<Point *>> j_opt_to_all_point_;

 private:               // Storages to solve Schur Complement
  DiagBlockMat66 A_;    // N_opt (6x6) block diagonal part for poses
  FullBlockMat6x3 B_;   // N_opt x M_opt (6x3) block part (side)
  FullBlockMat3x6 Bt_;  // M_opt x N_opt (3x6) block part (side, transposed)
  DiagBlockMat33 C_;    // M_opt (3x3) block diagonal part for  3D points

  BlockVec6 a_;  // num_poses x 1 (6x1)
  BlockVec3 b_;  // num_points x 1 (3x1)

  BlockVec6 x_;  // num_poses (6x1)
  BlockVec3 y_;  // num_points (3x1)

  DiagBlockMat33 Cinv_;     // M_opt (3x3) block diagonal part for landmarks' 3D
                            // points (inverse)
  FullBlockMat6x3 BCinv_;   // N_opt X M_opt  (6x3)
  FullBlockMat3x6 CinvBt_;  // M_opt x N_opt (3x6)
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

};  // namespace analytic_solver

#endif
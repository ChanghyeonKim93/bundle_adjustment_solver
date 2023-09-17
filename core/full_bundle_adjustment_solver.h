#ifndef _FULL_BUNDLE_ADJUSTMENT_SOLVER_H_
#define _FULL_BUNDLE_ADJUSTMENT_SOLVER_H_

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
using _BA_Numeric = double;

using _BA_MatDynamic = Eigen::Matrix<_BA_Numeric, -1, -1>;
using _BA_Mat11 = _BA_Numeric;
using _BA_Mat22 = Eigen::Matrix<_BA_Numeric, 2, 2>;

using _BA_Mat13 = Eigen::Matrix<_BA_Numeric, 1, 3>;
using _BA_Mat31 = Eigen::Matrix<_BA_Numeric, 3, 1>;

using _BA_Mat23 = Eigen::Matrix<_BA_Numeric, 2, 3>;
using _BA_Mat32 = Eigen::Matrix<_BA_Numeric, 3, 2>;

using _BA_Mat26 = Eigen::Matrix<_BA_Numeric, 2, 6>;
using _BA_Mat62 = Eigen::Matrix<_BA_Numeric, 6, 2>;

using _BA_Mat33 = Eigen::Matrix<_BA_Numeric, 3, 3>;

using _BA_Mat36 = Eigen::Matrix<_BA_Numeric, 3, 6>;
using _BA_Mat63 = Eigen::Matrix<_BA_Numeric, 6, 3>;

using _BA_Mat66 = Eigen::Matrix<_BA_Numeric, 6, 6>;

using _BA_Vec1 = _BA_Numeric;
using _BA_Vec2 = Eigen::Matrix<_BA_Numeric, 2, 1>;
using _BA_Vec3 = Eigen::Matrix<_BA_Numeric, 3, 1>;
using _BA_Vec6 = Eigen::Matrix<_BA_Numeric, 6, 1>;

using _BA_Index = int;
using _BA_Size_t = int;
using _BA_Pixel = Eigen::Matrix<_BA_Numeric, 2, 1>;
using _BA_Point = Eigen::Matrix<_BA_Numeric, 3, 1>;
using _BA_Rotation3 = Eigen::Matrix<_BA_Numeric, 3, 3>;
using _BA_Position3 = Eigen::Matrix<_BA_Numeric, 3, 1>;
using _BA_Pose = Eigen::Transform<_BA_Numeric, 3, 1>;

using _BA_PoseSE3 = Eigen::Matrix<_BA_Numeric, 4, 4>;
using _BA_PoseSE3Tangent = Eigen::Matrix<_BA_Numeric, 6, 1>;

using _BA_ErrorVec = std::vector<_BA_Numeric>;
using _BA_IndexVec = std::vector<_BA_Index>;
using _BA_PixelVec = std::vector<_BA_Pixel>;
using _BA_PointVec = std::vector<_BA_Point>;

using DiagBlockMat33 = std::vector<_BA_Mat33>;
using DiagBlockMat66 = std::vector<_BA_Mat66>;

using FullBlockMat11 = std::vector<std::vector<_BA_Mat11>>;
using FullBlockMat13 = std::vector<std::vector<_BA_Mat13>>;
using FullBlockMat31 = std::vector<std::vector<_BA_Mat31>>;
using FullBlockMat33 = std::vector<std::vector<_BA_Mat33>>;
using FullBlockMat63 = std::vector<std::vector<_BA_Mat63>>;
using FullBlockMat36 = std::vector<std::vector<_BA_Mat36>>;
using FullBlockMat66 = std::vector<std::vector<_BA_Mat66>>;

using BlockVec1 = std::vector<_BA_Vec1>;
using BlockVec3 = std::vector<_BA_Vec3>;
using BlockVec6 = std::vector<_BA_Vec6>;

struct _BA_Camera {
  _BA_Camera(){};
  _BA_Camera(const _BA_Camera &camera) {
    fx = camera.fx;
    fy = camera.fy;
    cx = camera.cx;
    cy = camera.cy;
    pose_cam0_to_this = camera.pose_cam0_to_this;
    pose_this_to_cam0 = camera.pose_this_to_cam0;
  }

  _BA_Numeric fx{0.0};
  _BA_Numeric fy{0.0};
  _BA_Numeric cx{0.0};
  _BA_Numeric cy{0.0};
  _BA_Pose pose_cam0_to_this;
  _BA_Pose pose_this_to_cam0;
};

struct _BA_Observation {
  int camera_index{-1};
  _BA_Pose *related_pose{nullptr};
  _BA_Point *related_point{nullptr};
  _BA_Pixel pixel{-1.0, -1.0};
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

  void AddCamera(const _BA_Index camera_index, const _BA_Camera &camera);
  void AddPose(_BA_Pose *original_pose);
  void AddPoint(_BA_Point *original_point);

  void AddObservation(const _BA_Index index_camera, _BA_Pose *related_pose, _BA_Point *related_point,
                      const _BA_Pixel &pixel);

  void MakePoseFixed(_BA_Pose *original_pose_to_be_fixed);
  void MakePointFixed(_BA_Point *original_point_to_be_fixed);

  bool Solve(Options options, Summary *summary = nullptr);

  std::string GetSolverStatistics() const;

 private:
  _BA_Numeric scaler_;
  _BA_Numeric inverse_scaler_;

 private:
  void FinalizeParameters();
  void SetProblemSize();

 private:  // Solve related
  void CheckPoseAndPointConnectivity();

  void ZeroizeStorageMatrices();
  double EvaluateCurrentError();

  double EvaluateErrorChangeByQuadraticModel();

  void ReserveCurrentParameters();  // reserved_notupdated_opt_poses_, reserved_notupdated_opt_points_;
  void RevertToReservedParameters();
  void UpdateParameters(const std::vector<_BA_Vec6> &x_list, const std::vector<_BA_Vec3> &y_list);

 private:  // For fast calculations for symmetric matrices
  inline void CalcRijtRijOnlyUpperTriangle(const _BA_Mat23 &Rij, _BA_Mat33 &Rij_t_Rij);
  inline void CalcRijtRijweightOnlyUpperTriangle(const _BA_Numeric weight, const _BA_Mat23 &Rij, _BA_Mat33 &Rij_t_Rij);

  inline void CalcQijtQijOnlyUpperTriangle(const _BA_Mat26 &Qij, _BA_Mat66 &Qij_t_Qij);
  inline void CalcQijtQijweightOnlyUpperTriangle(const _BA_Numeric weight, const _BA_Mat26 &Qij, _BA_Mat66 &Qij_t_Qij);

  inline void AddUpperTriangle(_BA_Mat33 &C, _BA_Mat33 &Rij_t_Rij_upper);
  inline void AddUpperTriangle(_BA_Mat66 &A, _BA_Mat66 &Qij_t_Qij_upper);

  inline void FillLowerTriangleByUpperTriangle(_BA_Mat33 &C);
  inline void FillLowerTriangleByUpperTriangle(_BA_Mat66 &A);

 private:
  template <typename T>
  void se3Exp(const Eigen::Matrix<T, 6, 1> &xi, Eigen::Transform<T, 3, 1> &pose);
  template <typename T>
  void so3Exp(const Eigen::Matrix<T, 3, 1> &w, Eigen::Matrix<T, 3, 3> &R);

 private:                              // Problem sizes
  _BA_Size_t num_total_poses_;         // # of all inserted poses
  _BA_Size_t num_total_points_;        // # of all insertedmappoints
  _BA_Size_t num_opt_poses_;           // # of optimization poses
  _BA_Size_t num_opt_points_;          // # of optimization mappoints
  _BA_Size_t num_fixed_poses_;         // # of fixed poses
  _BA_Size_t num_fixed_points_;        // # of fixed points
  _BA_Size_t num_total_observations_;  // # of total observations

 private:
  bool is_parameter_finalized_{false};

 private:            // Camera list
  int num_cameras_;  // # of rigidly fixed cameras (number 0 is the major camera)
  std::unordered_map<_BA_Index, _BA_Camera> camera_index_to_camera_map_;

  std::unordered_map<_BA_Pose *, _BA_Pose> original_pose_to_T_jw_map_;    // map
  std::unordered_set<_BA_Pose *> fixed_original_pose_set_;                // set
  std::unordered_map<_BA_Pose *, _BA_Index> original_pose_to_j_opt_map_;  // map
  std::vector<_BA_Pose *> j_opt_to_original_pose_map_;                    // vector
  std::vector<_BA_Pose> reserved_opt_poses_;

  std::unordered_map<_BA_Point *, _BA_Point> original_point_to_Xi_map_;     // map
  std::unordered_set<_BA_Point *> fixed_original_point_set_;                // set
  std::unordered_map<_BA_Point *, _BA_Index> original_point_to_i_opt_map_;  // map
  std::vector<_BA_Point *> i_opt_to_original_point_map_;                    // vector
  std::vector<_BA_Point> reserved_opt_points_;

  std::vector<_BA_Observation> observation_list_;

 private:  // related to connectivity
  std::vector<std::unordered_set<_BA_Index>> i_opt_to_j_opt_;
  std::vector<std::unordered_set<_BA_Index>> j_opt_to_i_opt_;

  std::vector<std::unordered_set<_BA_Pose *>> i_opt_to_all_pose_;
  std::vector<std::unordered_set<_BA_Point *>> j_opt_to_all_point_;

 private:              // Storages to solve Schur Complement
  DiagBlockMat66 A_;   // N_opt (6x6) block diagonal part for poses
  FullBlockMat63 B_;   // N_opt x M_opt (6x3) block part (side)
  FullBlockMat36 Bt_;  // M_opt x N_opt (3x6) block part (side, transposed)
  DiagBlockMat33 C_;   // M_opt (3x3) block diagonal part for landmarks' 3D points

  BlockVec6 a_;  // N_opt x 1 (6x1)
  BlockVec3 b_;  // M_opt x 1 (3x1)

  BlockVec6 x_;  // N_opt (6x1)
  BlockVec3 y_;  // M_opt (3x1)

  std::vector<_BA_Pose> params_poses_;    // N_opt (Eigen::Isometry3) parameter vector for poses
  std::vector<_BA_Point> params_points_;  // M_opt (3x1) parameter vector for points

  DiagBlockMat33 Cinv_;     // M_opt (3x3) block diagonal part for landmarks' 3D points (inverse)
  FullBlockMat63 BCinv_;    // N_opt X M_opt  (6x3)
  FullBlockMat36 CinvBt_;   // M_opt x N_opt (3x6)
  FullBlockMat66 BCinvBt_;  // N_opt x N_opt (6x6)
  BlockVec6 BCinv_b_;       // N_opt (6x1)
  BlockVec3 Bt_x_;          // M_opt (3x1)
  BlockVec3 Cinv_b_;        // M_opt (3x1)

  FullBlockMat66 Am_BCinvBt_;  // N_opt x N_opt (6x6)
  BlockVec6 am_BCinv_b_;       // N_opt (6x1)
  BlockVec3 CinvBt_x_;         // M_opt (3x1)

  // Dynamic matrix
  _BA_MatDynamic Am_BCinvBt_mat_;  // 6*N_opt x 6*N_opt
  _BA_MatDynamic am_BCinv_b_mat_;  // 6*N_opt x 1

  _BA_MatDynamic x_mat_;  // 6*N_opt x 1
};

};  // namespace analytic_solver

#endif
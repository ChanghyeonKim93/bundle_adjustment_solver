#ifndef _FULL_BUNDLE_ADJUSTMENT_SOLVER_CERES_H_
#define _FULL_BUNDLE_ADJUSTMENT_SOLVER_CERES_H_

#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "solver_option_and_summary.h"

struct Camera {
  float fx{0.0f};
  float fy{0.0f};
  float cx{0.0f};
  float cy{0.0f};
  Eigen::Isometry3f camera_to_camera_link_pose{Eigen::Isometry3f::Identity()};
};

namespace visual_navigation {
namespace analytic_solver {

using SolverNumeric = double;
using Pose = Eigen::Transform<SolverNumeric, 3, 1>;
using Point = Eigen::Matrix<SolverNumeric, 3, 1>;
using Pixel = Eigen::Matrix<SolverNumeric, 2, 1>;

struct OptimizerCameraForCeres {
  SolverNumeric fx{0.0};
  SolverNumeric fy{0.0};
  SolverNumeric cx{0.0};
  SolverNumeric cy{0.0};
  Pose pose_this_to_cam0;

  OptimizerCameraForCeres(){};
  OptimizerCameraForCeres(const float fx, const float fy, const float cx,
                          const float cy,
                          const Eigen::Isometry3f &camera_link_to_camera_pose) {
    this->fx = fx;
    this->fy = fy;
    this->cx = cx;
    this->cy = cy;
    pose_this_to_cam0 = camera_link_to_camera_pose.cast<double>();
  }
};

struct PointObservation {
  int related_camera_id{-1};
  Pose *related_pose{nullptr};
  Point *related_point{nullptr};
  Pixel pixel{-1.0, -1.0};
};

class FullBundleAdjustmentSolverCeres {
 public:
  FullBundleAdjustmentSolverCeres();  // just reserve the memory
  ~FullBundleAdjustmentSolverCeres();

  void Reset();

  void AddCamera(const int camera_index, const Camera &camera);
  void AddPose(Pose *original_pose);
  void AddPoint(Point *original_point);

  void AddObservation(const int camera_index, Pose *related_pose,
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
  void UpdateParameters(const std::vector<_BA_Vec6> &x_list,
                        const std::vector<_BA_Vec3> &y_list);

 private:  // For fast calculations for symmetric matrices
  inline void CalcRijtRijOnlyUpperTriangle(const _BA_Mat23 &Rij,
                                           _BA_Mat33 &Rij_t_Rij);
  inline void CalcRijtRijweightOnlyUpperTriangle(const SolverNumeric weight,
                                                 const _BA_Mat23 &Rij,
                                                 _BA_Mat33 &Rij_t_Rij);

  inline void CalcQijtQijOnlyUpperTriangle(const _BA_Mat26 &Qij,
                                           _BA_Mat66 &Qij_t_Qij);
  inline void CalcQijtQijweightOnlyUpperTriangle(const SolverNumeric weight,
                                                 const _BA_Mat26 &Qij,
                                                 _BA_Mat66 &Qij_t_Qij);

  inline void AddUpperTriangle(_BA_Mat33 &C, _BA_Mat33 &Rij_t_Rij_upper);
  inline void AddUpperTriangle(_BA_Mat66 &A, _BA_Mat66 &Qij_t_Qij_upper);

  inline void FillLowerTriangleByUpperTriangle(_BA_Mat33 &C);
  inline void FillLowerTriangleByUpperTriangle(_BA_Mat66 &A);

 private:
  template <typename T>
  void se3Exp(const Eigen::Matrix<T, 6, 1> &xi,
              Eigen::Transform<T, 3, 1> &pose);
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
  int num_cameras_;  // # of rigidly fixed cameras (number 0 is the major
                     // camera)
  std::unordered_map<_BA_Index, _BA_Camera> camera_index_to_camera_map_;

  std::unordered_map<Pose *, Pose> original_pose_to_T_jw_map_;        // map
  std::unordered_set<Pose *> fixed_original_pose_set_;                // set
  std::unordered_map<Pose *, _BA_Index> original_pose_to_j_opt_map_;  // map
  std::vector<Pose *> j_opt_to_original_pose_map_;                    // vector
  std::vector<Pose> reserved_opt_poses_;

  std::unordered_map<Point *, Point> original_point_to_Xi_map_;         // map
  std::unordered_set<Point *> fixed_original_point_set_;                // set
  std::unordered_map<Point *, _BA_Index> original_point_to_i_opt_map_;  // map
  std::vector<Point *> i_opt_to_original_point_map_;  // vector
  std::vector<Point> reserved_opt_points_;

  std::vector<PointObservation> observation_list_;

 private:  // related to connectivity
  std::vector<std::unordered_set<_BA_Index>> i_opt_to_j_opt_;
  std::vector<std::unordered_set<_BA_Index>> j_opt_to_i_opt_;

  std::vector<std::unordered_set<Pose *>> i_opt_to_all_pose_;
  std::vector<std::unordered_set<Point *>> j_opt_to_all_point_;

 private:               // Storages to solve Schur Complement
  DiagBlockMat66 A_;    // N_opt (6x6) block diagonal part for poses
  FullBlockMat6x3 B_;   // N_opt x M_opt (6x3) block part (side)
  FullBlockMat3x6 Bt_;  // M_opt x N_opt (3x6) block part (side, transposed)
  DiagBlockMat33
      C_;  // M_opt (3x3) block diagonal part for landmarks' 3D points

  BlockVec6 a_;  // N_opt x 1 (6x1)
  BlockVec3 b_;  // M_opt x 1 (3x1)

  BlockVec6 x_;  // N_opt (6x1)
  BlockVec3 y_;  // M_opt (3x1)

  std::vector<Pose>
      params_poses_;  // N_opt (Eigen::Isometry3) parameter vector for poses
  std::vector<Point> params_points_;  // M_opt (3x1) parameter vector for points

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
  _BA_MatDynamic Am_BCinvBt_mat_;  // 6*N_opt x 6*N_opt
  _BA_MatDynamic am_BCinv_b_mat_;  // 6*N_opt x 1

  _BA_MatDynamic x_mat_;  // 6*N_opt x 1
};

}  // namespace analytic_solver
}  // namespace visual_navigation

#endif
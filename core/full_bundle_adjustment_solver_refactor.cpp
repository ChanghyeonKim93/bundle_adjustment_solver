#include "full_bundle_adjustment_solver_refactor.h"

#include <stdexcept>

namespace visual_navigation {
namespace analytic_solver {

FullBundleAdjustmentSolverRefactor::FullBundleAdjustmentSolverRefactor()
    : num_total_poses_(0),
      num_total_points_(0),
      num_optimization_poses_(0),
      num_optimization_points_(0),
      num_fixed_poses_(0),
      num_fixed_points_(0),
      num_total_observations_(0),
      is_parameter_finalized_(false) {
  original_pose_to_inverse_optimized_pose_map_.reserve(500);
  fixed_original_pose_set_.reserve(500);
  original_pose_to_pose_index_map_.reserve(500);
  pose_index_to_original_pose_map_.reserve(500);
  reserved_optimized_poses_.reserve(500);

  original_point_to_optimized_point_map_.reserve(100000);
  fixed_original_point_set_.reserve(100000);
  original_point_to_point_index_map_.reserve(100000);
  point_index_to_original_point_map_.reserve(100000);
  reserved_optimized_points_.reserve(100000);

  point_observation_list_.reserve(2000000);

  A_.reserve(500);  // reserve expected # of optimizable poses (N_opt)
  B_.reserve(500);  //
  Bt_.reserve(100000);
  C_.reserve(100000);  // reserve expected # of optimizable landmarks (M_opt)

  std::cerr << std::string(__func__) + ": " +
                   "initialization is done in constructor.\n";
}

void FullBundleAdjustmentSolverRefactor::Reset() {
  camera_id_to_camera_map_.clear();

  num_total_poses_ = 0;
  num_total_points_ = 0;
  num_optimization_poses_ = 0;
  num_optimization_points_ = 0;
  num_fixed_poses_ = 0;
  num_fixed_points_ = 0;
  num_total_observations_ = 0;

  is_parameter_finalized_ = false;

  original_pose_to_inverse_optimized_pose_map_.clear();
  fixed_original_pose_set_.clear();
  original_pose_to_pose_index_map_.clear();
  pose_index_to_original_pose_map_.resize(0);
  reserved_optimized_poses_.resize(0);

  original_point_to_optimized_point_map_.clear();
  fixed_original_point_set_.clear();
  original_point_to_point_index_map_.clear();
  point_index_to_original_point_map_.resize(0);
  reserved_optimized_poses_.resize(0);

  point_observation_list_.resize(0);
  // TODO(@)
}

void FullBundleAdjustmentSolverRefactor::RegisterCamera(
    const Index camera_id, const OptimizerCamera &camera) {
  if (camera_id_to_camera_map_.count(camera_id) > 0) {
    std::cerr << TEXT_YELLOW(std::string(__func__) + ": " +
                             "WANNING: existing camera\n");
    return;
  }

  auto scaled_camera = camera;
  scaled_camera.camera_to_body_pose.translation() *= kTranslationScaler;
  scaled_camera.fx *= kTranslationScaler;
  scaled_camera.fy *= kTranslationScaler;
  scaled_camera.cx *= kTranslationScaler;
  scaled_camera.cy *= kTranslationScaler;

  camera_id_to_camera_map_.insert({camera_id, scaled_camera});

  std::stringstream ss;
  ss << "New camera is added.\n";
  ss << "  fx: " << scaled_camera.fx << ", fy: " << scaled_camera.fy
     << ", cx: " << scaled_camera.cx << ", cy: " << scaled_camera.cy << "\n";
  ss << "  scaled camera_to_body_pose:\n"
     << scaled_camera.camera_to_body_pose.linear() << "\n"
     << scaled_camera.camera_to_body_pose.translation().transpose() << "\n";
  std::cerr << ss.str();
}

void FullBundleAdjustmentSolverRefactor::RegisterWorldToBodyPose(
    Pose *original_pose) {
  if (is_parameter_finalized_)
    throw std::runtime_error(TEXT_RED(
        std::string(__func__) + ": " +
        "Cannot enroll parameter. (is_parameter_finalized_ == true)\n"));

  if (original_pose_to_inverse_optimized_pose_map_.count(original_pose) == 0) {
    Pose T_jw = original_pose->inverse();
    T_jw.translation() = T_jw.translation() * kTranslationScaler;
    original_pose_to_inverse_optimized_pose_map_.insert({original_pose, T_jw});
    ++num_total_poses_;
  }
}

void FullBundleAdjustmentSolverRefactor::RegisterWorldPoint(
    Point *original_point) {
  if (is_parameter_finalized_)
    throw std::runtime_error(TEXT_RED(
        std::string(__func__) + ": " +
        "Cannot enroll parameter. (is_parameter_finalized_ == true)\n"));

  if (original_point_to_optimized_point_map_.count(original_point) == 0) {
    Point Xi = *original_point;
    Xi = Xi * kTranslationScaler;
    original_point_to_optimized_point_map_.insert({original_point, Xi});
    ++num_total_points_;
  }
}

void FullBundleAdjustmentSolverRefactor::MakePoseFixed(Pose *original_poseptr) {
  if (is_parameter_finalized_)
    throw std::runtime_error(TEXT_RED(
        std::string(__func__) + ": " +
        "Cannot enroll parameter. (is_parameter_finalized_ == true)\n"));
  if (original_poseptr == nullptr)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " +
                 "Empty pose pointer is conveyed. Skip this one.\n"));
  if (original_pose_to_inverse_optimized_pose_map_.count(original_poseptr) == 0)
    throw std::runtime_error("There is no pointer in the BA pose pool.");

  fixed_original_pose_set_.insert(original_poseptr);
  ++num_fixed_poses_;
}

void FullBundleAdjustmentSolverRefactor::MakePointFixed(
    Point *original_pointptr_to_be_fixed) {
  if (is_parameter_finalized_)
    throw std::runtime_error(TEXT_RED(
        std::string(__func__) +
        "Cannot enroll parameter. (is_parameter_finalized_ == true)\n"));
  if (original_pointptr_to_be_fixed == nullptr)
    throw std::runtime_error(TEXT_RED(std::string(__func__) + ": " +
                                      "Empty point pointer is conveyed.\n"));
  if (original_point_to_optimized_point_map_.count(
          original_pointptr_to_be_fixed) == 0)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " +
                 "There is no pointer in the BA point pool."));

  fixed_original_point_set_.insert(original_pointptr_to_be_fixed);
  ++num_fixed_points_;
}

void FullBundleAdjustmentSolverRefactor::FinalizeParameters() {
  num_optimization_poses_ = 0;
  for (auto &[original_pose, inverse_optimized_pose] :
       original_pose_to_inverse_optimized_pose_map_) {
    if (fixed_original_pose_set_.count(original_pose) > 0) continue;
    pose_index_to_original_pose_map_.push_back(original_pose);
    original_pose_to_pose_index_map_.insert(
        {original_pose, num_optimization_poses_});
    ++num_optimization_poses_;
  }
  reserved_optimized_poses_.resize(pose_index_to_original_pose_map_.size());

  num_optimization_points_ = 0;
  for (auto &[original_point, Xi] : original_point_to_optimized_point_map_) {
    if (fixed_original_point_set_.count(original_point) > 0) continue;
    point_index_to_original_point_map_.push_back(original_point);
    original_point_to_point_index_map_.insert(
        {original_point, num_optimization_points_});
    ++num_optimization_points_;
  }
  reserved_optimized_points_.resize(point_index_to_original_point_map_.size());

  SetProblemSize();

  is_parameter_finalized_ = true;
}

std::string FullBundleAdjustmentSolverRefactor::GetSolverStatistics() const {
  std::stringstream ss;
  ss << "| Bundle Adjustment Statistics:" << std::endl;
  ss << "| # cameras in rigid body system: " << camera_id_to_camera_map_.size()
     << std::endl;
  ss << "|   "
     << TEXT_CYAN("(Note: The reference camera is 'camera_list_[0]'.)")
     << std::endl;
  ss << "|             # of total poses: " << num_total_poses_ << std::endl;
  ss << "|               - # fix  poses: " << num_fixed_poses_ << std::endl;
  ss << "|               - # opt. poses: " << num_optimization_poses_
     << std::endl;
  ss << "|            # of total points: " << num_total_points_ << std::endl;
  ss << "|              - # fix  points: " << num_fixed_points_ << std::endl;
  ss << "|              - # opt. points: " << num_optimization_points_
     << std::endl;
  ss << "|            # of observations: " << num_total_observations_
     << std::endl;
  ss << "|                Jacobian size: " << 6 * num_total_observations_
     << " rows x " << 3 * num_optimization_points_ + 6 * num_optimization_poses_
     << " cols" << std::endl;
  ss << "|                Residual size: " << 2 * num_total_observations_
     << " rows" << std::endl;
  ss << std::endl;

  std::cerr << ss.str();

  return ss.str();
}

void FullBundleAdjustmentSolverRefactor::AddObservation(
    const Index camera_index, Pose *related_pose, Point *related_point,
    const Pixel &pixel) {
  PointObservation observation;
  if (camera_id_to_camera_map_.count(camera_index) == 0)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " + "Invalid camera index.\n"));
  if (original_pose_to_inverse_optimized_pose_map_.count(related_pose) == 0)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " + "Nonexisting pose.\n"));
  if (original_point_to_optimized_point_map_.count(related_point) == 0)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " + "Nonexisting point.\n"));

  observation.related_camera_id = camera_index;
  observation.related_pose = related_pose;
  observation.related_point = related_point;
  observation.pixel = pixel * kTranslationScaler;

  point_observation_list_.push_back(observation);
  ++num_total_observations_;
}

void FullBundleAdjustmentSolverRefactor::SetProblemSize() {
  // Resize storages.
  A_.resize(num_optimization_poses_, Mat6x6::Zero());

  B_.resize(num_optimization_poses_);
  for (Index j = 0; j < num_optimization_poses_; ++j)
    B_[j].resize(num_optimization_points_,
                 Mat6x3::Zero());  // 6x3, N_opt X M blocks

  Bt_.resize(num_optimization_points_);
  for (Index i = 0; i < num_optimization_points_; ++i)
    Bt_[i].resize(num_optimization_poses_,
                  Mat3x6::Zero());  // 3x6, N_opt X M blocks

  C_.resize(num_optimization_points_, Mat3x3::Zero());

  a_.resize(num_optimization_poses_, Vec6::Zero());  // 6x1, N_opt blocks
  x_.resize(num_optimization_poses_, Vec6::Zero());  // 6x1, N_opt blocks

  b_.resize(num_optimization_points_, Vec3::Zero());  // 3x1, M blocks
  y_.resize(num_optimization_points_, Vec3::Zero());  // 3x1, M blocks

  Cinv_.resize(num_optimization_points_,
               Mat3x3::Zero());  // 3x3, M diagonal blocks

  BCinv_.resize(num_optimization_poses_);
  for (Index j = 0; j < num_optimization_poses_; ++j)
    BCinv_[j].resize(num_optimization_points_,
                     Mat6x3::Zero());  // 6x3, N_opt X M blocks

  CinvBt_.resize(num_optimization_points_);
  for (Index i = 0; i < num_optimization_points_; ++i)
    CinvBt_[i].resize(num_optimization_poses_, Mat3x6::Zero());

  BCinvBt_.resize(num_optimization_poses_);
  for (Index j = 0; j < num_optimization_poses_; ++j)
    BCinvBt_[j].resize(num_optimization_poses_,
                       Mat6x6::Zero());  // 6x6, N_opt X N_opt blocks

  BCinv_b_.resize(num_optimization_poses_);     // 6x1, N_opt x 1 blocks
  am_BCinv_b_.resize(num_optimization_poses_);  // 6x1, N_opt x 1 blocks

  Am_BCinvBt_.resize(num_optimization_poses_);
  for (Index j = 0; j < num_optimization_poses_; ++j)
    Am_BCinvBt_[j].resize(num_optimization_poses_,
                          Mat6x6::Zero());  // 6x6, N_opt X N_opt blocks

  Cinv_b_.resize(num_optimization_points_, Vec3::Zero());
  Bt_x_.resize(num_optimization_points_, Vec3::Zero());  // 3x1, M x 1 blocks
  CinvBt_x_.resize(num_optimization_points_, Vec3::Zero());

  // Dynamic matrices
  Am_BCinvBt_mat_.resize(6 * num_optimization_poses_,
                         6 * num_optimization_poses_);
  am_BCinv_b_mat_.resize(6 * num_optimization_poses_, 1);
  x_mat_.resize(6 * num_optimization_poses_, 1);

  Am_BCinvBt_mat_.setZero();
  am_BCinv_b_mat_.setZero();
  x_mat_.setZero();
}

void FullBundleAdjustmentSolverRefactor::CheckPoseAndPointConnectivity() {
  static constexpr int kMinNumObservedPoints = 3;
  static constexpr int kMinNumRelatedPoses = 2;

  for (size_t j_opt = 0; j_opt < j_opt_to_all_point_.size(); ++j_opt) {
    const auto &observed_points = j_opt_to_all_point_[j_opt];
    const size_t num_observed_point = observed_points.size();

    const auto &observed_opt_points = j_opt_to_i_opt_[j_opt];
    const size_t num_observed_optimizable_point = observed_opt_points.size();

    if (num_observed_point < kMinNumObservedPoints) {
      std::stringstream ss;
      ss << TEXT_YELLOW(std::to_string(j_opt) +
                        "-th pose: It might diverge because some frames "
                        "have insufficient related points.\n");
      std::cerr << ss.str();
    }
  }

  for (size_t i_opt = 0; i_opt < i_opt_to_all_pose_.size(); ++i_opt) {
    const auto &related_poses = i_opt_to_all_pose_[i_opt];
    const int num_related_pose = static_cast<int>(related_poses.size());

    const auto &related_opt_poses = i_opt_to_j_opt_[i_opt];
    const int num_observed_optimizable_pose =
        static_cast<int>(related_opt_poses.size());

    if (num_related_pose < kMinNumRelatedPoses) {
      std::stringstream ss;
      ss << TEXT_YELLOW(std::to_string(i_opt) +
                        "-th point: It might diverge because some "
                        "points have insufficient related poses.\n");
      std::cerr << ss.str();
    }
  }
}

void FullBundleAdjustmentSolverRefactor::ResetStorageMatrices() {
  for (Index j = 0; j < num_optimization_poses_; ++j) {
    A_[j].setZero();
    a_[j].setZero();
    x_[j].setZero();
    BCinv_b_[j].setZero();
    am_BCinv_b_[j].setZero();

    for (Index i = 0; i < num_optimization_points_; ++i) {
      B_[j][i].setZero();
      Bt_[i][j].setZero();
      BCinv_[j][i].setZero();
      CinvBt_[i][j].setZero();
    }
    for (Index k = 0; k < num_optimization_poses_; ++k) {
      BCinvBt_[j][k].setZero();
      Am_BCinvBt_[j][k].setZero();
    }
  }
  for (Index i = 0; i < num_optimization_points_; ++i) {
    C_[i].setZero();
    Cinv_[i].setZero();
    b_[i].setZero();
    y_[i].setZero();
    Bt_x_[i].setZero();
    Cinv_b_[i].setZero();
    CinvBt_x_[i].setZero();
  }

  Am_BCinvBt_mat_.setZero();
  am_BCinv_b_mat_.setZero();
  x_mat_.setZero();
}

double FullBundleAdjustmentSolverRefactor::EvaluateCurrentCost() {
  // Evaluate residual only
  Index cnt = 0;
  double error_current = 0.0;
  // See observations
  for (const auto &observation : point_observation_list_) {
    const auto &camera_index = observation.related_camera_id;
    const auto &pixel = observation.pixel;
    const auto &original_pose = observation.related_pose;
    const auto &original_point = observation.related_point;

    // Get intrinsic parameter
    const auto &cam = camera_id_to_camera_map_[camera_index];
    const auto &fx = cam.fx, &fy = cam.fy, &cx = cam.cx, &cy = cam.cy;

    const bool is_optimize_pose =
        (original_pose_to_pose_index_map_.count(original_pose) > 0);
    const bool is_optimize_point =
        (original_point_to_point_index_map_.count(original_point) > 0);

    // Get Tjw (camera pose)
    const Pose &T_jw =
        original_pose_to_inverse_optimized_pose_map_[original_pose];
    const Rotation3D &R_jw = T_jw.linear();
    const Translation3D &t_jw = T_jw.translation();

    // Get Xi (3D point)
    const Point &Xi = original_point_to_optimized_point_map_[original_point];

    // Get pij (pixel observation)
    const Pixel &pij = pixel;

    // Warp Xij = Rjw * Xi + tjw
    const Point Xij = R_jw * Xi + t_jw;

    const Pose &T_cj = cam.camera_to_body_pose;
    const Point Xijc = T_cj * Xij;

    const SolverNumeric &xj = Xijc(0), &yj = Xijc(1), &zj = Xijc(2);
    const SolverNumeric invz = 1.0 / zj;

    // Calculate rij
    Pixel ptw;
    ptw << fx * xj * invz + cx, fy * yj * invz + cy;
    Vec2 rij = ptw - pij;

    error_current += rij.norm();

    ++cnt;
  }  // END for observations

  return error_current;
}

double
FullBundleAdjustmentSolverRefactor::EvaluateCostChangeByQuadraticModel() {
  SolverNumeric estimated_cost_change = 0.0;
  for (Index j_opt = 0; j_opt < num_optimization_poses_; ++j_opt) {
    estimated_cost_change +=
        a_[j_opt].transpose() * x_[j_opt];  // 2*gradient.transpose()*delta_x
    estimated_cost_change += x_[j_opt].transpose() * A_[j_opt] * x_[j_opt];
  }
  for (Index i_opt = 0; i_opt < num_optimization_points_; ++i_opt) {
    estimated_cost_change +=
        b_[i_opt].transpose() * y_[i_opt];  // 2*gradient.transpose()*delta_x
    estimated_cost_change += y_[i_opt].transpose() * C_[i_opt] * y_[i_opt];

    const auto &j_opt_list = i_opt_to_j_opt_[i_opt];
    Mat3x1 Bji_xj = Mat3x1::Zero();
    for (const auto &j_opt : j_opt_list)
      Bji_xj += Bt_[i_opt][j_opt] * x_[j_opt];

    estimated_cost_change += 2.0 * y_[i_opt].transpose() * Bji_xj;
  }
  return -estimated_cost_change;
}

void FullBundleAdjustmentSolverRefactor::
    ReserveCurrentOptimizationParameters() {
  for (Index j_opt = 0; j_opt < num_optimization_poses_; ++j_opt) {
    const auto &original_pose = pose_index_to_original_pose_map_[j_opt];
    auto &T_jw = original_pose_to_inverse_optimized_pose_map_[original_pose];
    reserved_optimized_poses_[j_opt] = T_jw;
  }

  for (Index i_opt = 0; i_opt < num_optimization_points_; ++i_opt) {
    const auto &original_point = point_index_to_original_point_map_[i_opt];
    auto &Xi = original_point_to_optimized_point_map_[original_point];
    reserved_optimized_points_[i_opt] = Xi;
  }
}

void FullBundleAdjustmentSolverRefactor::
    RevertToReservedOptimizationParameters() {
  for (Index j_opt = 0; j_opt < num_optimization_poses_; ++j_opt) {
    const auto &original_pose = pose_index_to_original_pose_map_[j_opt];
    auto &T_jw = original_pose_to_inverse_optimized_pose_map_[original_pose];
    T_jw = reserved_optimized_poses_[j_opt];
  }

  for (Index i_opt = 0; i_opt < num_optimization_points_; ++i_opt) {
    const auto &original_point = point_index_to_original_point_map_[i_opt];
    auto &Xi = original_point_to_optimized_point_map_[original_point];
    Xi = reserved_optimized_points_[i_opt];
  }
}

void FullBundleAdjustmentSolverRefactor::UpdateOptimizationParameters(
    const std::vector<Vec6> &update_of_optimization_pose_parameter_list,
    const std::vector<Vec3> &update_of_optimization_point_parameter_list) {
  // Update step
  for (Index j_opt = 0; j_opt < num_optimization_poses_; ++j_opt) {
    const auto &original_pose = pose_index_to_original_pose_map_[j_opt];
    Pose &inverse_optimized_pose =
        original_pose_to_inverse_optimized_pose_map_[original_pose];

    Pose delta_pose;
    se3Exp<SolverNumeric>(update_of_optimization_pose_parameter_list[j_opt],
                          delta_pose);
    inverse_optimized_pose = delta_pose * inverse_optimized_pose;
  }
  for (Index i_opt = 0; i_opt < num_optimization_points_; ++i_opt) {
    const auto &original_point = point_index_to_original_point_map_[i_opt];
    Point &optimized_point =
        original_point_to_optimized_point_map_[original_point];
    optimized_point.noalias() +=
        update_of_optimization_point_parameter_list[i_opt];
  }
}

inline bool FullBundleAdjustmentSolverRefactor::IsFixedPose(
    Pose *original_pose) {
  return (fixed_original_pose_set_.count(original_pose) > 0);
}

inline bool FullBundleAdjustmentSolverRefactor::IsFixedPoint(
    Point *original_point) {
  return (fixed_original_point_set_.count(original_point) > 0);
}

// For fast calculations for symmetric matrices
inline void
FullBundleAdjustmentSolverRefactor::CalculatePointHessianOnlyUpperTriangle(
    const SolverNumeric weight, const Mat2x3 &Rij, Mat3x3 *Rij_t_Rij) {
  Rij_t_Rij->setZero();

  (*Rij_t_Rij)(0, 0) = weight * (Rij(0, 0) * Rij(0, 0) + Rij(1, 0) * Rij(1, 0));
  (*Rij_t_Rij)(0, 1) = weight * (Rij(0, 0) * Rij(0, 1) + Rij(1, 0) * Rij(1, 1));
  (*Rij_t_Rij)(0, 2) = weight * (Rij(0, 0) * Rij(0, 2) + Rij(1, 0) * Rij(1, 2));

  (*Rij_t_Rij)(1, 1) = weight * (Rij(0, 1) * Rij(0, 1) + Rij(1, 1) * Rij(1, 1));
  (*Rij_t_Rij)(1, 2) = weight * (Rij(0, 1) * Rij(0, 2) + Rij(1, 1) * Rij(1, 2));

  (*Rij_t_Rij)(2, 2) = weight * (Rij(0, 2) * Rij(0, 2) + Rij(1, 2) * Rij(1, 2));
}

inline void
FullBundleAdjustmentSolverRefactor::CalculatePoseHessianOnlyUpperTriangle(
    const SolverNumeric weight, const Mat2x6 &Qij, Mat6x6 *Qij_t_Qij) {
  Qij_t_Qij->setZero();
  const Mat2x6 &wQij = weight * Qij;

  (*Qij_t_Qij)(0, 0) = (wQij(0, 0) * Qij(0, 0) + wQij(1, 0) * Qij(1, 0));
  (*Qij_t_Qij)(0, 1) = (wQij(0, 0) * Qij(0, 1) + wQij(1, 0) * Qij(1, 1));
  (*Qij_t_Qij)(0, 2) = (wQij(0, 0) * Qij(0, 2) + wQij(1, 0) * Qij(1, 2));
  (*Qij_t_Qij)(0, 3) = (wQij(0, 0) * Qij(0, 3) + wQij(1, 0) * Qij(1, 3));
  (*Qij_t_Qij)(0, 4) = (wQij(0, 0) * Qij(0, 4) + wQij(1, 0) * Qij(1, 4));
  (*Qij_t_Qij)(0, 5) = (wQij(0, 0) * Qij(0, 5) + wQij(1, 0) * Qij(1, 5));

  (*Qij_t_Qij)(1, 1) = (wQij(0, 1) * Qij(0, 1) + wQij(1, 1) * Qij(1, 1));
  (*Qij_t_Qij)(1, 2) = (wQij(0, 1) * Qij(0, 2) + wQij(1, 1) * Qij(1, 2));
  (*Qij_t_Qij)(1, 3) = (wQij(0, 1) * Qij(0, 3) + wQij(1, 1) * Qij(1, 3));
  (*Qij_t_Qij)(1, 4) = (wQij(0, 1) * Qij(0, 4) + wQij(1, 1) * Qij(1, 4));
  (*Qij_t_Qij)(1, 5) = (wQij(0, 1) * Qij(0, 5) + wQij(1, 1) * Qij(1, 5));

  (*Qij_t_Qij)(2, 2) = (wQij(0, 2) * Qij(0, 2) + wQij(1, 2) * Qij(1, 2));
  (*Qij_t_Qij)(2, 3) = (wQij(0, 2) * Qij(0, 3) + wQij(1, 2) * Qij(1, 3));
  (*Qij_t_Qij)(2, 4) = (wQij(0, 2) * Qij(0, 4) + wQij(1, 2) * Qij(1, 4));
  (*Qij_t_Qij)(2, 5) = (wQij(0, 2) * Qij(0, 5) + wQij(1, 2) * Qij(1, 5));

  (*Qij_t_Qij)(3, 3) = (wQij(0, 3) * Qij(0, 3) + wQij(1, 3) * Qij(1, 3));
  (*Qij_t_Qij)(3, 4) = (wQij(0, 3) * Qij(0, 4) + wQij(1, 3) * Qij(1, 4));
  (*Qij_t_Qij)(3, 5) = (wQij(0, 3) * Qij(0, 5) + wQij(1, 3) * Qij(1, 5));

  (*Qij_t_Qij)(4, 4) = (wQij(0, 4) * Qij(0, 4) + wQij(1, 4) * Qij(1, 4));
  (*Qij_t_Qij)(4, 5) = (wQij(0, 4) * Qij(0, 5) + wQij(1, 4) * Qij(1, 5));

  (*Qij_t_Qij)(5, 5) = (wQij(0, 5) * Qij(0, 5) + wQij(1, 5) * Qij(1, 5));
}

inline void
FullBundleAdjustmentSolverRefactor::AccumulatePointHessianOnlyUpperTriangle(
    const Mat3x3 &Rij_t_Rij, Mat3x3 *C) {
  (*C)(0, 0) += Rij_t_Rij(0, 0);
  (*C)(0, 1) += Rij_t_Rij(0, 1);
  (*C)(0, 2) += Rij_t_Rij(0, 2);

  (*C)(1, 1) += Rij_t_Rij(1, 1);
  (*C)(1, 2) += Rij_t_Rij(1, 2);

  (*C)(2, 2) += Rij_t_Rij(2, 2);
}

inline void
FullBundleAdjustmentSolverRefactor::AccumulatePoseHessianOnlyUpperTriangle(
    const Mat6x6 &Qij_t_Qij, Mat6x6 *A) {
  (*A)(0, 0) += Qij_t_Qij(0, 0);
  (*A)(0, 1) += Qij_t_Qij(0, 1);
  (*A)(0, 2) += Qij_t_Qij(0, 2);
  (*A)(0, 3) += Qij_t_Qij(0, 3);
  (*A)(0, 4) += Qij_t_Qij(0, 4);
  (*A)(0, 5) += Qij_t_Qij(0, 5);

  (*A)(1, 1) += Qij_t_Qij(1, 1);
  (*A)(1, 2) += Qij_t_Qij(1, 2);
  (*A)(1, 3) += Qij_t_Qij(1, 3);
  (*A)(1, 4) += Qij_t_Qij(1, 4);
  (*A)(1, 5) += Qij_t_Qij(1, 5);

  (*A)(2, 2) += Qij_t_Qij(2, 2);
  (*A)(2, 3) += Qij_t_Qij(2, 3);
  (*A)(2, 4) += Qij_t_Qij(2, 4);
  (*A)(2, 5) += Qij_t_Qij(2, 5);

  (*A)(3, 3) += Qij_t_Qij(3, 3);
  (*A)(3, 4) += Qij_t_Qij(3, 4);
  (*A)(3, 5) += Qij_t_Qij(3, 5);

  (*A)(4, 4) += Qij_t_Qij(4, 4);
  (*A)(4, 5) += Qij_t_Qij(4, 5);

  (*A)(5, 5) += Qij_t_Qij(5, 5);
}

inline void
FullBundleAdjustmentSolverRefactor::FillLowerTriangleByUpperTriangle(
    Mat3x3 *C) {
  (*C)(1, 0) = (*C)(0, 1);
  (*C)(2, 0) = (*C)(0, 2);
  (*C)(2, 1) = (*C)(1, 2);
}

inline void
FullBundleAdjustmentSolverRefactor::FillLowerTriangleByUpperTriangle(
    Mat6x6 *A) {
  (*A)(1, 0) = (*A)(0, 1);
  (*A)(2, 0) = (*A)(0, 2);
  (*A)(3, 0) = (*A)(0, 3);
  (*A)(4, 0) = (*A)(0, 4);
  (*A)(5, 0) = (*A)(0, 5);

  (*A)(2, 1) = (*A)(1, 2);
  (*A)(3, 1) = (*A)(1, 3);
  (*A)(4, 1) = (*A)(1, 4);
  (*A)(5, 1) = (*A)(1, 5);

  (*A)(3, 2) = (*A)(2, 3);
  (*A)(4, 2) = (*A)(2, 4);
  (*A)(5, 2) = (*A)(2, 5);

  (*A)(4, 3) = (*A)(3, 4);
  (*A)(5, 3) = (*A)(3, 5);

  (*A)(5, 4) = (*A)(4, 5);
}

bool FullBundleAdjustmentSolverRefactor::Solve(Options options,
                                               Summary *summary) {
  timer::StopWatch stopwatch(std::string(__func__));
  const auto &max_iteration = options.iteration_handle.max_num_iterations;
  const auto &threshold_convergence_delta_error =
      options.convergence_handle.threshold_cost_change;
  const auto &threshold_convergence_delta_pose =
      options.convergence_handle.threshold_step_size;
  const auto &threshold_huber_loss =
      options.outlier_handle.threshold_huber_loss;
  const auto &threshold_outlier_reproj_error =
      options.outlier_handle.threshold_outlier_rejection;

  const auto &initial_lambda = options.trust_region_handle.initial_lambda;
  const auto &decrease_ratio_lambda =
      options.trust_region_handle.decrease_ratio_lambda;
  const auto &increase_ratio_lambda =
      options.trust_region_handle.increase_ratio_lambda;
  if (summary != nullptr) {
    summary->max_iteration_ = max_iteration;
    summary->threshold_cost_change_ = threshold_convergence_delta_error;
    summary->threshold_step_size_ = threshold_convergence_delta_pose;
    summary->convergence_status_ = true;
  }

  const auto &MAX_ITERATION = max_iteration;
  const auto &THRES_HUBER = threshold_huber_loss;  // pixels
  const auto &THRES_DELTA_XI = threshold_convergence_delta_pose;
  const auto &THRES_DELTA_ERROR = threshold_convergence_delta_error;
  const auto &THRES_REPROJ_ERROR = threshold_outlier_reproj_error;  // pixels

  // Start to solve
  stopwatch.Start();

  FinalizeParameters();
  GetSolverStatistics();

  if (!is_parameter_finalized_)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " + "Solver is not finalized."));

  bool is_success = true;

  // Make connectivity map
  i_opt_to_j_opt_.clear();
  j_opt_to_i_opt_.clear();
  i_opt_to_all_pose_.clear();
  j_opt_to_all_point_.clear();
  i_opt_to_j_opt_.resize(num_optimization_points_);
  j_opt_to_i_opt_.resize(num_optimization_poses_);
  i_opt_to_all_pose_.resize(num_optimization_points_);
  j_opt_to_all_point_.resize(num_optimization_poses_);

  for (const auto &observation : point_observation_list_) {
    const auto &original_pose = observation.related_pose;
    const auto &original_point = observation.related_point;
    const bool is_optimize_pose =
        (original_pose_to_pose_index_map_.count(original_pose) > 0);
    const bool is_optimize_point =
        (original_point_to_point_index_map_.count(original_point) > 0);

    Index i_opt = -1;
    Index j_opt = -1;
    if (is_optimize_point) {
      i_opt = original_point_to_point_index_map_.at(original_point);
      i_opt_to_all_pose_[i_opt].insert(original_pose);
    }
    if (is_optimize_pose) {
      j_opt = original_pose_to_pose_index_map_.at(original_pose);
      j_opt_to_all_point_[j_opt].insert(original_point);
    }

    if (is_optimize_point && is_optimize_pose) {
      i_opt_to_j_opt_[i_opt].insert(j_opt);
      j_opt_to_i_opt_[j_opt].insert(i_opt);
    }
  }

  // Check connectivity
  CheckPoseAndPointConnectivity();

  bool is_converged = false;
  // double error_previous = 1e25;
  double previous_cost = EvaluateCurrentCost();
  SolverNumeric lambda = initial_lambda;
  for (int iteration = 0; iteration < MAX_ITERATION; ++iteration) {
    ResetStorageMatrices();  // Reset A, B, Bt, C, Cinv, a, b, x, y...

    // Iteratively solve. (Levenberg-Marquardt algorithm)
    // Calculate hessian and gradient by observations
    int num_observations = 0;
    for (const auto &observation : point_observation_list_) {
      const auto &camera_id = observation.related_camera_id;
      const auto &original_pose = observation.related_pose;
      const auto &original_point = observation.related_point;
      const auto &observed_pixel = observation.pixel;

      // Get intrinsic parameter
      const auto &camera = camera_id_to_camera_map_[camera_id];
      const Pose &camera_to_body_pose = camera.camera_to_body_pose;

      const bool is_optimize_pose =
          (original_pose_to_pose_index_map_.count(original_pose) > 0);
      const bool is_optimize_point =
          (original_point_to_point_index_map_.count(original_point) > 0);

      // Get Tjw (body to world pose)
      const Pose &T_jw =
          original_pose_to_inverse_optimized_pose_map_[original_pose];
      const auto &R_jw = T_jw.linear();
      const auto &t_jw = T_jw.translation();

      // Get Xi (world point)
      const Point &Xi = original_point_to_optimized_point_map_[original_point];

      // Warp Xij = Rjw * Xi + tjw
      const Point &Xij = R_jw * Xi + t_jw;            // point at body
      const Point &Xijc = camera_to_body_pose * Xij;  // point at camera

      const SolverNumeric xj = Xijc(0);
      const SolverNumeric yj = Xijc(1);
      const SolverNumeric zj = Xijc(2);
      const SolverNumeric invz = 1.0 / zj;
      const SolverNumeric invz2 = invz * invz;
      const SolverNumeric fx_invz = camera.fx * invz;
      const SolverNumeric fy_invz = camera.fy * invz;
      const SolverNumeric x_invz = xj * invz;
      const SolverNumeric y_invz = yj * invz;
      const SolverNumeric fx_x_invz2 = fx_invz * x_invz;
      const SolverNumeric fy_y_invz2 = fy_invz * y_invz;
      // const _BA_Numeric xinvz_yinvz = xinvz * yinvz;

      // Calculate rij
      Pixel projected_pixel;
      projected_pixel << camera.fx * x_invz + camera.cx,
          camera.fy * y_invz + camera.cy;
      Vec2 rij = projected_pixel - observed_pixel;

      // Calculate weight
      const SolverNumeric absrxry = abs(rij.x()) + abs(rij.y());
      // r_prev[cnt] = absrxry;
      SolverNumeric weight =
          (absrxry > THRES_HUBER) ? (THRES_HUBER / absrxry) : 1.0f;

      Vec2 weighted_rij = weight * rij;

      Mat2x3 dpij_dXi;
      dpij_dXi << fx_invz, 0.0, -fx_x_invz2, 0.0, fy_invz, -fy_y_invz2;

      const Rotation3D &R_cj = camera.camera_to_body_pose.linear();
      // _BA_Mat23 dpij_dXi_Rcj = dpij_dXi * R_cj;
      Mat2x3 dpij_dXi_Rcj;
      dpij_dXi_Rcj(0, 0) =
          dpij_dXi(0, 0) * R_cj(0, 0) + dpij_dXi(0, 2) * R_cj(2, 0);
      dpij_dXi_Rcj(0, 1) =
          dpij_dXi(0, 0) * R_cj(0, 1) + dpij_dXi(0, 2) * R_cj(2, 1);
      dpij_dXi_Rcj(0, 2) =
          dpij_dXi(0, 0) * R_cj(0, 2) + dpij_dXi(0, 2) * R_cj(2, 2);
      dpij_dXi_Rcj(1, 0) =
          dpij_dXi(1, 1) * R_cj(1, 0) + dpij_dXi(1, 2) * R_cj(2, 0);
      dpij_dXi_Rcj(1, 1) =
          dpij_dXi(1, 1) * R_cj(1, 1) + dpij_dXi(1, 2) * R_cj(2, 1);
      dpij_dXi_Rcj(1, 2) =
          dpij_dXi(1, 1) * R_cj(1, 2) + dpij_dXi(1, 2) * R_cj(2, 2);

      Mat2x6 Qij;
      Mat2x3 Rij;
      Mat6x2 Qij_t;
      Mat3x2 Rij_t;
      Index j_opt = -1;
      Index i_opt = -1;
      if (is_optimize_pose)  // Jacobian w.r.t. j-th pose
      {
        Mat3x3 m_Xij_skew;
        m_Xij_skew << 0.0, Xij(2), -Xij(1), -Xij(2), 0.0, Xij(0), Xij(1),
            -Xij(0), 0.0;
        Qij << dpij_dXi_Rcj, dpij_dXi_Rcj * m_Xij_skew;
        Qij_t = Qij.transpose();

        Mat6x6 Qij_t_Qij;
        CalculatePoseHessianOnlyUpperTriangle(weight, Qij, &Qij_t_Qij);

        j_opt = original_pose_to_pose_index_map_[original_pose];
        AccumulatePoseHessianOnlyUpperTriangle(
            Qij_t_Qij, &A_[j_opt]);  // A_[j_opt].noalias() += Qij_t_Qij;
        a_[j_opt].noalias() -= (Qij_t * weighted_rij);
      }

      if (is_optimize_point)  // Jacobian w.r.t. i-th point
      {
        Rij = dpij_dXi_Rcj * R_jw;
        Rij_t = Rij.transpose();

        Mat3x3 Rij_t_Rij;
        CalculatePointHessianOnlyUpperTriangle(weight, Rij, &Rij_t_Rij);

        i_opt = original_point_to_point_index_map_[original_point];
        AccumulatePointHessianOnlyUpperTriangle(
            Rij_t_Rij, &C_[i_opt]);  // C_[i_opt].noalias() += Rij_t_Rij;
        b_[i_opt].noalias() -= (Rij_t * weighted_rij);

        if (is_optimize_pose) {
          B_[j_opt][i_opt] = weight * (Qij_t * Rij);
          Bt_[i_opt][j_opt] = B_[j_opt][i_opt].transpose();
        }
      }
      ++num_observations;
    }  // END for observations

    if (num_observations < 1)
      throw std::runtime_error(
          TEXT_RED(std::string(__func__) + ": " + "num_observations < 1\n"));

    // 1) Damping 'A_' diagonal
    const auto lambda_plus_one = 1.0 + lambda;
    for (Index j = 0; j < num_optimization_poses_; ++j) {
      Mat6x6 &A_tmp = A_[j];
      FillLowerTriangleByUpperTriangle(&A_tmp);
      A_tmp(0, 0) *= lambda_plus_one;
      A_tmp(1, 1) *= lambda_plus_one;
      A_tmp(2, 2) *= lambda_plus_one;
      A_tmp(3, 3) *= lambda_plus_one;
      A_tmp(4, 4) *= lambda_plus_one;
      A_tmp(5, 5) *= lambda_plus_one;
    }

    // 2) Damping 'C_' diagonal, and Calculate 'Cinv_' & 'Cinvb_'
    for (Index i = 0; i < num_optimization_points_; ++i) {
      Mat3x3 &C_tmp = C_[i];
      FillLowerTriangleByUpperTriangle(&C_tmp);
      C_tmp(0, 0) *= lambda_plus_one;
      C_tmp(1, 1) *= lambda_plus_one;
      C_tmp(2, 2) *= lambda_plus_one;

      Cinv_[i] = C_[i].ldlt().solve(Mat3x3::Identity());
      Cinv_b_[i] = Cinv_[i] * b_[i];  // FILL STORAGE (10)
    }

    // 3) Calculate 'BCinv_', 'BCinvb_',' BCinvBt_'
    for (Index i = 0; i < num_optimization_points_; ++i) {
      const auto &j_opt_list = i_opt_to_j_opt_[i];
      for (const Index &j : j_opt_list) {
        BCinv_[j][i] = B_[j][i] * Cinv_[i];
        CinvBt_[i][j] = BCinv_[j][i].transpose().eval();
        BCinv_b_[j].noalias() += BCinv_[j][i] * b_[i];

        for (const Index &k : j_opt_list) {
          if (k < j) continue;
          BCinvBt_[j][k].noalias() += BCinv_[j][i] * Bt_[i][k];
        }
      }  // END j_opt
    }    // END i_opt

    for (Index j = 0; j < num_optimization_poses_; ++j)
      for (Index k = j; k < num_optimization_poses_; ++k)
        BCinvBt_[k][j] = BCinvBt_[j][k].transpose();

    for (Index j = 0; j < num_optimization_poses_; ++j) {
      for (Index k = 0; k < num_optimization_poses_; ++k) {
        if (j == k)
          Am_BCinvBt_[j][k] = A_[j] - BCinvBt_[j][k];
        else
          Am_BCinvBt_[j][k] = -BCinvBt_[j][k];
      }
    }

    for (Index j = 0; j < num_optimization_poses_; ++j)
      am_BCinv_b_[j] = a_[j] - BCinv_b_[j];

    // Solve problem.
    // 1) solve x (w.r.t. pose)
    MatrixDynamic &Am_BCinvBt_mat = Am_BCinvBt_mat_;
    MatrixDynamic &am_BCinv_b_mat = am_BCinv_b_mat_;

    Index idx0 = 0;
    for (Index j = 0; j < num_optimization_poses_; ++j, idx0 += 6) {
      Index idx1 = 0;
      for (Index u = 0; u < num_optimization_poses_; ++u, idx1 += 6)
        Am_BCinvBt_mat.block(idx0, idx1, 6, 6) = Am_BCinvBt_[j][u];

      am_BCinv_b_mat.block(idx0, 0, 6, 1) = am_BCinv_b_[j];
    }

    MatrixDynamic &x_mat = x_mat_;
    x_mat = Am_BCinvBt_mat.ldlt().solve(am_BCinv_b_mat);

    idx0 = 0;
    for (Index j = 0; j < num_optimization_poses_; ++j, idx0 += 6)
      x_[j] = x_mat.block<6, 1>(idx0, 0);

    // 2) solve y (w.r.t. point)
    for (Index i = 0; i < num_optimization_points_; ++i) {
      const auto &j_opt_list = i_opt_to_j_opt_[i];
      for (const auto &j : j_opt_list)
        CinvBt_x_[i].noalias() += CinvBt_[i][j] * x_[j];

      y_[i] = Cinv_b_[i] - CinvBt_x_[i];
    }

    double current_cost = 0.0;
    IterationStatus iter_status{IterationStatus::UPDATE};

    // Trust region method
    if (options.solver_type == SolverType::LEVENBERG_MARQUARDT) {
      // 1) reserve parameters
      ReserveCurrentOptimizationParameters();

      // 2) Evaluate the updated cost (reserved unupdated parameters)
      UpdateOptimizationParameters(x_, y_);

      current_cost = EvaluateCurrentCost();
      const auto estimated_cost_change = EvaluateCostChangeByQuadraticModel();
      const auto ratio_of_cost_changes = (current_cost - previous_cost) *
                                         kInverseTranslationScaler /
                                         estimated_cost_change;

      struct {
        const double threshold_update = 0.25;
        const double threshold_trust_more = 0.5;
      } trust_region;

      if (ratio_of_cost_changes > trust_region.threshold_update) {
        iter_status = IterationStatus::UPDATE;
      } else {
        RevertToReservedOptimizationParameters();
        iter_status = IterationStatus::SKIPPED;
      }

      if (ratio_of_cost_changes > trust_region.threshold_trust_more) {
        lambda = std::max(1e-10,
                          static_cast<double>(lambda * decrease_ratio_lambda));
        iter_status = IterationStatus::UPDATE_TRUST_MORE;
      } else if (ratio_of_cost_changes <= trust_region.threshold_update)
        lambda = std::min(100.0,
                          static_cast<double>(lambda * increase_ratio_lambda));
    } else if (options.solver_type == SolverType::GAUSS_NEWTON) {
      UpdateOptimizationParameters(x_, y_);

      current_cost = EvaluateCurrentCost();

      iter_status = IterationStatus::UPDATE;
    }

    // Error calculation
    double average_error = current_cost / static_cast<double>(num_observations);
    const auto cost_change = abs(current_cost - previous_cost);

    // Calculate delta_parameter
    SolverNumeric step_size_pose_norm = 0.0;
    SolverNumeric step_size_point_norm = 0.0;
    for (const auto &xj : x_) step_size_pose_norm += xj.norm();
    for (const auto &yi : y_) step_size_point_norm += yi.norm();
    double total_step_size = step_size_point_norm + step_size_pose_norm;

    const auto average_delta_error =
        cost_change / static_cast<double>(num_observations);
    const auto average_total_step_size =
        (total_step_size) /
        static_cast<double>(num_optimization_poses_ + num_optimization_points_);
    if (average_total_step_size < THRES_DELTA_XI ||
        cost_change < THRES_DELTA_ERROR) {
      // Early convergence.
      is_converged = true;
    }

    if (iteration >= max_iteration - 1) {
      is_converged = false;
    }

    const double time_per_iteration = stopwatch.GetLapTimeFromLatest();

    if (summary != nullptr) {
      OptimizationInfo optimization_info;
      optimization_info.cost = current_cost;
      optimization_info.cost_change = cost_change;
      optimization_info.average_reprojection_error = average_error;

      optimization_info.abs_step = average_total_step_size;
      optimization_info.abs_gradient = 0;
      optimization_info.damping_term = lambda;
      optimization_info.iter_time = time_per_iteration;
      optimization_info.iteration_status = iter_status;

      if (optimization_info.iteration_status == IterationStatus::SKIPPED) {
        optimization_info.cost = previous_cost;
        optimization_info.cost_change = 0;
        optimization_info.average_reprojection_error =
            sqrt(previous_cost / static_cast<double>(num_total_observations_));
      }

      summary->optimization_info_list_.push_back(optimization_info);
    }

    previous_cost = current_cost;

    if (is_converged) break;
  }  // END iteration

  // Finally, update parameters to the original poses / points
  for (Index j_opt = 0; j_opt < num_optimization_poses_; ++j_opt) {
    auto &original_pose = pose_index_to_original_pose_map_[j_opt];
    auto T_jw = original_pose_to_inverse_optimized_pose_map_[original_pose];

    T_jw.translation() *= kInverseTranslationScaler;  // recover scale
    *original_pose = T_jw.inverse();
  }
  for (Index i_opt = 0; i_opt < num_optimization_points_; ++i_opt) {
    auto &original_point = point_index_to_original_point_map_[i_opt];
    auto Xi = original_point_to_optimized_point_map_[original_point];
    *original_point = (Xi * kInverseTranslationScaler);
  }

  const double total_time = stopwatch.GetLapTimeFromStart();
  if (summary != nullptr) {
    summary->convergence_status_ = is_converged;
    summary->total_time_in_millisecond_ = total_time;
  }

  // if (!std::isnan(pose_b2b1_optimized.linear().norm()))
  // {
  //   pose_world_to_current = pose_world_to_current_optimized;
  // }
  // else
  // {
  //   std::cerr << "!! WARNING !! poseonly BA yields NAN value!!"
  //             << ", pose_21_optimized: \n"
  //             << pose_b2b1_optimized.linear() << " " <<
  //             pose_b2b1_optimized.translation() << "\n";
  //   is_success = false; // if nan, do not update.
  // }

  return is_success;
}

bool FullBundleAdjustmentSolverRefactor::SolveByGradientDescent(
    Options options, Summary *summary) {
  timer::StopWatch stopwatch(std::string(__func__));
  const auto &max_iteration = options.iteration_handle.max_num_iterations;
  const auto &threshold_convergence_delta_error =
      options.convergence_handle.threshold_cost_change;
  const auto &threshold_convergence_delta_pose =
      options.convergence_handle.threshold_step_size;
  const auto &threshold_huber_loss =
      options.outlier_handle.threshold_huber_loss;
  const auto &threshold_outlier_reproj_error =
      options.outlier_handle.threshold_outlier_rejection;

  const auto &initial_lambda = options.trust_region_handle.initial_lambda;
  const auto &decrease_ratio_lambda =
      options.trust_region_handle.decrease_ratio_lambda;
  const auto &increase_ratio_lambda =
      options.trust_region_handle.increase_ratio_lambda;
  if (summary != nullptr) {
    summary->max_iteration_ = max_iteration;
    summary->threshold_cost_change_ = threshold_convergence_delta_error;
    summary->threshold_step_size_ = threshold_convergence_delta_pose;
    summary->convergence_status_ = true;
  }

  const auto &MAX_ITERATION = max_iteration;
  const auto &THRES_HUBER = threshold_huber_loss;  // pixels
  const auto &THRES_DELTA_XI = threshold_convergence_delta_pose;
  const auto &THRES_DELTA_ERROR = threshold_convergence_delta_error;
  const auto &THRES_REPROJ_ERROR = threshold_outlier_reproj_error;  // pixels

  // Start to solve
  stopwatch.Start();

  FinalizeParameters();
  GetSolverStatistics();

  if (!is_parameter_finalized_)
    throw std::runtime_error(
        TEXT_RED(std::string(__func__) + ": " + "Solver is not finalized."));

  bool is_success = true;

  // Make connectivity map
  i_opt_to_j_opt_.clear();
  j_opt_to_i_opt_.clear();
  i_opt_to_all_pose_.clear();
  j_opt_to_all_point_.clear();
  i_opt_to_j_opt_.resize(num_optimization_points_);
  j_opt_to_i_opt_.resize(num_optimization_poses_);
  i_opt_to_all_pose_.resize(num_optimization_points_);
  j_opt_to_all_point_.resize(num_optimization_poses_);

  for (const auto &observation : point_observation_list_) {
    const auto &original_pose = observation.related_pose;
    const auto &original_point = observation.related_point;
    const bool is_optimize_pose =
        (original_pose_to_pose_index_map_.count(original_pose) > 0);
    const bool is_optimize_point =
        (original_point_to_point_index_map_.count(original_point) > 0);

    Index i_opt = -1;
    Index j_opt = -1;
    if (is_optimize_point) {
      i_opt = original_point_to_point_index_map_.at(original_point);
      i_opt_to_all_pose_[i_opt].insert(original_pose);
    }
    if (is_optimize_pose) {
      j_opt = original_pose_to_pose_index_map_.at(original_pose);
      j_opt_to_all_point_[j_opt].insert(original_point);
    }

    if (is_optimize_point && is_optimize_pose) {
      i_opt_to_j_opt_[i_opt].insert(j_opt);
      j_opt_to_i_opt_[j_opt].insert(i_opt);
    }
  }

  // Check connectivity
  CheckPoseAndPointConnectivity();

  bool is_converged = false;
  // double error_previous = 1e25;
  double previous_cost = EvaluateCurrentCost();
  SolverNumeric lambda = initial_lambda;
  for (int iteration = 0; iteration < MAX_ITERATION; ++iteration) {
    ResetStorageMatrices();  // Reset A, B, Bt, C, Cinv, a, b, x, y...

    // Iteratively solve. (Levenberg-Marquardt algorithm)
    // Calculate hessian and gradient by observations
    int num_observations = 0;
    for (const auto &observation : point_observation_list_) {
      const auto &camera_id = observation.related_camera_id;
      const auto &original_pose = observation.related_pose;
      const auto &original_point = observation.related_point;
      const auto &observed_pixel = observation.pixel;

      // Get intrinsic parameter
      const auto &camera = camera_id_to_camera_map_[camera_id];
      const Pose &camera_to_body_pose = camera.camera_to_body_pose;

      const bool is_optimize_pose =
          (original_pose_to_pose_index_map_.count(original_pose) > 0);
      const bool is_optimize_point =
          (original_point_to_point_index_map_.count(original_point) > 0);

      // Get Tjw (body to world pose)
      const Pose &T_jw =
          original_pose_to_inverse_optimized_pose_map_[original_pose];
      const auto &R_jw = T_jw.linear();
      const auto &t_jw = T_jw.translation();

      // Get Xi (world point)
      const Point &Xi = original_point_to_optimized_point_map_[original_point];

      // Warp Xij = Rjw * Xi + tjw
      const Point &Xij = R_jw * Xi + t_jw;            // point at body
      const Point &Xijc = camera_to_body_pose * Xij;  // point at camera

      const SolverNumeric xj = Xijc(0);
      const SolverNumeric yj = Xijc(1);
      const SolverNumeric zj = Xijc(2);
      const SolverNumeric invz = 1.0 / zj;
      const SolverNumeric invz2 = invz * invz;
      const SolverNumeric fx_invz = camera.fx * invz;
      const SolverNumeric fy_invz = camera.fy * invz;
      const SolverNumeric x_invz = xj * invz;
      const SolverNumeric y_invz = yj * invz;
      const SolverNumeric fx_x_invz2 = fx_invz * x_invz;
      const SolverNumeric fy_y_invz2 = fy_invz * y_invz;
      // const _BA_Numeric xinvz_yinvz = xinvz * yinvz;

      // Calculate rij
      Pixel projected_pixel;
      projected_pixel << camera.fx * x_invz + camera.cx,
          camera.fy * y_invz + camera.cy;
      Vec2 rij = projected_pixel - observed_pixel;

      // Calculate weight
      const SolverNumeric absrxry = abs(rij.x()) + abs(rij.y());
      // r_prev[cnt] = absrxry;
      SolverNumeric weight =
          (absrxry > THRES_HUBER) ? (THRES_HUBER / absrxry) : 1.0f;

      Vec2 weighted_rij = weight * rij;

      Mat2x3 dpij_dXi;
      dpij_dXi << fx_invz, 0.0, -fx_x_invz2, 0.0, fy_invz, -fy_y_invz2;

      const Rotation3D &R_cj = camera.camera_to_body_pose.linear();
      // _BA_Mat23 dpij_dXi_Rcj = dpij_dXi * R_cj;
      Mat2x3 dpij_dXi_Rcj;
      dpij_dXi_Rcj(0, 0) =
          dpij_dXi(0, 0) * R_cj(0, 0) + dpij_dXi(0, 2) * R_cj(2, 0);
      dpij_dXi_Rcj(0, 1) =
          dpij_dXi(0, 0) * R_cj(0, 1) + dpij_dXi(0, 2) * R_cj(2, 1);
      dpij_dXi_Rcj(0, 2) =
          dpij_dXi(0, 0) * R_cj(0, 2) + dpij_dXi(0, 2) * R_cj(2, 2);
      dpij_dXi_Rcj(1, 0) =
          dpij_dXi(1, 1) * R_cj(1, 0) + dpij_dXi(1, 2) * R_cj(2, 0);
      dpij_dXi_Rcj(1, 1) =
          dpij_dXi(1, 1) * R_cj(1, 1) + dpij_dXi(1, 2) * R_cj(2, 1);
      dpij_dXi_Rcj(1, 2) =
          dpij_dXi(1, 1) * R_cj(1, 2) + dpij_dXi(1, 2) * R_cj(2, 2);

      Mat2x6 Qij;
      Mat2x3 Rij;
      Mat6x2 Qij_t;
      Mat3x2 Rij_t;
      Index j_opt = -1;
      Index i_opt = -1;
      if (is_optimize_pose)  // Jacobian w.r.t. j-th pose
      {
        Mat3x3 m_Xij_skew;
        m_Xij_skew << 0.0, Xij(2), -Xij(1), -Xij(2), 0.0, Xij(0), Xij(1),
            -Xij(0), 0.0;
        Qij << dpij_dXi_Rcj, dpij_dXi_Rcj * m_Xij_skew;
        Qij_t = Qij.transpose();

        j_opt = original_pose_to_pose_index_map_[original_pose];
        a_[j_opt].noalias() -= (Qij_t * weighted_rij);
      }

      if (is_optimize_point)  // Jacobian w.r.t. i-th point
      {
        Rij = dpij_dXi_Rcj * R_jw;
        Rij_t = Rij.transpose();

        i_opt = original_point_to_point_index_map_[original_point];
        b_[i_opt].noalias() -= (Rij_t * weighted_rij);
      }
      ++num_observations;
    }  // END for observations

    if (num_observations < 1)
      throw std::runtime_error(
          TEXT_RED(std::string(__func__) + ": " + "num_observations < 1\n"));

    double current_cost = 0.0;
    IterationStatus iter_status{IterationStatus::UPDATE};

    constexpr double max_pose_step = 0.001;
    constexpr double max_point_step = 0.001;

    for (auto &vec : a_)
      if (vec.norm() > max_pose_step) vec = vec * (max_pose_step / vec.norm());
    for (auto &vec : b_)
      if (vec.norm() > max_point_step)
        vec = vec * (max_point_step / vec.norm());

    UpdateOptimizationParameters(a_, b_);

    current_cost = EvaluateCurrentCost();

    iter_status = IterationStatus::UPDATE;

    // Error calculation
    double average_error = current_cost / static_cast<double>(num_observations);
    const auto cost_change = abs(current_cost - previous_cost);

    // Calculate delta_parameter
    SolverNumeric step_size_pose_norm = 0.01;
    SolverNumeric step_size_point_norm = 0.01;
    for (const auto &xj : a_) step_size_pose_norm += xj.norm();
    for (const auto &yi : b_) step_size_point_norm += yi.norm();
    double total_step_size = step_size_point_norm + step_size_pose_norm;

    const auto average_delta_error =
        cost_change / static_cast<double>(num_observations);
    const auto average_total_step_size =
        (total_step_size) /
        static_cast<double>(num_optimization_poses_ + num_optimization_points_);
    if (average_total_step_size < THRES_DELTA_XI ||
        cost_change < THRES_DELTA_ERROR) {
      // Early convergence.
      is_converged = true;
    }

    if (iteration >= max_iteration - 1) {
      is_converged = false;
    }

    const double time_per_iteration = stopwatch.GetLapTimeFromLatest();

    if (summary != nullptr) {
      OptimizationInfo optimization_info;
      optimization_info.cost = current_cost;
      optimization_info.cost_change = cost_change;
      optimization_info.average_reprojection_error = average_error;

      optimization_info.abs_step = average_total_step_size;
      optimization_info.abs_gradient = 0;
      optimization_info.damping_term = lambda;
      optimization_info.iter_time = time_per_iteration;
      optimization_info.iteration_status = iter_status;

      if (optimization_info.iteration_status == IterationStatus::SKIPPED) {
        optimization_info.cost = previous_cost;
        optimization_info.cost_change = 0;
        optimization_info.average_reprojection_error =
            sqrt(previous_cost / static_cast<double>(num_total_observations_));
      }

      summary->optimization_info_list_.push_back(optimization_info);
    }

    previous_cost = current_cost;

    if (is_converged) break;
  }  // END iteration

  // Finally, update parameters to the original poses / points
  for (Index j_opt = 0; j_opt < num_optimization_poses_; ++j_opt) {
    auto &original_pose = pose_index_to_original_pose_map_[j_opt];
    auto T_jw = original_pose_to_inverse_optimized_pose_map_[original_pose];

    T_jw.translation() *= kInverseTranslationScaler;  // recover scale
    *original_pose = T_jw.inverse();
  }
  for (Index i_opt = 0; i_opt < num_optimization_points_; ++i_opt) {
    auto &original_point = point_index_to_original_point_map_[i_opt];
    auto Xi = original_point_to_optimized_point_map_[original_point];
    *original_point = (Xi * kInverseTranslationScaler);
  }

  const double total_time = stopwatch.GetLapTimeFromStart();
  if (summary != nullptr) {
    summary->convergence_status_ = is_converged;
    summary->total_time_in_millisecond_ = total_time;
  }

  return is_success;
}

template <typename T>
void FullBundleAdjustmentSolverRefactor::se3Exp(
    const Eigen::Matrix<T, 6, 1> &xi, Eigen::Transform<T, 3, 1> &pose) {
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
  wx << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  if (theta < 1e-7) {
    R = Eigen::Matrix<T, 3, 3>::Identity() + wx + 0.5 * wx * wx;
    V = Eigen::Matrix<T, 3, 3>::Identity() + 0.5 * wx +
        wx * wx * 0.33333333333333333333333333;
  } else {
    R = Eigen::Matrix<T, 3, 3>::Identity() + (sin(theta) / theta) * wx +
        ((1 - cos(theta)) / (theta * theta)) * (wx * wx);
    V = Eigen::Matrix<T, 3, 3>::Identity() +
        ((1 - cos(theta)) / (theta * theta)) * wx +
        ((theta - sin(theta)) / (theta * theta * theta)) * (wx * wx);
  }
  t = V * v;

  // assign rigid body transformation matrix (in SE(3))
  pose.linear() = R;
  pose.translation() = t;
}

template <typename T>
void FullBundleAdjustmentSolverRefactor::so3Exp(const Eigen::Matrix<T, 3, 1> &w,
                                                Eigen::Matrix<T, 3, 3> &R) {
  // initialize variables
  T theta = 0.0;
  Eigen::Matrix<T, 3, 3> wx;

  theta = std::sqrt(w.transpose() * w);
  wx << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

  Eigen::Matrix<T, 3, 3> wxwx = wx * wx;
  if (theta < 1e-9) {
    R = Eigen::Matrix<T, 3, 3>::Identity() + wx + 0.5 * wxwx;
  } else {
    double invtheta2 = 1.0 / (theta * theta);
    R = Eigen::Matrix<T, 3, 3>::Identity() + (sin(theta) / theta) * wx +
        ((1 - cos(theta)) * invtheta2) * wxwx;
  }
}

}  // namespace analytic_solver
}  // namespace visual_navigation

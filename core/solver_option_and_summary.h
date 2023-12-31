#ifndef _SOLVER_OPTION_AND_SUMMARY_H_
#define _SOLVER_OPTION_AND_SUMMARY_H_
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ceres/ceres.h"

namespace visual_navigation {
namespace analytic_solver {
#define TEXT_RED(str) (std::string("\033[0;31m") + str + std::string("\033[0m"))
#define TEXT_GREEN(str) \
  (std::string("\033[0;32m") + str + std::string("\033[0m"))
#define TEXT_YELLOW(str) \
  (std::string("\033[0;33m") + str + std::string("\033[0m"))
#define TEXT_BLUE(str) \
  (std::string("\033[0;34m") + str + std::string("\033[0m"))
#define TEXT_MAGENTA(str) \
  (std::string("\033[0;35m") + str + std::string("\033[0m"))
#define TEXT_CYAN(str) \
  (std::string("\033[0;36m") + str + std::string("\033[0m"))

enum class SolverType {
  UNDEFINED = -1,
  GRADIENT_DESCENT = 0,
  GAUSS_NEWTON = 1,
  LEVENBERG_MARQUARDT = 2
};
enum class IterationStatus {
  UNDEFINED = -1,
  UPDATE = 0,
  UPDATE_TRUST_MORE = 1,
  SKIPPED = 2
};
struct OptimizationInfo {
  double cost{-1.0};
  double cost_change{-1.0};
  double average_reprojection_error{-1.0};
  double abs_gradient{-1.0};
  double abs_step{-1.0};
  double damping_term{-1.0};
  double iter_time{-1.0};
  IterationStatus iteration_status{IterationStatus::UNDEFINED};
};
class Options {
  friend class PoseOnlyBundleAdjustmentSolver;
  friend class FullBundleAdjustmentSolver;

 public:
  Options() {}
  ~Options() {}

  SolverType solver_type{SolverType::GAUSS_NEWTON};
  struct {
    float threshold_step_size{1e-5};
    float threshold_cost_change{1e-5};
  } convergence_handle;
  struct {
    float threshold_huber_loss{1.0};
    float threshold_outlier_rejection{2.0};
  } outlier_handle;
  struct {
    int max_num_iterations{50};
  } iteration_handle;
  struct {
    float initial_lambda{100.0};
    float decrease_ratio_lambda{0.33f};
    float increase_ratio_lambda{3.0f};
  } trust_region_handle;
};

class Summary {
  friend class PoseOnlyBundleAdjustmentSolver;
  friend class FullBundleAdjustmentSolver;
  friend class FullBundleAdjustmentSolverRefactor;

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

}  // namespace analytic_solver
}  // namespace visual_navigation

#endif
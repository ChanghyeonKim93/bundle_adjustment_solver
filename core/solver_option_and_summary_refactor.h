#ifndef _SOLVER_OPTION_AND_SUMMARY_REFACTOR_H_
#define _SOLVER_OPTION_AND_SUMMARY_REFACTOR_H_
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

class Options {
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

struct OverallSummary {
  double initial_cost{0.0};
  double final_cost{0.0};
  bool is_converged{true};
  int num_residuals{-1};
  int num_residual_blocks{-1};
  int num_parameters{-1};
  int num_parameter_blocks{-1};
};

struct IterationSummary {
  int iteration{-1};
  double iteration_time_in_seconds{0.0};
  double cumulative_time_in_seconds{0.0};
  double cost{0.0};
  double cost_change{0.0};
  double gradient_norm{0.0};
  double step_norm{0.0};
  double step_size{0.0};
  double step_solver_time_in_seconds{0.0};
  double trust_region_radius{0.0};
  IterationStatus iteration_status{IterationStatus::UNDEFINED};
};

class Summary {
 public:
  Summary();
  ~Summary();
  std::string BriefReport() const;
  double GetTotalTimeInSeconds() const;

 public:
  void SetIterationSummary(const IterationSummary& iteration_summary);
  void SetOverallSummary(const OverallSummary& overall_summary);

 private:
  bool is_overall_summary_set_;
  bool is_iteration_summary_set_;
  OverallSummary overall_summary_;
  std::vector<IterationSummary> iteration_summary_list_;
};

}  // namespace analytic_solver

#endif  // _SOLVER_OPTION_AND_SUMMARY_REFACTOR_H_
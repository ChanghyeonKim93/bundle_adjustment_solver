#include "solver_option_and_summary.h"

namespace analytic_solver {
Summary::Summary() {}
Summary::~Summary() {}
const double Summary::GetTotalTimeInSecond() const { return total_time_in_millisecond_ * 0.001; }

std::string Summary::BriefReport() {
  const auto default_precision{std::cout.precision()};
  std::stringstream ss;
  ss << "itr ";            // 5
  ss << "  total_cost  ";  // 14
  ss << " avg.reproj. ";   // 13
  ss << " cost_change ";   // 13
  ss << " |step|  ";       // 10
  ss << " |gradient| ";    // 12
  ss << " damp_term ";     // 12
  ss << " itr_time[ms] ";  // 11
  ss << "itr_stat\n";

  const size_t num_iterations = optimization_info_list_.size();
  for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
    const OptimizationInfo &optimization_info = optimization_info_list_[iteration];
    ss << std::setw(3) << iteration << " ";
    ss << " " << std::scientific << optimization_info.cost;
    ss << "    " << std::setprecision(2) << std::scientific << optimization_info.average_reprojection_error;
    ss << "    " << std::setprecision(2) << std::scientific << optimization_info.cost_change;
    ss << "   " << std::setprecision(2) << std::scientific << optimization_info.abs_step;
    ss << "   " << std::setprecision(2) << std::scientific << optimization_info.abs_gradient;
    ss << "    " << std::setprecision(2) << std::scientific << optimization_info.damping_term;
    ss << "   " << std::setprecision(2) << std::scientific << optimization_info.iter_time;
    switch (optimization_info.iteration_status) {
      case iteration_status_enum::UPDATE:
        ss << "     "
           << "UPDATE";
        break;
      case iteration_status_enum::SKIPPED:
        ss << "     " << TEXT_YELLOW(" SKIP ");
        break;
      case iteration_status_enum::UPDATE_TRUST_MORE:
        ss << "     " << TEXT_GREEN("UPDATE");
        break;
      default:
        ss << "     ";
    }
    ss << "\n";
    ss << std::setprecision(default_precision);  // restore defaults
  }
  ss << std::setprecision(5);
  ss << "Analytic Solver Report:\n";
  ss << "  Iterations      : " << num_iterations << "\n";
  ss << "  Total time      : " << total_time_in_millisecond_ * 0.001 << " [second]\n";
  ss << "  Initial cost    : " << optimization_info_list_.front().cost << "\n";
  ss << "  Final cost      : " << optimization_info_list_.back().cost << "\n";
  ss << "  Initial reproj. : " << optimization_info_list_.front().average_reprojection_error << " [pixel]\n";
  ss << "  Final reproj.   : " << optimization_info_list_.back().average_reprojection_error << " [pixel]\n";
  ss << ", Termination     : " << (convergence_status_ ? TEXT_GREEN("CONVERGENCE") : TEXT_YELLOW("NO_CONVERGENCE"))
     << "\n";
  if (max_iteration_ == num_iterations) {
    ss << TEXT_YELLOW(" WARNIING: MAX ITERATION is reached ! The solution could be local minima.\n");
  }
  ss << std::setprecision(default_precision);  // restore defaults
  return ss.str();
}
};  // namespace analytic_solver

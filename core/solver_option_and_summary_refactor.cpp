#include "solver_option_and_summary_refactor.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace analytic_solver {

Summary::Summary()
    : is_overall_summary_set_(false), is_iteration_summary_set_(false) {}

Summary::~Summary() {}

std::string Summary::BriefReport() const {
  const std::streamsize default_precision{std::cout.precision()};
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
  const size_t num_iterations = iteration_summary_list_.size();
  for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
    const IterationSummary& iteration_summary =
        iteration_summary_list_[iteration];

    ss << std::setw(3) << iteration << " ";
    ss << " " << std::scientific << iteration_summary.cost;
    ss << "    " << std::setprecision(2) << std::scientific
       << iteration_summary.cost / static_cast<double>(1.0);
    ss << "    " << std::setprecision(2) << std::scientific
       << iteration_summary.cost_change;
    ss << "   " << std::setprecision(2) << std::scientific
       << iteration_summary.step_norm;
    ss << "   " << std::setprecision(2) << std::scientific
       << iteration_summary.gradient_norm;
    ss << "   " << std::setprecision(2) << std::scientific
       << iteration_summary.trust_region_radius;
    ss << "   " << std::setprecision(2) << std::scientific
       << iteration_summary.iteration_time_in_seconds;

    switch (iteration_summary.iteration_status) {
      case IterationStatus::UPDATE:
        ss << "     "
           << "UPDATE";
        break;
      case IterationStatus::SKIPPED:
        ss << "     " << TEXT_YELLOW(" SKIP ");
        break;
      case IterationStatus::UPDATE_TRUST_MORE:
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
  ss << "  Total time      : " << total_time_in_millisecond_ * 0.001
     << " [second]\n";
  ss << "  Initial cost    : " << optimization_info_list_.front().cost << "\n";
  ss << "  Final cost      : " << optimization_info_list_.back().cost << "\n";
  ss << "  Initial reproj. : "
     << optimization_info_list_.front().average_reprojection_error
     << " [pixel]\n";
  ss << "  Final reproj.   : "
     << optimization_info_list_.back().average_reprojection_error
     << " [pixel]\n";
  ss << ", Termination     : "
     << (convergence_status_ ? TEXT_GREEN("CONVERGENCE")
                             : TEXT_YELLOW("NO_CONVERGENCE"))
     << "\n";
  if (max_iteration_ == num_iterations) {
    ss << TEXT_YELLOW(
        " WARNIING: MAX ITERATION is reached ! The solution could be local "
        "minima.\n");
  }
  ss << std::setprecision(default_precision);  // restore defaults
  return ss.str();
}

double Summary::GetTotalTimeInSeconds() const {
  if (is_iteration_summary_set_)
    throw std::runtime_error("iteration_summary is not set.");
  const double total_time_in_seconds =
      iteration_summary_list_.back().cumulative_time_in_seconds;
  return total_time_in_seconds;
}

void Summary::SetIterationSummary(const IterationSummary& iteration_summary) {
  iteration_summary_list_.push_back(iteration_summary);
  is_iteration_summary_set_ = true;
}

void Summary::SetOverallSummary(const OverallSummary& overall_summary) {
  overall_summary_ = overall_summary;
  is_overall_summary_set_ = true;
}

}  // namespace analytic_solver

#include <iostream>

#include "full_bundle_adjustment_solver.h"
#include "full_bundle_adjustment_solver_analytic.h"
#include "types.h"

int main() {
  FullBundleAdjustmentSolver *ba_solver = new FullBundleAdjustmentSolverAnalytic();

  return 0;
}
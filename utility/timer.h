#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <iostream>
#include <string>

namespace timer {
void tic();
double toc(bool flag_verbose);        // return elapsed time from "tic()" in milliseconds.
const std::string currentDateTime();  // Get current data/time, format is yyyy-mm-dd.hh:mm:ss

class StopWatch {
 public:
  StopWatch(const std::string &stopwatch_name);
  ~StopWatch();

 public:
  double Start(const bool flag_verbose = false);
  double GetLapTimeFromStart(const bool flag_verbose = false);
  double GetLapTimeFromLatest(const bool flag_verbose = false);
  double Stop(const bool flag_verbose = false);

 private:
  std::string timer_name_;

  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point intermediate_;
  std::chrono::high_resolution_clock::time_point end_;

 public:
  inline static std::chrono::high_resolution_clock::time_point ref_time_ = std::chrono::high_resolution_clock::now();
};
}  // namespace timer
#endif
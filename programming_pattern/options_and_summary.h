#ifndef OPTIONS_AND_SUMMARY_H_
#define OPTIONS_AND_SUMMARY_H_

class Options {
 public:
  Options() {}
  ~Options() {}

 private:
  struct {
    float huber_threshold{1.0f};
  } outlier_handle;
  struct {
    float parameter_step_threshold{1e-6f};
    float cost_change_threshold{1e-6f};
  } convergence_handle;
};

class Summary {
 public:
  Summary() {}
  ~Summary() {}
};

#endif
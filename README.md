# bundle_adjustment_solver
### Usage
```
git clone "https://github.com/ChanghyeonKim93/bundle_adjustment_solver"
cd bundle_adjustment_solver
mkdir build
cd build
cmake ..
make -j4
```

### Brief usage
```c++
// Make solver object
analytic_solver::FullBundleAdjustmentSolver ba_solver;

// Add camera list you want to consider
for (const auto &camera : camera_list)
  ba_solver.AddCamera(camera);

// Add frame poses you want to consider
for (auto &[frame_id, stereoframe] : stereoframe_pool)
  ba_solver.AddPose(&stereoframe.pose);

// Add world positions you want to consider
for (auto &[landmark_id, landmark] : landmark_pool)
  ba_solver.AddPoint(&landmark.world_position);

// Fix frame poses you want
std::vector<int> fixed_pose_list;
for (int index = 0; index < num_fixed_poses; ++index)
  fixed_pose_list.push_back(index);
for (const auto &frame_id : fixed_pose_list) {
  auto &pose = stereoframe_pool[frame_id].pose;
  ba_solver.MakePoseFixed(&pose);
}

// Fix world positions you want
ba_solver.MakePointFixed({});   // Make some points which you want not to update

// Finalize setting variables
ba_solver.FinalizeParameters(); // This line is necessary before solving the problem.

// Add observations
for (auto &[frame_id, stereoframe] : stereoframe_pool) {
  int camera_id = 0; // left camera
  for (int index = 0; index < stereoframe.left.landmark_id_list.size(); ++index)   {
    auto &landmark = landmark_pool[stereoframe.left.landmark_id_list[index]];
    const auto &left_pixel = stereoframe.left.pixel_list[index];
    ba_solver.AddObservation(camera_id, &stereoframe.pose, &landmark.world_position, left_pixel);
  }

  camera_id = 1; // right camera
  for (int index = 0; index < stereoframe.right.landmark_id_list.size(); ++index) {
    auto &landmark = landmark_pool[stereoframe.right.landmark_id_list[index]];
    const auto &right_pixel = stereoframe.right.pixel_list[index];
    ba_solver.AddObservation(camera_id, &stereoframe.pose, &landmark.world_position, right_pixel);
  }
}

ba_solver.GetSolverStatistics();

// Solve!
analytic_solver::Options options;
options.iteration_handle.max_num_iterations = 3000;
analytic_solver::Summary summary;
ba_solver.Solve(options, &summary); // Solve the problem

std::cout << summary.BriefReport() << std::endl;
```

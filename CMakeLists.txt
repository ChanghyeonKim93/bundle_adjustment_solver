cmake_minimum_required(VERSION 3.10)
project(bundle_adjustment_solver)
message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -march=native")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories ("/usr/include/opencv4/")
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/)

set(SRC
  utility/timer.cpp
  utility/simd_library.cpp
  utility/geometry_library.cpp

  core/solver_option_and_summary.cpp
  core/pose_only_bundle_adjustment_solver_ceres.cpp
  core/pose_only_bundle_adjustment_solver.cpp
  core/full_bundle_adjustment_solver.cpp
  core/full_bundle_adjustment_solver_refactor.cpp
)

# TEST
add_executable(${CMAKE_PROJECT_NAME}_test_ba test/test_ba.cpp ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_test_ba ${Eigen3_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(${CMAKE_PROJECT_NAME}_test_warp_simd test/test_warp_simd.cpp ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_test_warp_simd ${Eigen3_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(${CMAKE_PROJECT_NAME}_test_compare_ceres_vs_native test/test_compare_ceres_vs_native.cpp ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_test_compare_ceres_vs_native ${Eigen3_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(${CMAKE_PROJECT_NAME}_test_ba_refactor test/test_ba_refactor.cpp ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_test_ba_refactor ${Eigen3_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(test_projection_of_3d_point test/test_projection_of_3d_point.cc ${SRC})
target_link_libraries(test_projection_of_3d_point ${Eigen3_LIBRARIES} ${OpenCV_LIBS})

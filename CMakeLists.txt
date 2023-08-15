cmake_minimum_required(VERSION 3.10)
project(test_bundle_adjustment)
message("   -- [PROJECT]: " ${CMAKE_PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -march=native")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories ("/usr/include/opencv4/")
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/)

set(SRC
  utility/timer.cpp
  utility/simd_library.cpp

  core/pose_only_bundle_adjustment_solver.cpp
  core/full_bundle_adjustment_solver.cpp
)

# TEST
add_executable(${CMAKE_PROJECT_NAME}_ba test_ba.cpp ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_ba ${Eigen3_LIBRARIES} ${OpenCV_LIBS})

add_executable(${CMAKE_PROJECT_NAME}_simd test_warp_simd.cpp ${SRC})
target_link_libraries(${CMAKE_PROJECT_NAME}_simd ${Eigen3_LIBRARIES} ${OpenCV_LIBS})


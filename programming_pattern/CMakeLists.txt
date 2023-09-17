cmake_minimum_required(VERSION 3.9)
message("CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})
project(programming_pattern)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
set(SRC ${SRC} 
  full_bundle_adjustment_solver_analytic.cpp
)

include_directories(${INCLUDE_DIRS})

add_executable (test_factory_pattern main_factory_pattern.cpp ${SRC})
target_link_libraries(test_factory_pattern PUBLIC ${OpenCV_LIBS})
target_include_directories(test_factory_pattern PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})
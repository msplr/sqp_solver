cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(solver_test VERSION 0.1.0 LANGUAGES CXX)

find_package(GTest REQUIRED)

set(CMAKE_BUILD_TYPE Debug)

add_executable(
    tests
    qp_solver_test.cpp
    # qp_solver_sparse_test.cpp
    sqp_test.cpp
    sqp_test_autodiff.cpp
    bfgs_test.cpp
    test_main.cpp
    )
target_link_libraries(
    tests
    sqp_solver
    Eigen3::Eigen
    ${GTEST_BOTH_LIBRARIES}
    pthread
    )

# Tests should be Debug build to enable Eigen asserts
set_target_properties(tests PROPERTIES CMAKE_BUILD_TYPE "Debug")
target_include_directories(tests PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(tests PRIVATE "-Wall")

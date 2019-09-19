cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sqp_solver VERSION 0.1.0
    LANGUAGES CXX
    # DESCRIPTION "A header only SQP solver using Eigen"
    # HOMEPAGE_URL "https://github.com/nuft/sqp_solver"
    )

option(SQP_SOLVER_BUILD_TESTS "Enable testing" ON)
option(SQP_SOLVER_INSTALL "Enable generation of install target" ON)

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include/)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

if(SQP_SOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

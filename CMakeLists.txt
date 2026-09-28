cmake_minimum_required(VERSION 3.24)
project(eth3d_scorer VERSION 0.1.0 LANGUAGES CXX)

option(ETH3D_SCORER_BUILD_TESTS "Build unit tests" ON)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)

add_library(eth3d_score
    src/cloud.cpp
    src/config.cpp
    src/dataset_input.cpp
    src/eth3d_trajectory_metrics.cpp
    src/math.cpp
    src/scorer.cpp
    src/trajectory.cpp
)
add_library(eth3d::score ALIAS eth3d_score)

target_compile_features(eth3d_score PUBLIC cxx_std_20)
target_include_directories(eth3d_score
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(eth3d_score PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(eth3d_score PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(eth3d-scorer app/main.cpp)
target_link_libraries(eth3d-scorer PRIVATE eth3d::score)

include(CTest)
if(BUILD_TESTING AND ETH3D_SCORER_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_executable(eth3d_score_tests
        tests/cloud_test.cpp
        tests/config_test.cpp
        tests/eth3d_trajectory_metrics_test.cpp
        tests/math_test.cpp
        tests/scorer_test.cpp
        tests/trajectory_test.cpp
    )
    target_link_libraries(eth3d_score_tests PRIVATE eth3d::score GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(eth3d_score_tests)
endif()


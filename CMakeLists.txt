cmake_minimum_required(VERSION 2.8.3)
project(test)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED)
find_package(Eigen3)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    "./include/"
)

add_executable(vins
    ./src/test.cpp
    ./src/analogCamera.cpp
    ./src/calibration.cpp
    ./src/featureTrack.cpp
    ./src/solveVinsHelp.cpp
    ./src/visionEstimator.cpp
)

target_link_libraries(vins ${OpenCV_LIBS})
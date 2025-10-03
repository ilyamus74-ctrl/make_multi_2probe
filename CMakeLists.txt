cmake_minimum_required(VERSION 3.10)
project(CalibrationWeb)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibrate_web calibrate_web.cpp)
target_link_libraries(calibrate_web ${OpenCV_LIBS} pthread)
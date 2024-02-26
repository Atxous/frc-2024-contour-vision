cmake_minimum_required(VERSION 3.0.0)
project(frc_localization VERSION 0.1.0)

# set the directory for opencv bin

set(OpenCV_DIR "")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES enhancement.cpp color_constancy2.cpp segmentation.cpp)  # Specify the extension .cpp for color_constancy2

add_executable(frc_localization ${SOURCES})

target_link_libraries(frc_localization ${OpenCV_LIBS})

# Add the directory containing calculations.h to the include directories
target_include_directories(frc_localization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

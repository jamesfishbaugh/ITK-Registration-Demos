cmake_minimum_required(VERSION 3.10.2)

project(BSplineRegistration)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


add_executable(BSplineRegistration BSplineRegistration.cxx)
target_link_libraries(BSplineRegistration ${ITK_LIBRARIES})

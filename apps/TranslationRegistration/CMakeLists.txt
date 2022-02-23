cmake_minimum_required(VERSION 3.10.2)

project(TranslationRegistration)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(TranslationRegistration TranslationRegistration.cxx)
target_link_libraries(TranslationRegistration ${ITK_LIBRARIES})

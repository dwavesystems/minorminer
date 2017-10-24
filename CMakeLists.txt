cmake_minimum_required(VERSION 3.0)

project(minorminer)
# set(VERSION_MAJOR 2)
# set(VERSION_MINOR 5)
# set(VERSION_PATCH 0)

option(MINORMINER_BUILD_TESTS "Build unit tests." ON)


add_library(minorminer INTERFACE)
target_include_directories(minorminer INTERFACE ${PROJECT_SOURCE_DIR}/include)


if(MINORMINER_BUILD_TESTS)
    add_subdirectory(tests)
endif()

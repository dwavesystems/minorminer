
set(CMAKE_C_OUTPUT_EXTENSION_REPLACE ON)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

# Set compiler flags for gcc
if(CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=c++1y -Wno-unused-variable -Wno-format-security")
endif()

add_library(minorminer_clib SHARED
  find_embedding_clib.cpp
  graph_clib.cpp
  local_interaction_clib.cpp
  optional_parameters_clib.cpp
  stl_clib.cpp)

target_include_directories(minorminer_clib PRIVATE include .)
set_target_properties(minorminer_clib PROPERTIES VERSION 1.0 SOVERSION 1
   PUBLIC_HEADER minorminer_clib.h)

add_executable(run_tests run_tests.c)
target_link_libraries(run_tests pthread minorminer_clib)

find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(run_tests PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(run_tests "${CMAKE_THREAD_LIBS_INIT}")
endif()

enable_testing()
add_test(MINORMINER_CLIB run_tests)
add_custom_command(TARGET run_tests COMMENT "Run tests" POST_BUILD COMMAND run_tests)

install(TARGETS minorminer_clib LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include)


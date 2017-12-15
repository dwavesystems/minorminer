
set(CMAKE_C_OUTPUT_EXTENSION_REPLACE ON)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

# Set compiler flags for gcc
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -std=c++1y -fprofile-arcs -ftest-coverage -DCPPDEBUG")
endif()

add_executable(run_tests run_tests.cpp test_input_graph.cpp test_components.cpp test_pairing_queue.cpp)
target_link_libraries(run_tests gtest pthread minorminer)

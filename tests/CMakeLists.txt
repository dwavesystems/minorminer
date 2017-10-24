
set(CMAKE_C_OUTPUT_EXTENSION_REPLACE ON)
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

# Set compiler flags for gcc
if(CMAKE_C_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -std=c++14 -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -Wall -Wextra -std=gnu14 -fprofile-arcs -ftest-coverage")
endif()

add_executable(run_tests run_tests.cpp test_input_graph.cpp test_components.cpp test_pairing_queue.cpp)
target_link_libraries(run_tests gtest pthread minorminer)

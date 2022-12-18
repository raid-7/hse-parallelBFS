cmake_minimum_required(VERSION 3.20)
project(bfs)

set(CMAKE_CXX_STANDARD 20)

find_package(TBB REQUIRED)

add_executable(sort main.cpp)
target_link_libraries(sort PRIVATE TBB::tbb)

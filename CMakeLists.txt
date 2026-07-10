cmake_minimum_required(VERSION 3.23)

project(TransformerFromScratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(train main.cppm)

target_include_directories(train PRIVATE nn)

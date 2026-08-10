cmake_minimum_required(VERSION 3.28)

project(TransformerFromScratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(NN)

target_sources(NN
  PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
      nn/Node.cppm
      nn/Layer.cppm
      nn/NeuralNetwork.cppm
      nn/utils/loss/loss.cppm
      loader/mnist/loader.cppm
)

add_executable(train)
target_sources(train PRIVATE main.cppm)

target_include_directories(train PRIVATE nn)

target_link_libraries(train PRIVATE NN)

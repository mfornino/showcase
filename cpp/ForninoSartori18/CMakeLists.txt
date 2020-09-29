cmake_minimum_required(VERSION 3.9)
project(many_classes_cpp)

# Set C++ standard (default is 11)
set(CMAKE_CXX_STANDARD 11)

# Include additional header folders
# MACPORTS
#include_directories(/opt/local/include/)

# HOMEBREW
#include_directories(/usr/local/Cellar/)

# HOME DIRECTORY
#include_directories(/Users/michele/adolc_base)

# MY OWN LIBRARIES DEVELOPED ELSEWHERE
# MARKOV CHAINS CLASSES
#include_directories(/Users/michele/git/own/numerical/general/cpp/markovchains)

# Set compiler to GCC 7.2 (Installed via Mac Ports)
#set(CMAKE_C_COMPILER "gcc-mp-7")
#set(CMAKE_CXX_COMPILER "g++-mp-7")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

# Set options for clang++ compiler
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARMA_DONT_USE_WRAPPER -framework Accelerate -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DNDEBUG")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")


#add_executable(many_classes_cpp main_full.cpp src/markovchains/MarkovChain.h)

#add_executable(model_trial old/main_full_trial.cpp src/markovchains/MarkovChain.h)
#add_executable(model main_full.cpp src/markovchains/MarkovChain.h)

add_executable(main_full main_full.cpp
#                    src/Model.cpp
#                    src/Model.h
                    src/StructuralParameters.h
                    src/TechnicalParameters.h
                    src/markovchains/MarkovChain.h
                    src/markovchains/MarkovChainSimulations.h)

#add_executable(model main.cpp src/StructuralParameters.h src/TechnicalParameters.h)

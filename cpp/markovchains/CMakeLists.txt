cmake_minimum_required(VERSION 3.9)
project(markovchains)

# Set C++ standard (default is 11)
set(CMAKE_CXX_STANDARD 11)

# Include additional header folders
# MACPORTS
#include_directories(/opt/local/include)

# HOMEBREW
#include_directories(/usr/local/Cellar/)

# Set compiler to GCC 7.2 (Installed via Mac Ports)
#set(CMAKE_C_COMPILER "gcc-mp-7")
#set(CMAKE_CXX_COMPILER "g++-mp-7")
#set(CMAKE_C_COMPILER "gcc")
#set(CMAKE_CXX_COMPILER "g++")


if(APPLE)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "c++")
elseif(UNIX AND NOT APPLE)
    set(CMAKE_C_COMPILER "cc")
    set(CMAKE_CXX_COMPILER "c++")
endif()

# Set options for clang++ compiler
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0") # USE FOR DEBUGGING PURPOSES
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp") # USE FOR ACTUAL EXECUTION AND TIMING
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") # USE FOR ACTUAL EXECUTION AND TIMING


# Files to be run
#add_executable(markovchains markovchains.cpp DenseCTMC.cpp DenseCTMC.h)
#add_executable(markovchains SparseCTMC.cpp SparseCTMC.h SimCTMC.h markovchains.cpp)
#add_executable(markovchains markovchains.cpp DenseDTMC.cpp DenseDTMC.h)



add_executable(markovchains MarkovChain.h MarkovChainSimulations.h main.cpp)
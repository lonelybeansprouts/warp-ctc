IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(ctc_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)


find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    set(WITH_OMP ON)
endif()
if(NOT WITH_OMP)
    add_definitions(-DCTC_DISABLE_OMP)
endif()
if (WITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++14")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
ENDIF()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


MESSAGE(STATUS "Building shared library with no GPU support")

if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
ENDIF()

ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp)

add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp )
TARGET_LINK_LIBRARIES(test_cpu warpctc)
SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")

INSTALL(TARGETS warpctc
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")

INSTALL(FILES include/ctc.h DESTINATION "include")

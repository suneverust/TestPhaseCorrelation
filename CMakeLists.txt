cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(TestPhaseCorrelation)

find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIR})
link_directories(${FFTW3_LIBRARIES})

add_executable (TestPhaseCorrelation  TestPhaseCorrelation.cpp)
target_link_libraries(TestPhaseCorrelation ${FFTW3_LIBRARIES} -lm)

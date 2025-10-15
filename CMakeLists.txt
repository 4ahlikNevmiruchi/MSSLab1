cmake_minimum_required(VERSION 4.0)
project(MSSLab1)

set(CMAKE_CXX_STANDARD 20)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams filesystem system)

# Use PkgConfig to find FFTW3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

add_executable(MSSLab1 main.cpp)

target_include_directories(MSSLab1 PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${FFTW3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(MSSLab1 PRIVATE
        ${FFTW3_LIBRARIES}
        ${Boost_LIBRARIES}
)

# Ensure Eigen is available
target_link_libraries(MSSLab1 PRIVATE Eigen3::Eigen)

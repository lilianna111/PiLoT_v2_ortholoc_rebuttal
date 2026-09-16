cmake_minimum_required(VERSION 3.16)
project(OrthoLoCGravityPnP LANGUAGES CXX)

include(FetchContent)
find_package(Eigen3 REQUIRED)
set(PYTHON_PACKAGE OFF CACHE BOOL "" FORCE)
set(WITH_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(poselib
    URL https://codeload.github.com/PoseLib/PoseLib/tar.gz/7e9f5f53372e43f89655040d4dfc4a00e5ace11c)
FetchContent_MakeAvailable(poselib)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
FetchContent_Declare(pybind11
    URL https://codeload.github.com/pybind/pybind11/tar.gz/a2e59f0e7065404b44dfe92a28aca47ba1378dc4)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_gravity_pnp gravity_pnp.cc)
target_compile_features(_gravity_pnp PRIVATE cxx_std_17)
target_include_directories(_gravity_pnp PRIVATE ${poselib_SOURCE_DIR}/pybind)
target_link_libraries(_gravity_pnp PRIVATE PoseLib Eigen3::Eigen)
set_target_properties(_gravity_pnp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${GRAVITY_OUTPUT_DIR}")

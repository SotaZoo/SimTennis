cmake_minimum_required(VERSION 3.24)
project(SimTennis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(DEPS "${CMAKE_CURRENT_SOURCE_DIR}/.scratch/deps" CACHE PATH "Local pinned dependencies")
option(TENNIS_BUILD_GODOT "Build the Godot presentation bridge" ON)
option(TENNIS_BUILD_PYTHON "Build the headless Python connection" OFF)
if(TENNIS_BUILD_GODOT)
    set(GODOTCPP_BUILD_PROFILE "${CMAKE_CURRENT_SOURCE_DIR}/tools/godot_profile.json" CACHE FILEPATH "" FORCE)
    set(GODOTCPP_DISABLE_EXCEPTIONS OFF CACHE BOOL "" FORCE)
    set(GODOTCPP_SYSTEM_HEADERS ON CACHE BOOL "" FORCE)
    add_subdirectory("${DEPS}/godot-cpp" "godot-cpp" EXCLUDE_FROM_ALL)
endif()

add_library(mujoco SHARED IMPORTED)
if(APPLE)
    set(MUJOCO_LIBRARY "libmujoco.3.12.0.dylib")
    set(TENNIS_RPATH "@loader_path")
else()
    set(MUJOCO_LIBRARY "libmujoco.so.3.12.0")
    set(TENNIS_RPATH "$ORIGIN")
endif()
set_target_properties(mujoco PROPERTIES
    IMPORTED_LOCATION "${DEPS}/mujoco/lib/${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${DEPS}/mujoco/include")

add_library(tennis_core STATIC native/simulation.cpp)
target_include_directories(tennis_core PUBLIC native)
target_link_libraries(tennis_core PUBLIC mujoco)
target_compile_options(tennis_core PRIVATE -Wall -Wextra -Wpedantic)

if(TENNIS_BUILD_GODOT)
    add_library(tennis SHARED native/bridge.cpp)
    target_link_libraries(tennis PRIVATE tennis_core godot-cpp)
    set_target_properties(tennis PROPERTIES
        OUTPUT_NAME "tennis.${GODOTCPP_TARGET}"
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TENNIS_RPATH}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/godot/bin")
    add_custom_command(TARGET tennis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:mujoco>" "${CMAKE_CURRENT_SOURCE_DIR}/godot/bin/")
    if(APPLE)
        add_custom_command(TARGET tennis POST_BUILD
            COMMAND codesign --force --sign - "$<TARGET_FILE:tennis>")
    endif()
endif()

if(TENNIS_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)
    pybind11_add_module(_tennis_native native/python.cpp)
    target_link_libraries(_tennis_native PRIVATE tennis_core)
    set(NATIVE_SOURCE_HASHES "")
    foreach(SOURCE native/simulation.cpp native/simulation.hpp native/python.cpp)
        file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}" SOURCE_HASH)
        string(APPEND NATIVE_SOURCE_HASHES "${SOURCE_HASH}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SOURCE}")
    endforeach()
    string(SHA256 NATIVE_BUILD_ID "${NATIVE_SOURCE_HASHES}")
    target_compile_definitions(_tennis_native PRIVATE TENNIS_NATIVE_BUILD_ID="${NATIVE_BUILD_ID}")
    set_target_properties(_tennis_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
        BUILD_RPATH "${DEPS}/mujoco/lib")
endif()

add_executable(core_tests tests/core_tests.cpp tests/physics_references.cpp)
target_link_libraries(core_tests PRIVATE tennis_core)
set_target_properties(core_tests PROPERTIES BUILD_RPATH "${DEPS}/mujoco/lib")
enable_testing()
add_test(NAME core_behavior COMMAND core_tests "${CMAKE_CURRENT_SOURCE_DIR}/models")

add_executable(shared_world_tests tests/shared_world_tests.cpp)
target_link_libraries(shared_world_tests PRIVATE tennis_core)
set_target_properties(shared_world_tests PROPERTIES BUILD_RPATH "${DEPS}/mujoco/lib")
add_test(NAME shared_world_behavior COMMAND shared_world_tests "${CMAKE_CURRENT_SOURCE_DIR}/models")

add_executable(car_model_tests tests/car_model_tests.cpp)
target_link_libraries(car_model_tests PRIVATE tennis_core)
set_target_properties(car_model_tests PROPERTIES BUILD_RPATH "${DEPS}/mujoco/lib")
add_test(NAME car_model_behavior COMMAND car_model_tests "${CMAKE_CURRENT_SOURCE_DIR}/models/car_match.xml")

add_executable(return_control_tests tests/return_control_tests.cpp)
target_link_libraries(return_control_tests PRIVATE tennis_core)
set_target_properties(return_control_tests PROPERTIES BUILD_RPATH "${DEPS}/mujoco/lib")
add_test(NAME return_control_behavior COMMAND return_control_tests "${CMAKE_CURRENT_SOURCE_DIR}/models/car_match.xml")

add_executable(native_trace native/trace.cpp)
target_link_libraries(native_trace PRIVATE tennis_core)
set_target_properties(native_trace PROPERTIES BUILD_RPATH "${DEPS}/mujoco/lib")

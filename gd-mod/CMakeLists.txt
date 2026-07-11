cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gdrl_bridge VERSION 1.0.0)

add_library(${PROJECT_NAME} SHARED
    src/main.cpp
    src/socket_server.cpp
)

# Pull in the Geode SDK. $ENV{GEODE_SDK} must point at an installed SDK
# (set by `geode sdk install`).
if (NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "GEODE_SDK not set — run `geode sdk install` first")
endif()

add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode)

setup_geode_mod(${PROJECT_NAME})

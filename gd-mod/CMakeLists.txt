cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GD 2.208 ships a universal binary and runs its arm64 slice natively on Apple
# Silicon, so the mod must be arm64 there (Geode's legacy default is x86_64,
# which dlopen rejects against the arm64 game). Default to the host arch; pass
# -DCMAKE_OSX_ARCHITECTURES=x86_64 to build for an Intel Mac / Rosetta GD.
if (APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "${CMAKE_HOST_SYSTEM_PROCESSOR}" CACHE STRING "" FORCE)
endif()

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

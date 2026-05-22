cmake_minimum_required(VERSION 3.16)
project(LoRaWANSimGUI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt setup
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Core Widgets Concurrent QUIET)
if(NOT Qt6_FOUND)
    find_package(Qt5 5.15 REQUIRED COMPONENTS Core Widgets Concurrent)
    set(QT_VERSION_MAJOR 5)
else()
    set(QT_VERSION_MAJOR 6)
endif()

# NOTE: ns-3 integration is temporarily disabled due to header conflicts
# between ns-3's custom <cstring> and Qt/STL headers.
# To enable ns-3 integration, see NS3_INTEGRATION.md

option(ENABLE_NS3 "Enable ns-3 integration (requires header compatibility fixes)" OFF)

if(ENABLE_NS3)
    message(STATUS "ns-3 integration enabled")
    # ns-3 setup
    set(NS3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ns-3-dev" CACHE PATH "Path to ns-3 root directory")
    
    # Find ns-3 build directory
    if(EXISTS "${NS3_ROOT}/build")
        set(NS3_BUILD_DIR "${NS3_ROOT}/build")
    elseif(EXISTS "${NS3_ROOT}/cmake-cache")
        set(NS3_BUILD_DIR "${NS3_ROOT}/cmake-cache")
    else()
        message(FATAL_ERROR "Cannot find ns-3 build directory. Please build ns-3 first.")
    endif()
    
    # Find ns-3 libraries
    find_library(NS3_LORAWAN_SIM_LIB
        NAMES ns3.45-lorawan-sim-default libns3.45-lorawan-sim-default
        PATHS ${NS3_BUILD_DIR}/lib ${NS3_BUILD_DIR}
        NO_DEFAULT_PATH
    )
    
    if(NOT NS3_LORAWAN_SIM_LIB)
        message(FATAL_ERROR "Cannot find ns3 lorawan-sim library. Please build the lorawan-sim module first.")
    endif()
    
    message(STATUS "Found ns-3 lorawan-sim library: ${NS3_LORAWAN_SIM_LIB}")
else()
    message(STATUS "ns-3 integration disabled - building GUI only")
endif()

# Source files
set(SOURCES
    main.cpp
    ui/mainwindow.cpp
    ui/mapscene.cpp
)

set(HEADERS
    ui/mainwindow.h
    ui/mapscene.h
)

# Create executable
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Concurrent
)

# Optionally add ns-3 integration
if(ENABLE_NS3)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${NS3_ROOT}/build/include
        ${NS3_ROOT}/src/lorawan-sim/model
        ${NS3_ROOT}/src/lorawan/model
    )
    
    target_link_libraries(${PROJECT_NAME}
        ${NS3_LORAWAN_SIM_LIB}
        ${NS3_LORAWAN_LIB}
        ${NS3_CORE_LIB}
        ${NS3_NETWORK_LIB}
        ${NS3_MOBILITY_LIB}
    )
    
    target_compile_definitions(${PROJECT_NAME} PRIVATE NS3_ENABLED)
endif()

# Set output directory and name
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME lorawan-gui
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "LoRaWAN Simulation GUI Configuration")
message(STATUS "========================================")
message(STATUS "Qt Version: ${QT_VERSION_MAJOR}")
if(ENABLE_NS3)
    message(STATUS "ns-3 Integration: ENABLED")
    message(STATUS "ns-3 Root: ${NS3_ROOT}")
    message(STATUS "ns-3 Build: ${NS3_BUILD_DIR}")
else()
    message(STATUS "ns-3 Integration: DISABLED (GUI only)")
    message(STATUS "  To enable: cmake .. -DENABLE_NS3=ON")
endif()
message(STATUS "========================================")

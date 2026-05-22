#!/bin/bash

# Build script for LoRaWAN Qt GUI application
# Usage: ./build.sh [clean|rebuild]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
NS3_ROOT="../ns-3-dev"

# Parse arguments
CLEAN=false
if [ "$1" = "clean" ] || [ "$1" = "rebuild" ]; then
    CLEAN=true
fi

# Banner
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}LoRaWAN Qt GUI Build Script${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Check ns-3 exists
if [ ! -d "$NS3_ROOT" ]; then
    echo -e "${RED}Error: ns-3 directory not found at $NS3_ROOT${NC}"
    echo "Please set NS3_ROOT in this script or ensure ns-3-dev is in parent directory"
    exit 1
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Detect Qt version
echo -e "${YELLOW}Detecting Qt installation...${NC}"
if command -v qmake6 &> /dev/null; then
    echo -e "${GREEN}Found Qt6${NC}"
    QT_VERSION=6
elif command -v qmake &> /dev/null; then
    QT_PATH=$(qmake -query QT_INSTALL_PREFIX)
    echo -e "${GREEN}Found Qt at: $QT_PATH${NC}"
    QT_VERSION=5
else
    echo -e "${RED}Warning: qmake not found in PATH${NC}"
    echo "Qt may not be properly installed or not in PATH"
    echo "Continuing anyway - CMake will try to find Qt..."
fi

# Check if ns-3 lorawan-sim module is built
ENABLE_NS3=OFF
NS3_LIB_DIR="$(cd ../$NS3_ROOT/build/lib 2>/dev/null && pwd)"
if [ -d "$NS3_LIB_DIR" ]; then
    # Check for both .so (Linux) and .dylib (macOS)
    if ls "$NS3_LIB_DIR"/libns3.*lorawan-sim*.dylib 1> /dev/null 2>&1 || \
       ls "$NS3_LIB_DIR"/libns3.*lorawan-sim*.so 1> /dev/null 2>&1; then
        echo -e "${GREEN}✓ Found ns-3 lorawan-sim library${NC}"
        ENABLE_NS3=ON
    else
        echo -e "${YELLOW}⚠ ns-3 found but lorawan-sim module not built${NC}"
        echo "  Building GUI-only version (placeholder simulation)"
    fi
else
    echo -e "${YELLOW}⚠ ns-3 build directory not found${NC}"
    echo "  Building GUI-only version (placeholder simulation)"
fi

# Run CMake
echo ""
echo -e "${YELLOW}Running CMake configuration...${NC}"
cmake .. -DNS3_ROOT="$(cd ../$NS3_ROOT && pwd)" \
         -DENABLE_NS3=$ENABLE_NS3 \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build
echo ""
echo -e "${YELLOW}Building application...${NC}"
cmake --build . -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Success
echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo -e "Executable: ${GREEN}$BUILD_DIR/lorawan-gui${NC}"
echo ""
echo "To run:"
echo -e "  ${YELLOW}cd $BUILD_DIR && ./lorawan-gui${NC}"
echo ""
echo "Or from project root:"
echo -e "  ${YELLOW}./$BUILD_DIR/lorawan-gui${NC}"
echo ""

# Check if executable exists and is runnable
if [ -f "lorawan-gui" ]; then
    echo -e "${GREEN}✓ Executable created and ready to run${NC}"
    
    # Make sure it's executable
    chmod +x lorawan-gui
    
    # Show size
    SIZE=$(du -h lorawan-gui | cut -f1)
    echo -e "  Size: $SIZE"
else
    echo -e "${RED}✗ Warning: Executable not found${NC}"
    echo "  Build may have succeeded but executable is not in expected location"
fi

echo ""

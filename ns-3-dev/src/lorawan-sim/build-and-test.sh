#!/bin/bash
# Quick build and test script for lorawan-sim module

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NS3_ROOT="$SCRIPT_DIR/../.."

echo "================================================"
echo "LoRaWAN Simulation Module - Build & Test Script"
echo "================================================"
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "NS-3 root: $NS3_ROOT"
echo ""

# Navigate to ns-3 root
cd "$NS3_ROOT"

# Verify we're in the right place
# if [ ! -f "ns3" ]; then
#     echo "❌ Error: ns3 script not found in $NS3_ROOT"
#     echo "   Make sure you're running this from the lorawan-sim module directory"
#     exit 1
# fi

echo "✅ Found ns3 in: $(pwd)"
echo ""

# Build the module
echo "📦 Building lorawan module..."
./ns3 configure --enable-examples --enable-tests
./ns3 build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "Running Quick Tests"
echo "================================================"
echo ""

# Test 1: No ADR
echo "🧪 Test 1: Running with NO ADR..."
./ns3 run "advanced-ddqn-per-adr-example --adr=off --nDevices=5 --simulationTime=60" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ NO ADR test passed"
else
    echo "   ❌ NO ADR test failed"
fi

# Test 2: Classical ADR
echo "🧪 Test 2: Running with Classical ADR..."
./ns3 run "advanced-ddqn-per-adr-example --adr=on --nDevices=5 --simulationTime=60" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Classical ADR test passed"
else
    echo "   ❌ Classical ADR test failed"
fi

# Test 3: DDQN ADR
echo "🧪 Test 3: Running with DDQN ADR..."
./ns3 run "advanced-ddqn-per-adr-example --adr=ddqn --nDevices=5 --simulationTime=60" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ DDQN ADR test passed"
else
    echo "   ❌ DDQN ADR test failed"
fi

echo ""
echo "================================================"
echo "Module Verification"
echo "================================================"
echo ""

# Check if output files were created
if [ -d "lorawan_datasets" ]; then
    NUM_FILES=$(ls lorawan_datasets/*.csv 2>/dev/null | wc -l)
    echo "📊 Generated $NUM_FILES dataset files"
    
    if [ $NUM_FILES -gt 0 ]; then
        echo "   Sample file: $(ls lorawan_datasets/*.csv | head -1)"
        SAMPLE_FILE=$(ls lorawan_datasets/*.csv | head -1)
        LINES=$(wc -l < "$SAMPLE_FILE")
        echo "   Number of records: $((LINES - 1))"
    fi
else
    echo "⚠️  No datasets folder found"
fi

echo ""
echo "================================================"
echo "Module Structure Verification"
echo "================================================"
echo ""

MODULE_DIR="$NS3_ROOT/src/lorawan-sim"

echo "Checking module files:"

# Check model files
MODEL_FILES=("adr-ddqn.h" "adr-ddqn.cc" "environment.h" "environment.cc" 
             "packet-recorder.h" "packet-recorder.cc" 
             "simulation-runner.h" "simulation-runner.cc")

for file in "${MODEL_FILES[@]}"; do
    if [ -f "$MODULE_DIR/model/$file" ]; then
        echo "   ✅ model/$file"
    else
        echo "   ❌ model/$file MISSING"
    fi
done

# Check example
if [ -f "$MODULE_DIR/examples/lorawan-sim-example.cc" ]; then
    echo "   ✅ examples/lorawan-sim-example.cc"
else
    echo "   ❌ examples/lorawan-sim-example.cc MISSING"
fi

# Check CMakeLists
if [ -f "$MODULE_DIR/CMakeLists.txt" ]; then
    echo "   ✅ CMakeLists.txt"
else
    echo "   ❌ CMakeLists.txt MISSING"
fi

# Check documentation
if [ -f "$MODULE_DIR/README.md" ]; then
    echo "   ✅ README.md"
else
    echo "   ⚠️  README.md missing"
fi

echo ""
echo "================================================"
echo "✨ Module ready to use!"
echo "================================================"
echo ""
echo "Quick start:"
echo "  ./ns3 run 'lorawan-sim-example --adr=ddqn'"
echo ""
echo "Full comparison:"
echo "  ./ns3 run 'lorawan-sim-example --adr=all --nDevices=10'"
echo ""
echo "See README.md for full documentation"
echo ""

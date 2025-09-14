#!/bin/bash

# Validation script to check if the ns-3 LoRaWAN setup is working correctly

echo "🔍 Validating ns-3 LoRaWAN setup..."

# Check if we're in the right directory structure
if [ ! -d "ns-3-dev" ]; then
    echo "❌ ns-3-dev directory not found"
    echo "Please run this script from the directory containing ns-3-dev"
    exit 1
fi

cd ns-3-dev

# Check if LoRaWAN module exists
if [ ! -d "src/lorawan" ]; then
    echo "❌ LoRaWAN module not found in src/lorawan"
    exit 1
fi

echo "✅ Directory structure looks good"

# Check if ns3 build system is available
if [ ! -f "ns3" ]; then
    echo "❌ ns3 build script not found"
    exit 1
fi

echo "✅ ns3 build system found"

# Check if the module is configured
echo "🔍 Checking if LoRaWAN module is configured..."
if ./ns3 show modules | grep -q lorawan; then
    echo "✅ LoRaWAN module is configured"
else
    echo "⚠️  LoRaWAN module not configured. Running configuration..."
    ./ns3 configure --enable-tests --enable-examples --enable-modules lorawan
fi

# Check if our example is present
if [ -f "src/lorawan/examples/pre-tx-rssi-example.cc" ]; then
    echo "✅ pre-tx-rssi-example.cc found"
else
    echo "⚠️  pre-tx-rssi-example.cc not found in src/lorawan/examples/"
    echo "Copying from parent directory..."
    if [ -f "../pre-tx-rssi-example.cc" ]; then
        cp ../pre-tx-rssi-example.cc src/lorawan/examples/
        echo "✅ Example copied successfully"
    else
        echo "❌ Cannot find pre-tx-rssi-example.cc"
        exit 1
    fi
fi

# Try to build
echo "🔨 Testing build process..."
if ./ns3 build; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

# Check if we can list the examples
echo "🔍 Available LoRaWAN examples:"
ls src/lorawan/examples/*.cc | head -5

# Test a simple run
echo "🧪 Testing a quick simulation run..."
if ./ns3 run "pre-tx-rssi-example --nDevices=2 --simulationTime=10 --csvFile=validation_test.csv"; then
    echo "✅ Test simulation completed successfully"
    
    # Check if CSV was created
    if [ -f "validation_test.csv" ]; then
        echo "✅ CSV output file created"
        echo "📊 First few lines of output:"
        head -3 validation_test.csv
        rm validation_test.csv  # Clean up test file
    else
        echo "⚠️  CSV file not created, but simulation ran"
    fi
else
    echo "❌ Test simulation failed"
    exit 1
fi

echo ""
echo "🎉 Validation complete! Your ns-3 LoRaWAN setup is working correctly."
echo ""
echo "Ready to run simulations:"
echo "  ./ns3 run pre-tx-rssi-example"
echo "  ./ns3 run \"pre-tx-rssi-example --help\" (to see all options)"
echo ""

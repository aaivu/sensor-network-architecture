#!/bin/bash

# Script to set up and run the pre-TX RSSI example
# Run this after setup_ns3_lorawan.sh completes successfully

set -e

if [ ! -d "ns-3-dev" ]; then
    echo "❌ ns-3-dev directory not found. Please run setup_ns3_lorawan.sh first."
    exit 1
fi

echo "🔧 Setting up pre-TX RSSI example..."

# Copy the example to the correct location
echo "📁 Copying example file to ns-3 lorawan examples directory..."
cp pre-tx-rssi-example.cc ns-3-dev/src/lorawan/examples/

cd ns-3-dev

# Build the project
echo "🔨 Building ns-3 with the new example..."
./ns3 clean
./ns3 configure --enable-tests --enable-examples --enable-modules lorawan
./ns3 build

echo "✅ Build completed successfully!"

# Run a test simulation
echo "🚀 Running test simulation..."
echo "Parameters: 5 devices, 1 gateway, 60 seconds, 20-second intervals"
./ns3 run "pre-tx-rssi-example --nDevices=5 --nGateways=1 --simulationTime=60 --appPeriod=20 --csvFile=test_dataset.csv --radius=500"

echo ""
echo "🎉 Test simulation completed!"
echo "📊 Check test_dataset.csv for results"
echo ""
echo "To run longer simulations:"
echo "./ns3 run \"pre-tx-rssi-example --nDevices=20 --simulationTime=300 --csvFile=full_dataset.csv\""
echo ""
echo "Available parameters:"
echo "  --nDevices: Number of end devices (default: 10)"
echo "  --nGateways: Number of gateways (default: 1)"
echo "  --simulationTime: Duration in seconds (default: 300)"
echo "  --appPeriod: Packet interval in seconds (default: 30)"
echo "  --csvFile: Output filename (default: lorawan_rssi_dataset.csv)"
echo "  --radius: Area radius in meters (default: 1000)"

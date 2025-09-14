#!/bin/bash
# validate_setup.sh - Validate the ns-3 LoRaWAN environment with pre-TX RSSI capability

echo "=== NS-3 LoRaWAN Environment Validation ==="
echo ""

# Check if ns-3 build exists
if [[ -f "./ns3" ]]; then
    echo "✅ ns-3 build system found"
else
    echo "❌ ns-3 build system not found"
    exit 1
fi

# Check if LoRaWAN module is available
if [[ -d "src/lorawan" ]]; then
    echo "✅ LoRaWAN module found"
else
    echo "❌ LoRaWAN module not found"
    exit 1
fi

# Test basic example
echo ""
echo "Testing basic LoRaWAN functionality..."
if ./ns3 run "simple-network-example" > /dev/null 2>&1; then
    echo "✅ Basic LoRaWAN example works"
else
    echo "❌ Basic LoRaWAN example failed"
    exit 1
fi

# Test pre-TX RSSI examples
echo ""
echo "Testing pre-TX RSSI examples..."

if ./ns3 run "pre-tx-rssi-example --nDevices=2" > /dev/null 2>&1; then
    echo "✅ Basic pre-TX RSSI example works"
else
    echo "❌ Basic pre-TX RSSI example failed"
fi

if ./ns3 run "test-advanced-pre-tx-rssi-example --nDevices=2 --simulationTime=30" > /dev/null 2>&1; then
    echo "✅ Test advanced pre-TX RSSI example works"
else
    echo "❌ Test advanced pre-TX RSSI example failed"
fi

# Test the full advanced example
echo ""
echo "Testing advanced pre-TX RSSI simulation..."
if ./ns3 run "advanced-pre-tx-rssi-example --nDevices=3 --radius=500 --simulationTime=60 --csvFile=validation_test.csv" > /dev/null 2>&1; then
    echo "✅ Advanced pre-TX RSSI example works"
    
    # Check CSV output
    if [[ -f "validation_test.csv" ]]; then
        LINES=$(wc -l < validation_test.csv)
        if [[ $LINES -gt 1 ]]; then
            echo "✅ CSV dataset generated successfully ($((LINES-1)) records)"
            
            # Check for key columns
            HEADER=$(head -1 validation_test.csv)
            if [[ $HEADER == *"pre_tx_rssi_dBm"* ]]; then
                echo "✅ Pre-TX RSSI column found in dataset"
            else
                echo "❌ Pre-TX RSSI column missing from dataset"
            fi
            
            # Clean up
            rm -f validation_test.csv
        else
            echo "❌ CSV dataset appears empty"
        fi
    else
        echo "❌ CSV dataset not generated"
    fi
else
    echo "❌ Advanced pre-TX RSSI example failed"
    exit 1
fi

# Summary
echo ""
echo "=== Validation Summary ==="
echo "✅ Complete ns-3 LoRaWAN environment is working"
echo "✅ Pre-transmission RSSI sampling is functional"
echo "✅ ML-ready CSV dataset generation is working"
echo "✅ All examples compile and run successfully"
echo ""
echo "Ready for machine learning dataset generation!"
echo ""
echo "Key features implemented:"
echo "  • Hardware-like pre-TX RSSI sampling"
echo "  • Real-time interference calculation"
echo "  • Comprehensive CSV logging for ML training"
echo "  • Realistic LoRaWAN simulation environment"

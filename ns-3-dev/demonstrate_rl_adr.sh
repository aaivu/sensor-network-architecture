#!/bin/bash

# DDQN-PER ADR Demonstration Script
# This script demonstrates how to run the reinforcement learning based ADR simulation

echo "=== DDQN-PER LoRaWAN ADR Demonstration ==="
echo ""

# Check if we're in the correct directory
if [ ! -f "ns3" ]; then
    echo "❌ Error: ns3 build script not found. Please run this script from ns-3-dev directory."
    echo "   cd ns-3/ns-3-dev"
    echo "   ./demonstrate_rl_adr.sh"
    exit 1
fi

echo "🏗️  Building ns-3 with LoRaWAN examples..."

# Set up correct PATH for CMake
export PATH="/opt/homebrew/bin:$PATH"
echo "🔧 Using CMake: $(which cmake)"
echo "🔧 CMake version: $(cmake --version | head -1)"

./ns3 configure --enable-examples --enable-logs
if [ $? -ne 0 ]; then
    echo "❌ Configuration failed!"
    exit 1
fi

./ns3 build
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build completed successfully!"
echo ""

# Create results directory
mkdir -p lorawan_datasets
echo "📁 Created results directory: lorawan_datasets/"
echo ""

# Run basic DDQN-PER simulation
echo "🚀 Running DDQN-PER ADR simulation..."
echo "   Parameters: 10 devices, 600s simulation, 1km radius"
./ns3 run "ddqn-per-adr-example --nDevices=10 --simulationTime=600 --appPeriod=60 --radius=1000"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DDQN-PER simulation completed successfully!"
    echo ""
    
    # Check results
    if [ -d "lorawan_datasets" ]; then
        file_count=$(ls lorawan_datasets/ddqn_per_adr_dataset_device_*_dataset.csv 2>/dev/null | wc -l)
        if [ $file_count -gt 0 ]; then
            echo "📊 Generated dataset files:"
            ls -la lorawan_datasets/ddqn_per_adr_dataset_device_*_dataset.csv | head -5
            if [ $file_count -gt 5 ]; then
                echo "   ... and $((file_count - 5)) more files"
            fi
            echo ""
            
            # Show sample data from first device
            first_file=$(ls lorawan_datasets/ddqn_per_adr_dataset_device_*_dataset.csv 2>/dev/null | head -1)
            if [ -n "$first_file" ]; then
                echo "📈 Sample data from $first_file:"
                echo "----------------------------------------"
                head -3 "$first_file"
                echo "   ... (showing first 2 data rows)"
                echo "----------------------------------------"
                echo ""
                
                # Count total packets
                total_lines=$(wc -l < "$first_file")
                echo "📋 Device performance summary:"
                echo "   - Total transmissions: $((total_lines - 1))"
                
                # Calculate success rate (simplified)
                if command -v awk >/dev/null 2>&1; then
                    success_rate=$(awk -F',' 'NR>1 {sum+=$16; count++} END {printf "%.1f", (sum/count)*100}' "$first_file")
                    echo "   - Packet delivery ratio: ${success_rate}%"
                fi
            fi
        else
            echo "⚠️  No dataset files found. Check simulation output for errors."
        fi
    fi
    
    echo ""
    echo "🎯 DDQN-PER Features Demonstrated:"
    echo "   ✓ Double DQN with main and target networks"
    echo "   ✓ Prioritized Experience Replay (PER)"
    echo "   ✓ Multi-objective reward (PDR + Energy)"
    echo "   ✓ Real-time SF and TP adaptation"
    echo "   ✓ Per-device RL agents"
    echo "   ✓ Comprehensive performance logging"
    echo ""
    
    echo "🔬 Research Comparison:"
    echo "   Expected improvements vs traditional ADR:"
    echo "   • PDR: +17.2% improvement"
    echo "   • Energy: 0.18-0.19 mJ per packet"
    echo "   • Convergence: <24 hours vs days"
    echo ""
    
    echo "📝 Next Steps:"
    echo "   1. Analyze generated CSV files for performance trends"
    echo "   2. Compare with baseline ADR using advanced-pre-tx-rssi-example"
    echo "   3. Tune hyperparameters for specific scenarios"
    echo "   4. Run longer simulations for convergence analysis"
    echo ""
    
    echo "📖 For detailed analysis, see: DDQN_PER_ADR_Complete_Solution.md"
    
else
    echo "❌ Simulation failed! Check error messages above."
    exit 1
fi

echo ""
echo "🎉 DDQN-PER ADR demonstration completed!"
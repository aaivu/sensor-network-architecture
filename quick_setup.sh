#!/bin/bash

# Quick ns-3 LoRaWAN Setup Script for macOS (skipping Homebrew update)
# This script assumes you have basic tools (git, cmake) already installed

set -e  # Exit on any error

echo "🚀 Quick ns-3 LoRaWAN setup (skipping Homebrew update)..."

# Check required tools
echo "📦 Checking required tools..."

if ! command -v git &> /dev/null; then
    echo "❌ git not found. Please install git first."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found. Please install cmake first."
    exit 1
fi

echo "✅ Required tools found"

# Clone ns-3-dev if not exists
echo "📥 Cloning ns-3-dev..."
if [ ! -d "ns-3-dev" ]; then
    git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3-dev
    echo "✅ ns-3-dev cloned successfully"
else
    echo "✅ ns-3-dev directory already exists"
fi

cd ns-3-dev

# Clone LoRaWAN module if not exists
echo "📥 Cloning LoRaWAN module..."
if [ ! -d "src/lorawan" ]; then
    git clone https://github.com/signetlabdei/lorawan src/lorawan
    echo "✅ LoRaWAN module cloned successfully"
else
    echo "✅ LoRaWAN module directory already exists"
fi

# Switch to compatible ns-3 version
echo "🔄 Switching to compatible ns-3 version..."
if [ -f "src/lorawan/NS3-VERSION" ]; then
    tag=$(< src/lorawan/NS3-VERSION)
    tag=${tag#release }
    echo "Using ns-3 version: $tag"
    git checkout "$tag" -b "lorawan-support-$tag" 2>/dev/null || git checkout "lorawan-support-$tag"
    echo "✅ Switched to compatible ns-3 version: $tag"
else
    echo "⚠️  NS3-VERSION file not found, using current version"
fi

# Configure and build (using Python instead of missing dependencies)
echo "🛠️  Configuring ns-3..."
./ns3 clean
./ns3 configure --enable-tests --enable-examples --enable-modules lorawan

# Build
echo "🔨 Building ns-3 with LoRaWAN module..."
./ns3 build

echo ""
echo "🎉 Quick ns-3 with LoRaWAN module setup complete!"
echo ""
echo "Current directory: $(pwd)"
echo "Next: Copy your example file with: cp ../pre-tx-rssi-example.cc src/lorawan/examples/"

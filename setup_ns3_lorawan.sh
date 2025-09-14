#!/bin/bash

# ns-3 LoRaWAN Setup Script for macOS
# Based on the signetlabdei/lorawan module installation guide

set -e  # Exit on any error

echo "🚀 Setting up ns-3 with LoRaWAN module on macOS..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS"
    exit 1
fi

# Install Xcode Command Line Tools if not present
echo "📦 Checking Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "Please complete the Xcode Command Line Tools installation and run this script again."
    exit 1
else
    echo "✅ Xcode Command Line Tools already installed"
fi

# Check if Homebrew is installed, install if not
echo "🍺 Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

# Update Homebrew and install dependencies
echo "📦 Installing dependencies..."
brew update
brew install git python3 cmake ninja ccache pkg-config

# Clone ns-3-dev
echo "📥 Cloning ns-3-dev..."
if [ ! -d "ns-3-dev" ]; then
    git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3-dev
    echo "✅ ns-3-dev cloned successfully"
else
    echo "✅ ns-3-dev directory already exists"
fi

cd ns-3-dev

# Clone LoRaWAN module
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

# Clean and configure
echo "🛠️  Configuring ns-3..."
./ns3 clean
./ns3 configure --enable-tests --enable-examples --enable-modules lorawan

# Build
echo "🔨 Building ns-3 with LoRaWAN module..."
./ns3 build

# Run tests
echo "🧪 Running tests..."
if ./test.py; then
    echo "✅ All tests passed!"
else
    echo "⚠️  Some tests failed, but this might be normal"
fi

echo ""
echo "🎉 ns-3 with LoRaWAN module setup complete!"
echo ""
echo "Next steps:"
echo "1. cd ns-3-dev"
echo "2. Run example: ./ns3 run simple-network-example"
echo "3. Or create your own example in src/lorawan/examples/"
echo ""
echo "Current directory: $(pwd)"

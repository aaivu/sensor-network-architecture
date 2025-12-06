# LoRaWAN Qt GUI - Setup and Running Guide

This guide provides step-by-step instructions for building and running the LoRaWAN Qt GUI application with ns-3 integration.

## Building Instructions

### Step 1: Build ns-3 with LoRaWAN Module

The ns-3 simulator with the custom lorawan-sim module must be built first as the GUI depends on it.

```bash
# Navigate to the ns-3-dev directory
cd ns-3-dev/src/lorawan-sim

# Clean any previous builds (optional, recommended for fresh build)
rm -rf build/src/lorawan-sim
rm -rf cmake-cache/src/lorawan-sim

# Build ns-3 and the lorawan-sim module
bash build-and-test.sh

```

### Step 2: Build the Qt GUI Application

After successfully building ns-3, build the GUI application.

```bash
# Navigate to the GUI directory
cd ../../../lorawan-qt-gui

# Delete existing build (optional)

# Use the provided build script (recommended)
bash build.sh

# run simulation
cd build
./lorawan-gui


```

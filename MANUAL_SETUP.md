# Manual Steps if Scripts Have Issues

If the automated scripts encounter issues, you can follow these manual steps to set up ns-3 with LoRaWAN:

## Prerequisites Check

```bash
# Check if you have required tools
git --version
cmake --version
python3 --version

# If missing, install manually:
# On macOS: xcode-select --install
# Then: brew install git cmake python3
```

## Manual Installation Steps

### 1. Clone ns-3-dev
```bash
cd /Users/shakir/Desktop/fyp/ns-3
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3-dev
cd ns-3-dev
```

### 2. Clone LoRaWAN module
```bash
git clone https://github.com/signetlabdei/lorawan src/lorawan
```

### 3. Switch to compatible version
```bash
# Check what version the module requires
cat src/lorawan/NS3-VERSION

# Switch to that version (example for release-3.41)
git checkout release-3.41 -b lorawan-support-3.41
```

### 4. Configure and build
```bash
./ns3 clean
./ns3 configure --enable-tests --enable-examples --enable-modules lorawan
./ns3 build
```

### 5. Test the installation
```bash
./test.py
./ns3 run simple-network-example
```

### 6. Add your custom example
```bash
# Copy the pre-TX RSSI example
cp ../pre-tx-rssi-example.cc src/lorawan/examples/

# Rebuild to include the new example
./ns3 build

# Run your example
./ns3 run pre-tx-rssi-example
```

## Common Issues and Solutions

### Build Errors
- **Missing dependencies**: Install via `brew install cmake ninja ccache pkg-config`
- **Version mismatch**: Ensure ns-3 version matches `src/lorawan/NS3-VERSION`
- **Clean build**: Run `./ns3 clean` and try again

### Runtime Errors
- **Module not found**: Check that `--enable-modules lorawan` was used during configure
- **Permission errors**: Check file permissions and disk space
- **Simulation crashes**: Start with smaller parameters (fewer devices, shorter time)

### Performance Optimization
- **Faster builds**: Use `./ns3 build --jobs=4` (adjust number based on CPU cores)
- **Debug vs Release**: For final runs, use `./ns3 configure --build-profile=optimized`

## Verification Commands

```bash
# Check if LoRaWAN module is available
./ns3 show modules | grep lorawan

# List available examples
ls src/lorawan/examples/

# Run with help to see parameters
./ns3 run "pre-tx-rssi-example --help"
```

## Alternative: Homebrew Installation

If you prefer the Homebrew approach (faster but may not work with custom modules):

```bash
brew install ns-3
# Note: This installs a pre-built version without the LoRaWAN module
```

For research work requiring custom modules, the source build is recommended.

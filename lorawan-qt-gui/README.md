# LoRaWAN Qt GUI Application

A desktop GUI application for running and visualizing ns-3 LoRaWAN network simulations with DDQN-PER ADR algorithm.

## ⚡ Quick Start

### With ns-3 Integration (Recommended)
```bash
cd lorawan-qt-gui
./build.sh        # Automatically detects ns-3
./build/lorawan-gui
```

### GUI Only (No ns-3 Required)
```bash
cd lorawan-qt-gui/build
cmake ..
cmake --build .
./lorawan-gui     # Uses placeholder simulation
```

**See [NS3_INTEGRATION.md](NS3_INTEGRATION.md) for detailed integration guide.**

## Features

### 📊 Interactive Parameter Control
- Number of devices (1-1000)
- Deployment radius (100m-10km)
- Simulation time, app period
- WiFi interferers
- ADR mode selection (Off/Standard/DDQN-PER)
- Custom CSV output filename

### 🗺️ 2D Network Map Editor
- Visual representation of network topology
- Draggable end device nodes
- Fixed gateway at center (red)
- Deployment radius visualization
- Grid overlay with coordinate axes
- Zoom and pan capabilities

### 🏢 Obstacle Management
- Add rectangular obstacles (buildings, walls)
- Customizable attenuation (dB)
- Drag to reposition obstacles
- Right-click context menu:
  - Change attenuation
  - Delete obstacle
- Clear all obstacles

### 🚀 Background Simulation Execution
- Non-blocking UI during simulation
- Progress indicator
- Stop simulation capability
- Status messages

### 📈 Results Visualization
- Real-time summary statistics
- Per-device metrics from CSV
- Spreading factor distribution
- Average PDR across network
- Export results functionality

## Building

### Prerequisites

1. **Qt5 or Qt6** with Widgets and Concurrent modules:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install qt5-default qtbase5-dev libqt5concurrent5
   
   # macOS
   brew install qt@5
   
   # Or Qt6
   brew install qt
   ```

2. **ns-3** with lorawan-sim module:
   ```bash
   cd ns-3-dev
   ./ns3 configure --enable-examples --enable-tests
   ./ns3 build
   ```

### Compilation

```bash
cd lorawan-qt-gui
mkdir build
cd build

# Configure
cmake .. -DNS3_ROOT=/path/to/ns-3-dev

# Build
cmake --build .

# Or with make
make -j$(nproc)
```

The executable will be `build/lorawan-gui`.

## Usage

### 1. Launch Application

```bash
./lorawan-gui
```

### 2. Configure Network

**Left Panel - Parameters:**
- Set number of devices
- Adjust deployment radius
- Configure simulation duration
- Select ADR mode

**Map View - Network Layout:**
- Drag blue nodes to reposition devices
- Gateway (red) is fixed at center
- Right-click on empty space → "Add Obstacle Here"
- Right-click on obstacle → Edit or delete

### 3. Run Simulation

1. Click **"Run Simulation"** button (green, bottom)
2. Progress bar shows simulation status
3. UI remains responsive (background execution)
4. Stop anytime with **"Stop"** button (red)

### 4. View Results

**Results Panel (right):**
- Device count and average PDR
- SF distribution histogram
- Output CSV file path

**Export/Visualize:**
- Click **"Export Results"** to save CSV elsewhere
- Click **"Visualize Data"** to launch Python plots

## Project Structure

```
lorawan-qt-gui/
├── CMakeLists.txt          # Build configuration
├── main.cpp                # Application entry point
├── ui/
│   ├── mapscene.h          # 2D map editor (header)
│   ├── mapscene.cpp        # Map implementation
│   ├── mainwindow.h        # Main window (header)
│   └── mainwindow.cpp      # Main window implementation
└── README.md               # This file
```

## Class Overview

### MapScene (QGraphicsScene)
- **Purpose**: 2D network topology editor
- **Features**:
  - Gateway and end device rendering
  - Drag-and-drop node positioning
  - Obstacle creation/editing
  - Grid background with axes
  - Deployment radius circle

### NodeItem (QGraphicsEllipseItem)
- **Purpose**: Represents network nodes (gateway/devices)
- **Properties**:
  - Node ID and type (gateway/device)
  - Position (x, y)
  - Draggable (devices only)
  - Visual label

### ObstacleItem (QGraphicsRectItem)
- **Purpose**: Represents physical obstacles (buildings)
- **Properties**:
  - Position and dimensions (x, y, width, height)
  - Attenuation in dB
  - Context menu for editing

### MainWindow (QMainWindow)
- **Purpose**: Main application interface
- **Components**:
  - Parameter panel (spinboxes, comboboxes)
  - Map view (QGraphicsView)
  - Results display (QTextEdit)
  - Control buttons (run/stop)
  - Progress bar and status

## Integration with ns-3

### SimulationRunner API

The GUI calls `ns3::SimulationRunner` from the lorawan-sim module:

```cpp
ns3::SimulationRunner runner;
runner.SetNumDevices(50);
runner.SetRadius(1000.0);
runner.SetSimulationTime(3600);
runner.SetAppPeriod(60.0);
runner.SetWiFiInterferers(5);
runner.SetADREnabled(true);
runner.SetUseDDQN(true);
runner.SetOutputFile("results.csv");
runner.Run();
```

### Threading Model

- **Main Thread**: Qt UI event loop
- **Simulation Thread**: `QtConcurrent::run()` executes `SimulationRunner::Run()`
- **Communication**: `QFutureWatcher` signals when simulation completes

## Configuration Files

### CMakeLists.txt Key Settings

```cmake
# Find Qt (tries Qt6 first, falls back to Qt5)
find_package(Qt6 COMPONENTS Core Widgets Concurrent QUIET)
if(NOT Qt6_FOUND)
    find_package(Qt5 REQUIRED COMPONENTS Core Widgets Concurrent)
endif()

# Find ns-3 libraries
find_library(NS3_LORAWAN_SIM ...)
find_library(NS3_LORAWAN ...)
find_library(NS3_CORE ...)

# Link everything
target_link_libraries(lorawan-gui
    Qt::Core Qt::Widgets Qt::Concurrent
    ${NS3_LORAWAN_SIM} ${NS3_LORAWAN} ${NS3_CORE} ...)
```

## Troubleshooting

### Build Errors

**Qt not found:**
```bash
# Set Qt path
export CMAKE_PREFIX_PATH=/path/to/qt5/lib/cmake
cmake ..
```

**ns-3 libraries not found:**
```bash
# Ensure ns-3 is built
cd ../ns-3-dev
./ns3 build

# Specify ns-3 root explicitly
cmake .. -DNS3_ROOT=/absolute/path/to/ns-3-dev
```

**Linking errors:**
```bash
# Check library exists
ls ../ns-3-dev/build/lib/liblorawan-sim.*

# Rebuild ns-3 if needed
cd ../ns-3-dev
./ns3 clean
./ns3 build
```

### Runtime Errors

**Simulation doesn't start:**
- Check CSV output path is writable
- Verify ns-3 libraries are in LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
- Run with `--verbose` flag if available

**UI freezes:**
- Ensure QtConcurrent is properly linked
- Check `QFutureWatcher` is connected to finished signal

**Cannot load results:**
- Verify CSV file was created
- Check file permissions
- Ensure simulation completed successfully

## Future Enhancements

- [ ] Save/load configuration files (JSON)
- [ ] Import node positions from CSV
- [ ] Real-time simulation progress updates
- [ ] Integrated Python visualization (matplotlib embed)
- [ ] Export network topology as image
- [ ] Multi-simulation comparison mode
- [ ] Live RSSI heatmap during simulation
- [ ] Packet animation playback

## License

This GUI application is part of the ns-3 LoRaWAN simulator project and inherits the same license.

## Contributing

1. Follow Qt coding conventions
2. Use `clang-format` for C++ code
3. Test on both Qt5 and Qt6
4. Ensure backward compatibility with ns-3 module
5. Document new features in README

## Contact

For issues or questions about the GUI:
- GitHub Issues: [sensor-network-architecture](https://github.com/yourusername/sensor-network-architecture)
- ns-3 LoRaWAN Module: See main project README

---

**Note**: This GUI interfaces with the refactored lorawan-sim module. Do not modify the module code to accommodate the GUI - use the existing `SimulationRunner` API.

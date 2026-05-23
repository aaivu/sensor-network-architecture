# gui — Qt Desktop Visualiser

A lightweight Qt 5/6 desktop application that visualises LoRaWAN simulation results: device map, per-device SF / TX-power assignments, PDR over time, and energy consumption.

> **ns-3 live integration is currently disabled** due to header conflicts between ns-3's custom `<cstring>` and Qt/STL. The GUI operates in **post-processing mode**: you run the simulation first (which writes a CSV), then load the CSV into the GUI for visualisation. See [Enabling ns-3 integration](#enabling-ns-3-integration) if you need live coupling.

---

## Requirements

| Tool | Min version | macOS | Ubuntu / Debian |
|------|------------|-------|-----------------|
| Qt | **5.15** or **6.x** | `brew install qt` | `apt install qt6-base-dev` |
| CMake | 3.16 | `brew install cmake` | `apt install cmake` |
| GCC / Clang | ≥ 11 / ≥ 13 | via Xcode CLT | `apt install g++-12` |

---

## Build

```bash
cd gui

# Standard build (auto-detects Qt5 or Qt6)
./build.sh

# Clean rebuild
./build.sh rebuild
```

`build.sh` runs:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

The resulting binary is `build/lorawan-gui`.

### macOS — Qt path

If CMake cannot find Qt, pass the prefix explicitly:

```bash
# Qt6 installed via Homebrew
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(brew --prefix qt)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Ubuntu / Debian — Qt6

```bash
apt install qt6-base-dev libgl-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

---

## Running

```bash
# After building
./build/lorawan-gui
```

The application opens a window with:

| Panel | Description |
|-------|-------------|
| **Map view** | 2-D scatter of device positions; colour = current SF; size ∝ TX power |
| **Device table** | Per-device SF, TX power, PDR, energy, last RSSI |
| **Time chart** | PDR over simulation time for each algorithm |
| **Energy chart** | Energy-per-packet box plots per algorithm |

### Loading results

1. Run the simulation: `./run.sh --adr=all --nDevices=20 --simTime=3600`
2. Open the GUI: `./build/lorawan-gui`
3. **File → Open CSV** and select the output file (e.g. `lorawan_rl_adr_results.csv`)

---

## Source layout

```
gui/
├── CMakeLists.txt          ← CMake project (Qt5/Qt6 auto-detect, ENABLE_NS3 option)
├── build.sh                ← convenience build script
├── main.cpp                ← QApplication entry point
└── ui/
    ├── mainwindow.h/.cpp   ← main window: toolbar, splitter, chart/table layout
    └── mapscene.h/.cpp     ← QGraphicsScene subclass: device node rendering
```

---

## Enabling ns-3 integration

The `CMakeLists.txt` has an `ENABLE_NS3` CMake option (default `OFF`) that wires in ns-3 headers and the lorawan shared library. It is currently disabled because ns-3's vendored `<cstring>` conflicts with Qt's STL usage.

To experiment with it:

```bash
cmake -S . -B build \
  -DENABLE_NS3=ON \
  -DNS3_ROOT="$(pwd)/../ns3/ns-3-dev" \
  -DCMAKE_BUILD_TYPE=Debug
```

You will likely need to patch the conflicting ns-3 header or use `-isystem` to demote the ns-3 include path below system headers.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not find Qt5 / Qt6` | Set `CMAKE_PREFIX_PATH` to your Qt installation prefix |
| Black map view | Load a CSV first (File → Open CSV) |
| `dyld: Library not loaded` on macOS | Run `install_name_tool` or set `DYLD_LIBRARY_PATH=$(brew --prefix qt)/lib` |
| Build fails on Ubuntu with GL errors | `apt install libgl-dev libglu1-mesa-dev` |

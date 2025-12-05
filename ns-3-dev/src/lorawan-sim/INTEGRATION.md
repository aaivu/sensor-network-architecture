# Integration Guide: Using lorawan-sim Module Externally

This guide shows how to use the lorawan-sim module in external C++ projects like Qt applications, web servers, or CLI tools.

## Option 1: Link Against ns-3 Module (Recommended)

### Prerequisites
- ns-3 installed and built with lorawan-sim module
- CMake 3.10+
- C++17 compiler

### Project CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyLoRaWANApp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set ns-3 installation path
set(NS3_DIR "/path/to/ns-3-dev/build" CACHE PATH "ns-3 build directory")

# Include ns-3 headers
include_directories(
    ${NS3_DIR}/include
    ${NS3_DIR}/src/lorawan-sim/model
)

# Link directories
link_directories(${NS3_DIR}/lib)

# Your application
add_executable(my_lorawan_app
    main.cpp
    # ... your other sources
)

# Link ns-3 libraries
target_link_libraries(my_lorawan_app
    ns3-dev-lorawan-sim-default
    ns3-dev-lorawan-default
    ns3-dev-core-default
    ns3-dev-network-default
    ns3-dev-mobility-default
    ns3-dev-propagation-default
    ns3-dev-point-to-point-default
    ns3-dev-applications-default
)
```

### Example main.cpp

```cpp
#include "simulation-runner.h"
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace ns3::lorawan;
    
    // Create simulation with DDQN ADR
    SimulationRunner sim(
        10,              // 10 devices
        300.0,           // 5 minutes
        30.0,            // 30s packet interval
        1000.0,          // 1km radius
        "output.csv",    
        ADRMethod::DDQN,
        12,              // 12 WiFi interferers
        true             // Environmental modeling
    );
    
    std::cout << "Starting simulation..." << std::endl;
    sim.Run();
    std::cout << "Simulation complete!" << std::endl;
    
    // Access results
    auto& recorder = sim.GetRecorder();
    const auto& packets = recorder.GetCompletedPackets();
    
    std::cout << "Total packets: " << packets.size() << std::endl;
    
    return 0;
}
```

### Build

```bash
mkdir build && cd build
cmake -DNS3_DIR=/path/to/ns-3-dev/build ..
make
./my_lorawan_app
```

## Option 2: Qt Application Integration

### Qt Project (.pro file)

```qmake
QT += core gui widgets
TARGET = LoRaWANSimGUI
TEMPLATE = app

CONFIG += c++17

# ns-3 paths
NS3_BUILD = /path/to/ns-3-dev/build

INCLUDEPATH += $$NS3_BUILD/include \
               $$NS3_BUILD/src/lorawan-sim/model

LIBS += -L$$NS3_BUILD/lib \
        -lns3-dev-lorawan-sim-default \
        -lns3-dev-lorawan-default \
        -lns3-dev-core-default \
        -lns3-dev-network-default \
        -lns3-dev-mobility-default \
        -lns3-dev-propagation-default \
        -lns3-dev-point-to-point-default \
        -lns3-dev-applications-default

SOURCES += main.cpp \
           mainwindow.cpp \
           simulationthread.cpp

HEADERS += mainwindow.h \
           simulationthread.h

FORMS += mainwindow.ui
```

### simulationthread.h

```cpp
#ifndef SIMULATIONTHREAD_H
#define SIMULATIONTHREAD_H

#include <QThread>
#include "simulation-runner.h"

class SimulationThread : public QThread {
    Q_OBJECT
    
public:
    SimulationThread(uint32_t nDevices, double simTime, 
                     ns3::lorawan::ADRMethod adr, QObject* parent = nullptr);
    
protected:
    void run() override;
    
signals:
    void progressUpdate(int percent);
    void simulationComplete(int totalPackets, int received);
    void errorOccurred(QString error);
    
private:
    uint32_t m_nDevices;
    double m_simTime;
    ns3::lorawan::ADRMethod m_adrMethod;
};

#endif
```

### simulationthread.cpp

```cpp
#include "simulationthread.h"
#include <QDebug>

SimulationThread::SimulationThread(uint32_t nDevices, double simTime, 
                                   ns3::lorawan::ADRMethod adr, QObject* parent)
    : QThread(parent), m_nDevices(nDevices), m_simTime(simTime), m_adrMethod(adr) {
}

void SimulationThread::run() {
    try {
        using namespace ns3::lorawan;
        
        emit progressUpdate(10);
        
        SimulationRunner sim(
            m_nDevices, m_simTime, 30.0, 1000.0,
            "qt_sim_output.csv", m_adrMethod, 12, true
        );
        
        emit progressUpdate(30);
        
        sim.Run();
        
        emit progressUpdate(90);
        
        // Get results
        auto& recorder = sim.GetRecorder();
        const auto& packets = recorder.GetCompletedPackets();
        
        int received = 0;
        for (const auto& pkt : packets) {
            if (pkt.packetReceived) received++;
        }
        
        emit progressUpdate(100);
        emit simulationComplete(packets.size(), received);
        
    } catch (const std::exception& e) {
        emit errorOccurred(QString::fromStdString(e.what()));
    }
}
```

### mainwindow.cpp (excerpt)

```cpp
void MainWindow::onRunButtonClicked() {
    uint32_t nDevices = ui->devicesSpinBox->value();
    double simTime = ui->simTimeSpinBox->value();
    
    ADRMethod adr = ADRMethod::OFF;
    if (ui->adrComboBox->currentText() == "Classical") {
        adr = ADRMethod::ON;
    } else if (ui->adrComboBox->currentText() == "DDQN") {
        adr = ADRMethod::DDQN;
    }
    
    SimulationThread* thread = new SimulationThread(nDevices, simTime, adr, this);
    
    connect(thread, &SimulationThread::progressUpdate, 
            ui->progressBar, &QProgressBar::setValue);
    connect(thread, &SimulationThread::simulationComplete,
            this, &MainWindow::onSimulationComplete);
    connect(thread, &SimulationThread::errorOccurred,
            this, &MainWindow::onSimulationError);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    
    thread->start();
}
```

## Option 3: Python Bindings (Future Work)

```python
import ns3lorawan

# Create simulation
sim = ns3lorawan.SimulationRunner(
    n_devices=10,
    simulation_time=300.0,
    app_period=30.0,
    radius=1000.0,
    csv_file="python_sim.csv",
    adr_method=ns3lorawan.ADRMethod.DDQN,
    wifi_interferers=12,
    environmental_modeling=True
)

# Run
sim.run()

# Access results
packets = sim.get_packets()
print(f"Total packets: {len(packets)}")
print(f"PDR: {sim.get_pdr():.2%}")
```

## Option 4: Web API Wrapper

### REST API Server (using cpp-httplib)

```cpp
#include "httplib.h"
#include "simulation-runner.h"
#include "json.hpp"

using json = nlohmann::json;

int main() {
    httplib::Server server;
    
    server.Post("/api/simulate", [](const httplib::Request& req, httplib::Response& res) {
        auto params = json::parse(req.body);
        
        ns3::lorawan::SimulationRunner sim(
            params["nDevices"],
            params["simulationTime"],
            params["appPeriod"],
            params["radius"],
            "api_output.csv",
            params["adr"] == "ddqn" ? ns3::lorawan::ADRMethod::DDQN : 
                                      ns3::lorawan::ADRMethod::OFF,
            params.value("wifiInterferers", 12),
            params.value("environmental", true)
        );
        
        sim.Run();
        
        auto& recorder = sim.GetRecorder();
        const auto& packets = recorder.GetCompletedPackets();
        
        json result;
        result["totalPackets"] = packets.size();
        result["success"] = true;
        
        res.set_content(result.dump(), "application/json");
    });
    
    server.listen("0.0.0.0", 8080);
    return 0;
}
```

## Build Tips

### Finding ns-3 Libraries

```bash
# List available ns-3 libraries
ls /path/to/ns-3-dev/build/lib/libns3*.so

# Check library dependencies
ldd /path/to/ns-3-dev/build/lib/libns3-dev-lorawan-sim-default.so
```

### Setting LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=/path/to/ns-3-dev/build/lib:$LD_LIBRARY_PATH
```

### pkg-config Support (if installed)

```bash
pkg-config --cflags --libs ns3-lorawan-sim
```

## Common Issues

### Issue: "undefined reference to ns3::..."

**Solution**: Ensure all ns-3 libraries are linked in correct order. Core dependencies first:

```cmake
target_link_libraries(my_app
    ns3-dev-lorawan-sim-default
    ns3-dev-lorawan-default
    ns3-dev-applications-default
    ns3-dev-point-to-point-default
    ns3-dev-mobility-default
    ns3-dev-network-default
    ns3-dev-core-default  # Core last
)
```

### Issue: "cannot find -lns3-dev-lorawan-sim-default"

**Solution**: Rebuild ns-3 with the module:

```bash
cd ns-3-dev
./ns3 configure --enable-examples
./ns3 build
```

### Issue: Simulation hangs or crashes

**Solution**: Ensure proper initialization order:

1. Initialize logging before simulation
2. Don't mix Qt event loop with ns-3 Simulator
3. Run simulations in separate threads

## Performance Considerations

- **Threading**: Run simulations in background threads for GUI apps
- **Memory**: Each simulation allocates ~50-100 MB
- **Duration**: 300s sim-time = ~10-30s real-time (depends on complexity)
- **Datasets**: CSV files can be 10-50 MB for large simulations

## Example: Complete Qt Application

See `examples/qt-lorawan-sim/` directory for a complete Qt5/Qt6 application example.

## Support

For integration issues:
- Check ns-3 documentation: https://www.nsnam.org/docs/
- LoRaWAN module docs: https://www.nsnam.org/docs/models/html/lorawan.html
- Open an issue on GitHub

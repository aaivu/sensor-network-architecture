# Navigate to ns-3 directory
cd /Users/shakir/Desktop/fyp/ns-3/ns-3-dev

# 1. Configure (only needed once)
./ns3 configure --enable-examples

# 2. Build (when you make changes)
./ns3 build

# 3. Run different simulation scenarios:

# Basic simulation
./ns3 run "advanced-pre-tx-rssi-example --nDevices=10 --simulationTime=300"

# Advanced with environmental effects
./ns3 run "advanced-pre-tx-rssi-example --nDevices=20 --simulationTime=600 --radius=1500 --environmental=true --wifiInterferers=5"

# Large scale simulation
./ns3 run "advanced-pre-tx-rssi-example --nDevices=50 --simulationTime=1200 --radius=2000 --environmental=true --wifiInterferers=10"

# Both ADR and non-ADR comparison
./ns3 run "advanced-pre-tx-rssi-example --nDevices=30 --adr=both --environmental=true"

# 4. Run visualization
python3 python_visualization.py
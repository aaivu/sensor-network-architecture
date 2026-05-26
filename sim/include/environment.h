// -*- mode: c++; -*-
#pragma once
// Environment: SetupUrbanEnvironment, SetupWiFiInterferers, SetupLTEMobileInterferers, SamplePreTxRssi

void SetupUrbanEnvironment(double radius) {
    if (!enableEnvironmentalModeling) return;
    
    // Use fixed seed for consistent obstacle placement
    uint32_t obstaclesPerSide = static_cast<uint32_t>(radius / 80.0);
    double obstacleSpacing = (2.0 * radius) / obstaclesPerSide;
    
    Ptr<UniformRandomVariable> obstacleWidthRand = CreateObject<UniformRandomVariable>();
    obstacleWidthRand->SetAttribute("Min", DoubleValue(20.0));
    obstacleWidthRand->SetAttribute("Max", DoubleValue(120.0));
    obstacleWidthRand->SetStream(1001);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> obstacleHeightRand = CreateObject<UniformRandomVariable>();
    obstacleHeightRand->SetAttribute("Min", DoubleValue(8.0));
    obstacleHeightRand->SetAttribute("Max", DoubleValue(80.0));
    obstacleHeightRand->SetStream(1002);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> attenuationRand = CreateObject<UniformRandomVariable>();
    attenuationRand->SetAttribute("Min", DoubleValue(8.0));
    attenuationRand->SetAttribute("Max", DoubleValue(35.0));
    attenuationRand->SetStream(1003);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> placementProbability = CreateObject<UniformRandomVariable>();
    placementProbability->SetAttribute("Min", DoubleValue(0.0));
    placementProbability->SetAttribute("Max", DoubleValue(1.0));
    placementProbability->SetStream(1004);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> positionJitter = CreateObject<UniformRandomVariable>();
    positionJitter->SetAttribute("Min", DoubleValue(-15.0));
    positionJitter->SetAttribute("Max", DoubleValue(15.0));
    positionJitter->SetStream(1005);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < obstaclesPerSide; i++) {
        for (uint32_t j = 0; j < obstaclesPerSide; j++) {
            if (placementProbability->GetValue() < 0.25) continue;
            
            double centerX = -radius + i * obstacleSpacing + positionJitter->GetValue();
            double centerY = -radius + j * obstacleSpacing + positionJitter->GetValue();
            double distFromCenter = std::sqrt(centerX * centerX + centerY * centerY);
            
            if (distFromCenter < 150.0) continue;
            
            UrbanObstacle obstacle;
            obstacle.position = Vector(centerX, centerY, 0);
            obstacle.width = obstacleWidthRand->GetValue();
            obstacle.height = obstacleHeightRand->GetValue();
            obstacle.attenuationDb = attenuationRand->GetValue();
            
            urbanObstacles.push_back(obstacle);
        }
    }
    
    NS_LOG_INFO("Created " << urbanObstacles.size() << " virtual obstacles for urban modeling");
    std::cout << "Urban environment: Created " << urbanObstacles.size() << " virtual obstacles" << std::endl;
}

/**
 * Structure to hold RSSI sampling results
 */
struct RssiSample {
    double preTxRssiDbm;
    double ambientNoiseDbm;
};

/**
 * Set up simplified WiFi interferer modeling
 */
void SetupWiFiInterferers(double radius, uint32_t nInterferers) {
    if (!enableEnvironmentalModeling || nInterferers == 0) return;
    
    wifiInterferers.Create(nInterferers);
    
    MobilityHelper wifiMobility;
    Ptr<ListPositionAllocator> wifiAllocator = CreateObject<ListPositionAllocator>();
    
    Ptr<UniformRandomVariable> wifiXRand = CreateObject<UniformRandomVariable>();
    wifiXRand->SetAttribute("Min", DoubleValue(-radius * 0.8));
    wifiXRand->SetAttribute("Max", DoubleValue(radius * 0.8));
    wifiXRand->SetStream(2001);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> wifiYRand = CreateObject<UniformRandomVariable>();
    wifiYRand->SetAttribute("Min", DoubleValue(-radius * 0.8));
    wifiYRand->SetAttribute("Max", DoubleValue(radius * 0.8));
    wifiYRand->SetStream(2002);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < nInterferers; i++) {
        Vector wifiPos(wifiXRand->GetValue(), wifiYRand->GetValue(), 3.0);
        wifiAllocator->Add(wifiPos);
    }
    
    wifiMobility.SetPositionAllocator(wifiAllocator);
    wifiMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    wifiMobility.Install(wifiInterferers);
    
    NS_LOG_INFO("Created " << nInterferers << " WiFi interferers (simplified model)");
    std::cout << "Interference modeling: Created " << nInterferers << " WiFi interferers" << std::endl;
}

/**
 * Set up LTE mobile interferers with human-like mobility patterns
 * Simulates people walking with mobile phones causing RF interference
 */
void SetupLTEMobileInterferers(double radius, uint32_t nInterferers) {
    if (!enableEnvironmentalModeling || nInterferers == 0) return;
    
    lteInterferers.Create(nInterferers);
    
    MobilityHelper lteMobility;
    
    // Initial position allocator - spread LTE devices across the area
    Ptr<ListPositionAllocator> lteAllocator = CreateObject<ListPositionAllocator>();
    
    Ptr<UniformRandomVariable> lteXRand = CreateObject<UniformRandomVariable>();
    lteXRand->SetAttribute("Min", DoubleValue(-radius * 0.9));
    lteXRand->SetAttribute("Max", DoubleValue(radius * 0.9));
    lteXRand->SetStream(2101);  // Fixed stream for consistency
    
    Ptr<UniformRandomVariable> lteYRand = CreateObject<UniformRandomVariable>();
    lteYRand->SetAttribute("Min", DoubleValue(-radius * 0.9));
    lteYRand->SetAttribute("Max", DoubleValue(radius * 0.9));
    lteYRand->SetStream(2102);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < nInterferers; i++) {
        // LTE devices at human height (1.5m)
        Vector ltePos(lteXRand->GetValue(), lteYRand->GetValue(), 1.5);
        lteAllocator->Add(ltePos);
    }
    
    lteMobility.SetPositionAllocator(lteAllocator);
    
    // Use Random Walk 2D mobility model to simulate human walking patterns
    // People walk at speeds between 0.5 m/s (slow walk) to 2.0 m/s (fast walk)
    // Bounds format: "minX|maxX|minY|maxY"
    std::ostringstream boundsStr;
    boundsStr << "-" << radius << "|" << radius << "|-" << radius << "|" << radius;
    
    lteMobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                 "Bounds", StringValue(boundsStr.str()),
                                 "Speed", StringValue("ns3::UniformRandomVariable[Min=0.5|Max=2.0]"),
                                 "Distance", DoubleValue(50.0),  // Walk 50m before changing direction
                                 "Mode", StringValue("Distance"));  // Change direction after walking distance
    
    lteMobility.Install(lteInterferers);
    
    NS_LOG_INFO("Created " << nInterferers << " LTE mobile interferers with human mobility");
    std::cout << "Interference modeling: Created " << nInterferers << " LTE mobile devices (human walking patterns)" << std::endl;
    std::cout << "  - Walking speed: 0.5-2.0 m/s (1.8-7.2 km/h)" << std::endl;
    std::cout << "  - Direction change: every 30s or 50m" << std::endl;
    std::cout << "  - Device height: 1.5m (hand-held)" << std::endl;
}

/**
 * Initialize hardware variability for each device
 */
void InitializeHardwareVariability(uint32_t nDevices) {
    if (!enableEnvironmentalModeling) return;
    
    Ptr<NormalRandomVariable> antennaGainVar = CreateObject<NormalRandomVariable>();
    antennaGainVar->SetAttribute("Mean", DoubleValue(0.0));
    antennaGainVar->SetAttribute("Variance", DoubleValue(2.0));
    antennaGainVar->SetStream(3001);  // Fixed stream for consistency
    
    Ptr<NormalRandomVariable> noiseFigureVar = CreateObject<NormalRandomVariable>();
    noiseFigureVar->SetAttribute("Mean", DoubleValue(6.0));
    noiseFigureVar->SetAttribute("Variance", DoubleValue(1.5));
    noiseFigureVar->SetStream(3002);  // Fixed stream for consistency
    
    Ptr<NormalRandomVariable> hardwareVar = CreateObject<NormalRandomVariable>();
    hardwareVar->SetAttribute("Mean", DoubleValue(0.0));
    hardwareVar->SetAttribute("Variance", DoubleValue(1.0));
    hardwareVar->SetStream(3003);  // Fixed stream for consistency
    
    for (uint32_t i = 0; i < nDevices; i++) {
        nodeAntennaGains[i] = antennaGainVar->GetValue();
        nodeNoiseFigures[i] = noiseFigureVar->GetValue();
        nodeHardwareVariance[i] = hardwareVar->GetValue();
        
        Ptr<NormalRandomVariable> txPowerVar = CreateObject<NormalRandomVariable>();
        txPowerVar->SetAttribute("Mean", DoubleValue(0.0));
        txPowerVar->SetAttribute("Variance", DoubleValue(0.5));
        txPowerVar->SetStream(3100 + i);  // Fixed stream for consistency
        
        if (nodeTxPowers.find(i) != nodeTxPowers.end()) {
            nodeTxPowers[i] += txPowerVar->GetValue();
        }
    }
    
    NS_LOG_INFO("Initialized hardware variability for " << nDevices << " devices");
    std::cout << "Hardware modeling: Added manufacturing tolerances for " << nDevices << " devices" << std::endl;
}

/**
 * Core function: Simulate reading RSSI register at transmitter location
 */
RssiSample SamplePreTxRssi(uint32_t txNodeId, Time currentTime) {
    Vector txPos = nodePositions[txNodeId];
    
    Ptr<UniformRandomVariable> noiseVar = CreateObject<UniformRandomVariable>();
    
    if (enableEnvironmentalModeling) {
        double urbanNoiseIncrease = 5.0;
        double timeVaryingNoise = 2.0 * std::sin(currentTime.GetSeconds() / 60.0);
        
        noiseVar->SetAttribute("Min", DoubleValue(-115.0 + urbanNoiseIncrease));
        noiseVar->SetAttribute("Max", DoubleValue(-105.0 + urbanNoiseIncrease + timeVaryingNoise));
    } else {
        noiseVar->SetAttribute("Min", DoubleValue(-115.0));
        noiseVar->SetAttribute("Max", DoubleValue(-105.0));
    }
    
    double ambientNoise = noiseVar->GetValue();
    
    if (enableEnvironmentalModeling && nodeNoiseFigures.find(txNodeId) != nodeNoiseFigures.end()) {
        ambientNoise += (nodeNoiseFigures[txNodeId] - 6.0);
    }
    
    double totalPowerLinear = DbmToW(ambientNoise);
    
    for (const auto& pair : activeTransmitters) {
        uint32_t otherNodeId = pair.first;
        bool isTransmitting = pair.second;
        
        if (otherNodeId == txNodeId || !isTransmitting) continue;
        if (currentTime >= transmissionEndTimes[otherNodeId]) continue;
        
        Vector interfererPos = nodePositions[otherNodeId];
        double interfererTxPower = nodeTxPowers[otherNodeId];
        
        if (enableEnvironmentalModeling) {
            if (nodeAntennaGains.find(otherNodeId) != nodeAntennaGains.end()) {
                interfererTxPower += nodeAntennaGains[otherNodeId];
            }
            if (nodeAntennaGains.find(txNodeId) != nodeAntennaGains.end()) {
                interfererTxPower += nodeAntennaGains[txNodeId];
            }
        }
        
        Ptr<ConstantPositionMobilityModel> txMobility = CreateObject<ConstantPositionMobilityModel>();
        txMobility->SetPosition(txPos);
        
        Ptr<ConstantPositionMobilityModel> interfererMobility = CreateObject<ConstantPositionMobilityModel>();
        interfererMobility->SetPosition(interfererPos);
        
        double rxPowerDbm = globalChannel->GetRxPower(interfererTxPower, interfererMobility, txMobility);
        
        if (enableEnvironmentalModeling && nodeHardwareVariance.find(txNodeId) != nodeHardwareVariance.end()) {
            rxPowerDbm += nodeHardwareVariance[txNodeId];
        }
        
        totalPowerLinear += DbmToW(rxPowerDbm);
    }
    
    if (enableEnvironmentalModeling && wifiInterferers.GetN() > 0) {
        Ptr<UniformRandomVariable> wifiInterferenceVar = CreateObject<UniformRandomVariable>();
        wifiInterferenceVar->SetAttribute("Min", DoubleValue(-130.0));
        wifiInterferenceVar->SetAttribute("Max", DoubleValue(-110.0));
        
        double wifiTrafficLevel = 0.5 + 0.5 * std::sin(currentTime.GetSeconds() / 30.0);
        double wifiInterferenceDbm = wifiInterferenceVar->GetValue() + 10.0 * std::log10(wifiTrafficLevel);
        
        totalPowerLinear += DbmToW(wifiInterferenceDbm);
    }
    
    if (enableEnvironmentalModeling) {
        Ptr<UniformRandomVariable> adjacentChannelVar = CreateObject<UniformRandomVariable>();
        adjacentChannelVar->SetAttribute("Min", DoubleValue(-140.0));
        adjacentChannelVar->SetAttribute("Max", DoubleValue(-120.0));
        
        double adjacentInterferenceDbm = adjacentChannelVar->GetValue();
        totalPowerLinear += DbmToW(adjacentInterferenceDbm);
    }
    
    double preTxRssiDbm = WToDbm(totalPowerLinear);
    
    Ptr<NormalRandomVariable> rssiNoise = CreateObject<NormalRandomVariable>();
    rssiNoise->SetAttribute("Mean", DoubleValue(0.0));
    
    if (enableEnvironmentalModeling) {
        rssiNoise->SetAttribute("Variance", DoubleValue(1.5));
    } else {
        rssiNoise->SetAttribute("Variance", DoubleValue(1.0));
    }
    
    double noisyRssiDbm = preTxRssiDbm + rssiNoise->GetValue();
    
    RssiSample sample;
    sample.preTxRssiDbm = noisyRssiDbm;
    sample.ambientNoiseDbm = ambientNoise;
    return sample;
}

/**
 * Callback triggered when ADR changes TX power for a device - SAME AS WORKING VERSION
 */

#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QMenuBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QSplitter>
#include <QtConcurrent/QtConcurrent>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include <QWheelEvent>
#include <QEvent>
#include <QDir>
#include <QDateTime>
#include <QProcess>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDesktopServices>
#include <QUrl>
#include <QRegExp>
#include <QDialog>
#include <QTabWidget>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <algorithm>

#ifdef ENABLE_NS3
#include "ns3/core-module.h"
#include "ns3/simulation-runner.h"
#endif

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_simulationRunning(false)
    , m_lastSimulationSucceeded(false)
    , m_simulationProcess(nullptr)
    , m_stopRequested(false)
    , m_logFile(nullptr)
    , m_logStream(nullptr)
{
    // Create output directory structure
    createOutputDirectory();
    
    setupUi();
    createActions();
    createMenuBar();
    
    // Initialize simulation watcher
    m_simulationWatcher = new QFutureWatcher<void>(this);
    connect(m_simulationWatcher, &QFutureWatcher<void>::finished,
            this, &MainWindow::onSimulationFinished);
    
    setWindowTitle("LoRaWAN Network Simulator");
    resize(1400, 900);
    
    // Create initial nodes for visualization
    m_mapScene->createNodes(m_numDevicesSpin->value(), m_radiusSpin->value());
    
    logMessage("=== LoRaWAN Simulator Started ===", "INFO");
    logMessage(QString("Output directory: %1").arg(m_outputDirectory), "INFO");
    
    updateStatusBar("Ready");
}

MainWindow::~MainWindow() {
    if (m_simulationRunning) {
        m_simulationWatcher->cancel();
        m_simulationWatcher->waitForFinished();
    }
    
    logMessage("=== LoRaWAN Simulator Closed ===", "INFO");
    
    // Close log file
    if (m_logStream) {
        delete m_logStream;
        m_logStream = nullptr;
    }
    if (m_logFile) {
        m_logFile->close();
        delete m_logFile;
        m_logFile = nullptr;
    }
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
    if (obj == m_mapView->viewport() && event->type() == QEvent::Wheel) {
        QWheelEvent* wheelEvent = static_cast<QWheelEvent*>(event);
        
        // Zoom in/out with mouse wheel
        if (wheelEvent->angleDelta().y() > 0) {
            // Zoom in
            m_mapView->scale(1.15, 1.15);
        } else {
            // Zoom out
            m_mapView->scale(1.0 / 1.15, 1.0 / 1.15);
        }
        
        return true;  // Event handled
    }
    
    return QMainWindow::eventFilter(obj, event);
}

// ============================================================================
// UI Setup
// ============================================================================

void MainWindow::setupUi() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    // Main vertical layout
    QVBoxLayout* fullLayout = new QVBoxLayout(centralWidget);
    
    // Horizontal layout for main content
    QHBoxLayout* mainLayout = new QHBoxLayout();
    
    // Left panel - Parameters
    setupParameterPanel();
    mainLayout->addWidget(m_parameterPanel);
    
    // Center and right - Map and Results
    QSplitter* splitter = new QSplitter(Qt::Horizontal);
    
    // Map view
    QWidget* mapContainer = new QWidget();
    QVBoxLayout* mapLayout = new QVBoxLayout(mapContainer);
    setupMapView();
    mapLayout->addWidget(m_mapView);
    mapLayout->addWidget(m_mapInfoLabel);
    
    // Control buttons below map
    QHBoxLayout* mapButtonLayout = new QHBoxLayout();
    mapButtonLayout->addWidget(m_addObstacleButton);
    mapButtonLayout->addWidget(m_clearMapButton);
    mapButtonLayout->addWidget(m_randomizeButton);
    mapButtonLayout->addWidget(m_zoomInButton);
    mapButtonLayout->addWidget(m_zoomOutButton);
    mapButtonLayout->addWidget(m_zoomResetButton);
    mapLayout->addLayout(mapButtonLayout);
    
    splitter->addWidget(mapContainer);
    
    // Results panel
    setupResultsPanel();
    splitter->addWidget(m_resultsPanel);
    
    splitter->setStretchFactor(0, 3);  // Map gets more space
    splitter->setStretchFactor(1, 2);
    
    mainLayout->addWidget(splitter, 1);
    
    // Add main content to full layout
    fullLayout->addLayout(mainLayout, 1);
    
    // Bottom control panel
    setupControlPanel();
    fullLayout->addWidget(m_controlPanel);
    
    // Status bar
    statusBar()->showMessage("Ready");
}

void MainWindow::setupParameterPanel() {
    m_parameterPanel = new QWidget();
    m_parameterPanel->setMaximumWidth(300);
    QVBoxLayout* layout = new QVBoxLayout(m_parameterPanel);
    
    // Network Parameters Group
    QGroupBox* networkGroup = new QGroupBox("Network Parameters");
    QGridLayout* networkLayout = new QGridLayout();
    
    networkLayout->addWidget(new QLabel("Number of Devices:"), 0, 0);
    m_numDevicesSpin = new QSpinBox();
    m_numDevicesSpin->setRange(1, 1000);
    m_numDevicesSpin->setValue(10);  // Default 10 devices to match CLI
    connect(m_numDevicesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::onNumDevicesChanged);
    networkLayout->addWidget(m_numDevicesSpin, 0, 1);
    
    networkLayout->addWidget(new QLabel("Deployment Radius (m):"), 1, 0);
    m_radiusSpin = new QDoubleSpinBox();
    m_radiusSpin->setRange(100, 10000);
    m_radiusSpin->setValue(1000);
    m_radiusSpin->setSingleStep(100);
    connect(m_radiusSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onRadiusChanged);
    networkLayout->addWidget(m_radiusSpin, 1, 1);
    
    networkGroup->setLayout(networkLayout);
    layout->addWidget(networkGroup);
    
    // Simulation Parameters Group
    QGroupBox* simGroup = new QGroupBox("Simulation Parameters");
    QGridLayout* simLayout = new QGridLayout();
    
    simLayout->addWidget(new QLabel("Simulation Time (s):"), 0, 0);
    m_simTimeSpin = new QSpinBox();
    m_simTimeSpin->setRange(60, 86400);
    m_simTimeSpin->setValue(300);  // Default 300s to match CLI
    m_simTimeSpin->setSingleStep(60);
    simLayout->addWidget(m_simTimeSpin, 0, 1);
    
    simLayout->addWidget(new QLabel("App Period (s):"), 1, 0);
    m_appPeriodSpin = new QDoubleSpinBox();
    m_appPeriodSpin->setRange(1.0, 3600.0);
    m_appPeriodSpin->setValue(30.0);  // Default 30s to match CLI
    m_appPeriodSpin->setSingleStep(10.0);
    simLayout->addWidget(m_appPeriodSpin, 1, 1);
    
    simLayout->addWidget(new QLabel("WiFi Interferers:"), 2, 0);
    m_wifiInterferersSpin = new QSpinBox();
    m_wifiInterferersSpin->setRange(0, 100);
    m_wifiInterferersSpin->setValue(12);  // Default to 12 interferers (matches CLI)
    simLayout->addWidget(m_wifiInterferersSpin, 2, 1);

    simLayout->addWidget(new QLabel("LTE Interferers:"), 3, 0);
    m_lteInterferersSpin = new QSpinBox();
    m_lteInterferersSpin->setRange(0, 100);
    m_lteInterferersSpin->setValue(8);  // Default to 8 (matches CLI default)
    simLayout->addWidget(m_lteInterferersSpin, 3, 1);

    simLayout->addWidget(new QLabel("ADR Mode:"), 4, 0);
    m_adrModeCombo = new QComboBox();
    m_adrModeCombo->addItem("Off", "off");
    m_adrModeCombo->addItem("Standard (Classical)", "on");
    m_adrModeCombo->addItem("DDQN-PER", "ddqn");
    m_adrModeCombo->addItem("PPO", "ppo");
    m_adrModeCombo->addItem("MARL-DDQN", "marl");
    m_adrModeCombo->addItem("MAPPO-GAT", "mappo_gat");
    m_adrModeCombo->addItem("MARL-PPO", "marl_ppo");
    m_adrModeCombo->addItem("All (Comparison)", "all");
    m_adrModeCombo->setCurrentIndex(2);  // Default to DDQN
    simLayout->addWidget(m_adrModeCombo, 4, 1);

    simGroup->setLayout(simLayout);
    layout->addWidget(simGroup);
    
    layout->addStretch();
}

void MainWindow::setupMapView() {
    m_mapScene = new MapScene(this);
    m_mapView = new QGraphicsView(m_mapScene);
    m_mapView->setRenderHint(QPainter::Antialiasing);
    m_mapView->setDragMode(QGraphicsView::ScrollHandDrag);
    m_mapView->setMinimumSize(600, 600);
    
    // Enable mouse wheel zoom
    m_mapView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    m_mapView->setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    m_mapView->viewport()->installEventFilter(this);
    
    // Connect map signals
    connect(m_mapScene, &MapScene::nodePositionChanged,
            this, &MainWindow::onNodePositionChanged);
    connect(m_mapScene, &MapScene::obstacleAdded,
            this, &MainWindow::onObstacleAdded);
    connect(m_mapScene, &MapScene::obstacleRemoved,
            this, &MainWindow::onObstacleRemoved);
    
    // Map control buttons
    m_addObstacleButton = new QPushButton("Add Obstacle");
    connect(m_addObstacleButton, &QPushButton::clicked, [this]() {
        m_mapScene->addObstacle(0, 0, 100, 100, 20.0);
    });
    
    m_clearMapButton = new QPushButton("Clear Obstacles");
    connect(m_clearMapButton, &QPushButton::clicked,
            m_mapScene, &MapScene::clearObstacles);
    
    m_randomizeButton = new QPushButton("Randomize Positions");
    connect(m_randomizeButton, &QPushButton::clicked, [this]() {
        m_mapScene->createNodes(m_numDevicesSpin->value(), m_radiusSpin->value());
    });
    
    // Zoom buttons
    m_zoomInButton = new QPushButton("Zoom In (+)");
    connect(m_zoomInButton, &QPushButton::clicked, [this]() {
        m_mapView->scale(1.2, 1.2);
    });
    
    m_zoomOutButton = new QPushButton("Zoom Out (-)");
    connect(m_zoomOutButton, &QPushButton::clicked, [this]() {
        m_mapView->scale(1.0 / 1.2, 1.0 / 1.2);
    });
    
    m_zoomResetButton = new QPushButton("Reset View");
    connect(m_zoomResetButton, &QPushButton::clicked, [this]() {
        m_mapView->resetTransform();
        m_mapView->centerOn(0, 0);
    });
    
    m_mapInfoLabel = new QLabel();
    updateMapInfo();
}

void MainWindow::setupControlPanel() {
    m_controlPanel = new QWidget();
    QHBoxLayout* layout = new QHBoxLayout(m_controlPanel);
    
    m_runButton = new QPushButton("Run Simulation");
    m_runButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }");
    connect(m_runButton, &QPushButton::clicked, this, &MainWindow::onRunSimulation);
    layout->addWidget(m_runButton);
    
    m_stopButton = new QPushButton("Stop");
    m_stopButton->setEnabled(false);
    m_stopButton->setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }");
    connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::onStopSimulation);
    layout->addWidget(m_stopButton);
    
    m_progressBar = new QProgressBar();
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    layout->addWidget(m_progressBar, 1);
    
    m_statusLabel = new QLabel("Ready");
    layout->addWidget(m_statusLabel);
}

void MainWindow::setupResultsPanel() {
    m_resultsPanel = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_resultsPanel);
    
    QLabel* title = new QLabel("<b>Simulation Results</b>");
    title->setAlignment(Qt::AlignCenter);
    layout->addWidget(title);
    
    m_resultsText = new QTextEdit();
    m_resultsText->setReadOnly(true);
    m_resultsText->setPlaceholderText("Run simulation to see results...");
    layout->addWidget(m_resultsText, 1);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    
    m_exportResultsButton = new QPushButton("Export Results");
    m_exportResultsButton->setEnabled(false);
    connect(m_exportResultsButton, &QPushButton::clicked, [this]() {
        if (!m_currentOutputFile.isEmpty()) {
            QString dest = QFileDialog::getSaveFileName(this, "Export Results",
                                                        m_currentOutputFile,
                                                        "CSV Files (*.csv)");
            if (!dest.isEmpty()) {
                QFile::copy(m_currentOutputFile, dest);
                QMessageBox::information(this, "Export", "Results exported successfully!");
            }
        }
    });
    buttonLayout->addWidget(m_exportResultsButton);
    
    m_visualizeButton = new QPushButton("Visualize Data");
    m_visualizeButton->setEnabled(false);
    connect(m_visualizeButton, &QPushButton::clicked, [this]() {
        // Find the analysis plot image in the current simulation run's plots directory
        QString plotPath = m_outputDirectory + "/plots/lorawan_adr_performance_analysis.png";
        
        if (!QFile::exists(plotPath)) {
            QMessageBox::warning(this, "Visualize Data", 
                "Analysis plot not found. Please run the simulation first.\n\nExpected location:\n" + plotPath);
            return;
        }
        
        // Open the image with the default system viewer
        if (!QDesktopServices::openUrl(QUrl::fromLocalFile(plotPath))) {
            QMessageBox::warning(this, "Visualize Data", 
                "Failed to open the plot image.\n\nPath: " + plotPath);
        } else {
            logMessage("Opened analysis plot: " + plotPath, "INFO");
        }
    });
    buttonLayout->addWidget(m_visualizeButton);
    
    layout->addLayout(buttonLayout);
}

void MainWindow::createActions() {
    m_newAction = new QAction("&New Configuration", this);
    m_newAction->setShortcut(QKeySequence::New);
    
    m_openAction = new QAction("&Open Configuration", this);
    m_openAction->setShortcut(QKeySequence::Open);
    
    m_saveAction = new QAction("&Save Configuration", this);
    m_saveAction->setShortcut(QKeySequence::Save);
    
    m_exitAction = new QAction("E&xit", this);
    m_exitAction->setShortcut(QKeySequence::Quit);
    connect(m_exitAction, &QAction::triggered, this, &QMainWindow::close);
    
    m_aboutAction = new QAction("&About", this);
    connect(m_aboutAction, &QAction::triggered, [this]() {
        QMessageBox::about(this, "About LoRaWAN Simulator",
            "<h2>LoRaWAN Network Simulator</h2>"
            "<p>GUI interface for ns-3 LoRaWAN simulation with DDQN-PER ADR</p>"
            "<p>Version 1.0</p>");
    });
}

void MainWindow::createMenuBar() {
    QMenu* fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction(m_newAction);
    fileMenu->addAction(m_openAction);
    fileMenu->addAction(m_saveAction);
    fileMenu->addSeparator();
    fileMenu->addAction(m_exitAction);
    
    QMenu* helpMenu = menuBar()->addMenu("&Help");
    helpMenu->addAction(m_aboutAction);
}

// ============================================================================
// Simulation Logic
// ============================================================================

void MainWindow::cleanupOldDatasets() {
    // Find ns-3-dev directory
    QDir searchDir(QCoreApplication::applicationDirPath());
    QString ns3Dir;
    for (int i = 0; i < 5; ++i) {
        if (QDir(searchDir.absolutePath() + "/ns-3-dev").exists()) {
            ns3Dir = searchDir.absolutePath() + "/ns-3-dev";
            break;
        }
        searchDir.cdUp();
    }
    
    if (ns3Dir.isEmpty()) {
        return;
    }
    
    // Clean up lorawan_datasets directory (the simulation's own output folder)
    QDir datasetDir(ns3Dir + "/lorawan_datasets");
    if (datasetDir.exists()) {
        QStringList oldFiles = datasetDir.entryList(QStringList() << "*.csv", QDir::Files);
        int filesDeleted = 0;
        for (const QString& file : oldFiles) {
            if (QFile::remove(datasetDir.absoluteFilePath(file))) {
                filesDeleted++;
            }
        }
        if (filesDeleted > 0) {
            logMessage(QString("Cleaned up %1 old dataset file(s) from lorawan_datasets/").arg(filesDeleted), "INFO");
        }
    }
}

void MainWindow::onRunSimulation() {
    if (!validateParameters()) {
        return;
    }
    
    // Create a new timestamped output directory for this run
    createOutputDirectory();
    
    // Clean up old dataset files from ns-3 directory
    cleanupOldDatasets();
    
    logMessage("========================================", "INFO");
    logMessage("Starting new simulation run", "INFO");
    logMessage("========================================", "INFO");
    
    // Create nodes on map if not already done
    if (m_mapScene->getNodePositions().isEmpty()) {
        m_mapScene->createNodes(m_numDevicesSpin->value(), m_radiusSpin->value());
    }
    
    // Log simulation parameters
    logMessage("=== Simulation Parameters ===", "INFO");
    logMessage(QString("Number of Devices: %1").arg(m_numDevicesSpin->value()), "INFO");
    logMessage(QString("Deployment Radius: %1 m").arg(m_radiusSpin->value()), "INFO");
    logMessage(QString("Simulation Time: %1 s").arg(m_simTimeSpin->value()), "INFO");
    logMessage(QString("App Period: %1 s").arg(m_appPeriodSpin->value()), "INFO");
    logMessage(QString("WiFi Interferers: %1").arg(m_wifiInterferersSpin->value()), "INFO");
    logMessage(QString("LTE Interferers: %1").arg(m_lteInterferersSpin->value()), "INFO");
    logMessage(QString("ADR Mode: %1").arg(m_adrModeCombo->currentText()), "INFO");
    logMessage(QString("Output Directory: %1").arg(m_outputDirectory), "INFO");
    
    // Export configuration
    QString configDir = m_outputDirectory + "/config";
    QString positionsFile = configDir + "/node_positions.csv";
    QString obstaclesFile = configDir + "/obstacles.csv";
    
    exportNodePositions(positionsFile);
    exportObstacles(obstaclesFile);
    
    // Save configuration as JSON
    QJsonObject config;
    config["numDevices"] = m_numDevicesSpin->value();
    config["radius"] = m_radiusSpin->value();
    config["simulationTime"] = m_simTimeSpin->value();
    config["appPeriod"] = m_appPeriodSpin->value();
    config["wifiInterferers"] = m_wifiInterferersSpin->value();
    config["adrMode"] = m_adrModeCombo->currentText();
    config["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    QFile configFile(configDir + "/simulation_config.json");
    if (configFile.open(QIODevice::WriteOnly)) {
        QJsonDocument doc(config);
        configFile.write(doc.toJson(QJsonDocument::Indented));
        configFile.close();
        logMessage("Configuration saved to: " + configDir + "/simulation_config.json", "INFO");
    }
    
    m_simulationRunning = true;
    m_lastSimulationSucceeded = false;
    m_stopRequested.store(false);
    m_runButton->setEnabled(false);
    m_stopButton->setEnabled(true);
    m_progressBar->setValue(0);
    updateStatusBar("Running simulation...");
    logMessage("Simulation started...", "INFO");
    
    // Run in background thread
    QFuture<void> future = QtConcurrent::run([this]() {
        runSimulationThread();
    });
    
    m_simulationWatcher->setFuture(future);
}

void MainWindow::onStopSimulation() {
    logMessage("User requested simulation stop", "WARNING");
    m_stopRequested.store(true);
    // Kill the OS process immediately
    if (m_simulationProcess) {
        m_simulationProcess->kill();
    }
    m_simulationWatcher->cancel();
    m_simulationRunning = false;
    m_runButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    updateStatusBar("Simulation stopped by user");
    logMessage("Simulation stopped", "WARNING");
}

void MainWindow::organizeSimulationFiles(const QString& ns3Dir, const QString& csvDir, 
                                          const QString& adrMode, const QString& csvFilename) {
    QMetaObject::invokeMethod(this, [this, ns3Dir, csvDir, adrMode, csvFilename]() {
        logMessage(QString("Organizing files for ADR mode '%1'...").arg(adrMode), "INFO");

        // Determine the file prefix used by the simulation for this ADR mode
        QString expectedPrefix;
        if (adrMode == "ddqn")           expectedPrefix = "ddqn_adr_";
        else if (adrMode == "ppo")       expectedPrefix = "ppo_adr_";
        else if (adrMode == "marl")      expectedPrefix = "marl_adr_";
        else if (adrMode == "mappo_gat") expectedPrefix = "mappo_gat_adr_";
        else if (adrMode == "marl_ppo")  expectedPrefix = "marl_ppo_adr_";
        else if (adrMode == "on")        expectedPrefix = "adr_";
        else                             expectedPrefix = "no_adr_";

        // The simulation writes per-device CSVs to lorawan_datasets/<prefix><baseFile>_device_N_dataset.csv
        // Strip .csv extension to get the base name used internally
        QString baseName = csvFilename;
        if (baseName.endsWith(".csv", Qt::CaseInsensitive))
            baseName.chop(4);

        QDir datasetDir(ns3Dir + "/lorawan_datasets");
        if (!datasetDir.exists()) {
            logMessage("lorawan_datasets/ folder not found in ns-3-dev — no files to collect", "WARNING");
            return;
        }

        // Pattern: <prefix><baseName>_device_*_dataset.csv
        QString pattern = expectedPrefix + baseName + "_device_*_dataset.csv";
        QStringList deviceFiles = datasetDir.entryList(QStringList() << pattern, QDir::Files);

        if (deviceFiles.isEmpty()) {
            // Fallback: grab ALL files starting with the prefix (baseName may differ)
            deviceFiles = datasetDir.entryList(QStringList() << (expectedPrefix + "*_device_*_dataset.csv"), QDir::Files);
        }

        int deviceFilesCopied = 0;
        for (const QString& file : deviceFiles) {
            QString srcPath  = datasetDir.absoluteFilePath(file);
            QString destPath = csvDir + "/" + file;
            if (QFile::exists(destPath)) QFile::remove(destPath);
            if (QFile::copy(srcPath, destPath)) {
                deviceFilesCopied++;
                QFile::remove(srcPath);   // move semantics — clean up source
            }
        }

        if (deviceFilesCopied > 0) {
            logMessage(QString("Collected %1 device dataset file(s) for '%2'")
                       .arg(deviceFilesCopied).arg(adrMode), "SUCCESS");
            // Set the first file as the current output so Export button works
            if (m_currentOutputFile.isEmpty())
                m_currentOutputFile = csvDir + "/" + deviceFiles.first();
        } else {
            logMessage(QString("No device dataset files found for '%1' (pattern: %2)")
                       .arg(adrMode).arg(pattern), "WARNING");
        }

        logMessage(QString("File organization complete for '%1'").arg(adrMode), "SUCCESS");
    }, Qt::QueuedConnection);
}

void MainWindow::runSimulationThread() {
    auto reportProgress = [this](int value, const QString& status = QString()) {
        QMetaObject::invokeMethod(this, [this, value, status]() {
            // Keep progress monotonic while the simulation is running.
            if (value > m_progressBar->value()) {
                m_progressBar->setValue(value);
            }
            if (!status.isEmpty()) {
                updateStatusBar(status);
            }
        }, Qt::QueuedConnection);
    };

    reportProgress(5, "Preparing simulation...");

    QMetaObject::invokeMethod(this, [this]() {
        logMessage("Building ns-3 simulation command...", "INFO");
    }, Qt::QueuedConnection);
    
    // Determine output file path in our structured directory
    QString csvDir = m_outputDirectory + "/results-csv";
    QString csvFilename = "advanced_pre_tx_rssi_dataset.csv";
    QString outputCsvFile = csvDir + "/" + csvFilename;
    
    QMetaObject::invokeMethod(this, [this, outputCsvFile]() {
        logMessage("Output CSV file: " + outputCsvFile, "INFO");
    }, Qt::QueuedConnection);

    reportProgress(12, "Building command...");
    
#ifdef ENABLE_NS3
    try {
        reportProgress(20, "Initializing ns-3 runner...");

        QMetaObject::invokeMethod(this, [this]() {
            logMessage("Running ns-3 compiled simulation...", "INFO");
        }, Qt::QueuedConnection);
        
        // Determine ADR method(s)
        QString adrMode = m_adrModeCombo->currentData().toString();
        QStringList adrModes;
        if (adrMode == "all") {
            adrModes << "off" << "on" << "ddqn" << "ppo" << "marl";
            QMetaObject::invokeMethod(this, [this]() {
                logMessage("Running comparison mode - will execute 5 simulations (off, on, ddqn, ppo, marl)", "INFO");
            }, Qt::QueuedConnection);
        } else {
            adrModes << adrMode;
        }
        
        // Save simulation parameters to file
        QString paramsFile = m_outputDirectory + "/config/simulation_parameters.txt";
        QFile paramFile(paramsFile);
        if (paramFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&paramFile);
            out << "========== SIMULATION PARAMETERS ==========\n";
            out << "Number of Devices: " << m_numDevicesSpin->value() << "\n";
            out << "Simulation Time: " << m_simTimeSpin->value() << " seconds\n";
            out << "App Period: " << m_appPeriodSpin->value() << " seconds\n";
            out << "Radius: " << m_radiusSpin->value() << " meters\n";
            out << "WiFi Interferers: " << m_wifiInterferersSpin->value() << "\n";
            out << "LTE Interferers: " << m_lteInterferersSpin->value() << "\n";
            out << "ADR Mode(s): " << adrModes.join(", ") << "\n";
            out << "Output CSV: " << m_outputDirectory + "/results-csv/simulation_results.csv" << "\n";
            out << "Environmental Modeling: DISABLED\n";
            out << "==========================================\n";
            paramFile.close();
            
            QMetaObject::invokeMethod(this, [paramsFile]() {
                logMessage("Simulation parameters saved to: " + paramsFile, "INFO");
            }, Qt::QueuedConnection);
        }
        
        // Run simulation(s)
        for (int i = 0; i < adrModes.size(); i++) {
            QString currentAdrMode = adrModes[i];
            int stepBase = 25 + static_cast<int>((40.0 * i) / std::max(1, adrModes.size()));
            reportProgress(stepBase, QString("Running simulation mode %1...").arg(currentAdrMode));
            
            ns3::lorawan::ADRMethod adrMethod;
            if (currentAdrMode == "off") {
                adrMethod = ns3::lorawan::ADRMethod::OFF;
            } else if (currentAdrMode == "ddqn") {
                adrMethod = ns3::lorawan::ADRMethod::DDQN;
            } else if (currentAdrMode == "ppo") {
                adrMethod = ns3::lorawan::ADRMethod::PPO;
            } else if (currentAdrMode == "marl") {
                adrMethod = ns3::lorawan::ADRMethod::MARL;
            } else {
                adrMethod = ns3::lorawan::ADRMethod::ON;
            }
            
            QMetaObject::invokeMethod(this, [this, currentAdrMode, i, adrModes]() {
                if (adrModes.size() > 1) {
                    logMessage(QString("========== Simulation %1 of %2: ADR mode '%3' ==========")
                              .arg(i+1).arg(adrModes.size()).arg(currentAdrMode), "INFO");
                }
            }, Qt::QueuedConnection);
            
            // Create simulation runner with parameters
            ns3::lorawan::SimulationRunner runner(
                m_numDevicesSpin->value(),              // nDevices
                m_simTimeSpin->value(),                 // simulationTime
                m_appPeriodSpin->value(),               // appPeriodSeconds
                m_radiusSpin->value(),                  // radius
                outputCsvFile.toStdString(),            // csvFileName
                adrMethod,                              // adrMethod
                m_wifiInterferersSpin->value(),         // nWifiInterferers
                false                                   // enableEnvironmentalModeling
            );
            
            QMetaObject::invokeMethod(this, [this]() {
                logMessage("ns-3 simulation initialized successfully", "INFO");
                logMessage("Executing simulation (this may take a while)...", "INFO");
            }, Qt::QueuedConnection);
            
            // Run simulation
            runner.Run();

            int stepDone = 30 + static_cast<int>((50.0 * (i + 1)) / std::max(1, adrModes.size()));
            reportProgress(stepDone, QString("Completed mode %1").arg(currentAdrMode));
            
            QMetaObject::invokeMethod(this, [this, currentAdrMode, i, adrModes]() {
                if (adrModes.size() > 1) {
                    logMessage(QString("Completed simulation %1 of %2 (ADR mode '%3')")
                              .arg(i+1).arg(adrModes.size()).arg(currentAdrMode), "SUCCESS");
                } else {
                    logMessage("ns-3 simulation completed successfully", "SUCCESS");
                }
            }, Qt::QueuedConnection);
        }
        
        m_currentOutputFile = outputCsvFile;
        reportProgress(85, "Simulation completed, preparing analysis...");
        
    } catch (const std::exception& e) {
        QString errorMsg = QString("Simulation error: %1").arg(e.what());
        QMetaObject::invokeMethod(this, [this, errorMsg]() {
            logMessage(errorMsg, "ERROR");
            QMessageBox::critical(this, "Simulation Error", errorMsg);
        }, Qt::QueuedConnection);
    }
#else
    // Fallback: Run ns-3 via command line
    reportProgress(20, "Initializing command-line run...");

    QMetaObject::invokeMethod(this, [this]() {
        logMessage("Running via ns3 command line (advanced-ddqn-per-adr-example)...", "INFO");
    }, Qt::QueuedConnection);
    
    // Find ns-3-dev directory by walking up from the binary's location
    // Binary is at: .../ns-3-dev/lorawan-qt-gui/build/lorawan-gui
    // We look for the directory containing the 'ns3' run script.
    QString ns3Dir;
    QDir searchDir(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 6; ++i) {
        if (QFile::exists(searchDir.absolutePath() + "/ns3")) {
            ns3Dir = searchDir.absolutePath();
            break;
        }
        searchDir.cdUp();
    }
    
    if (ns3Dir.isEmpty()) {
        m_lastSimulationSucceeded = false;
        QMetaObject::invokeMethod(this, [this]() {
            logMessage("ERROR: Could not find ns-3-dev directory", "ERROR");
        }, Qt::QueuedConnection);
        return;
    }
    
    QString adrMode = m_adrModeCombo->currentData().toString();
    
    // Build ns-3 command for advanced-ddqn-per-adr-example
    QString command = QString("./ns3 run \"advanced-ddqn-per-adr-example "
                             "--nDevices=%1 --radius=%2 --simulationTime=%3 "
                             "--appPeriod=%4 --adr=%5 --csvFile=%6 "
                             "--wifiInterferers=%7 --lteInterferers=%8\"")
                        .arg(m_numDevicesSpin->value())
                        .arg(m_radiusSpin->value())
                        .arg(m_simTimeSpin->value())
                        .arg(m_appPeriodSpin->value())
                        .arg(adrMode)
                        .arg(csvFilename)
                        .arg(m_wifiInterferersSpin->value())
                        .arg(m_lteInterferersSpin->value());
    
    // Save command to file
    QString cmdFile = m_outputDirectory + "/config/simulation_command.sh";
    QFile commandFile(cmdFile);
    if (commandFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&commandFile);
        out << "#!/bin/bash\n";
        out << "# Simulation command for ADR mode: " << adrMode << "\n";
        out << "cd " << ns3Dir << "\n";
        out << command << "\n";
        commandFile.close();
        commandFile.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
    }
    
    QMetaObject::invokeMethod(this, [this, ns3Dir, command, cmdFile]() {
        logMessage("Command saved to: " + cmdFile, "INFO");
        logMessage("Working directory: " + ns3Dir, "INFO");
        logMessage("Executing: " + command, "CMD");
        logMessage("------- Simulation Output -------", "INFO");
    }, Qt::QueuedConnection);

    reportProgress(30, "Launching ns-3 process...");
    
    // Create process and store pointer for potential stop
    QProcess* process = new QProcess();
    m_simulationProcess = process;
    process->setWorkingDirectory(ns3Dir);
    process->setProcessChannelMode(QProcess::MergedChannels);  // merge stdout+stderr

    // On macOS, prefer Command Line Tools
    if (QDir("/Library/Developer/CommandLineTools").exists()) {
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("DEVELOPER_DIR", "/Library/Developer/CommandLineTools");
        process->setProcessEnvironment(env);
    }

    process->start("/bin/bash", QStringList() << "-c" << command);
    
    if (!process->waitForStarted(5000)) {
        m_lastSimulationSucceeded = false;
        m_simulationProcess = nullptr;
        delete process;
        QMetaObject::invokeMethod(this, [this, adrMode]() {
            logMessage(QString("Failed to start ns-3 process for ADR mode '%1'").arg(adrMode), "ERROR");
        }, Qt::QueuedConnection);
        return;
    }
    
    reportProgress(35, "Simulation running...");

    // ---- Stream output live and parse progress ----
    // ALL lines go to the log file; only milestone lines appear in the UI panel.
    const bool isAllMode = (adrMode == "all");

    // Decide if a line is worth showing in the results panel
    auto isImportantLine = [](const QString& l) -> bool {
        // step markers ([1/7] … [7/7])
        if (l.contains(QRegExp(R"(\[\d/7\])"))) return true;
        // simulation milestone output
        static const QStringList keywords = {
            "Starting simulation with parameters",
            "Simulation Time:", "Number of Devices:", "App Period:",
            "Configuring", "ADR trace sources connected",
            "Total packets transmitted", "Successfully received",
            "Packet Delivery Ratio", "Per-device datasets",
            "All seven methods", "completed",
            "Cleaning", "Old datasets removed",
            "ERROR", "error:", "Warning", "WARNING",
        };
        for (const QString& kw : keywords)
            if (l.contains(kw, Qt::CaseInsensitive)) return true;
        return false;
    };

    // Strip emoji characters from a line before displaying it
    auto stripEmojis = [](const QString& line) -> QString {
        QString result;
        result.reserve(line.size());
        int i = 0;
        while (i < line.size()) {
            const QChar c = line[i];
            const ushort code = c.unicode();
            // Skip surrogate pairs (emoji outside BMP, e.g. 🔧 🧹 🔒)
            if (c.isHighSurrogate() && i + 1 < line.size() && line[i + 1].isLowSurrogate()) {
                i += 2;
                continue;
            }
            // Skip common BMP emoji/symbol ranges
            if ((code >= 0x2600 && code <= 0x27BF) ||   // Misc Symbols, Dingbats (✅ ✂ etc.)
                (code >= 0x2300 && code <= 0x23FF) ||   // Misc Technical (⌚ ⏰ etc.)
                (code >= 0x2B00 && code <= 0x2BFF) ||   // Misc Symbols & Arrows
                code == 0x200D ||                        // Zero-width joiner
                code == 0xFE0F) {                        // Variation selector-16
                ++i;
                continue;
            }
            result += c;
            ++i;
        }
        return result.trimmed();
    };

    // Wall-clock-based progress while simulation runs (35 → 82%)
    qint64 simStartMs = QDateTime::currentMSecsSinceEpoch();
    double simTimeSec = m_simTimeSpin->value();        // total ns-3 sim seconds
    // Empirical: ns-3 runs ~3× faster than wall-clock for typical LoRaWAN setups.
    // We cap wall-clock estimate at 82% until we see the final summary line.
    bool simulationStarted = false;   // flips when we see "ADR trace sources connected"

    auto wallClockProgress = [&]() {
        if (!simulationStarted) return;
        double elapsedSec = (QDateTime::currentMSecsSinceEpoch() - simStartMs) / 1000.0;
        // Assume roughly simTimeSec / 3 wall-clock seconds for the run phase
        double estimatedTotalSec = simTimeSec / 3.0;
        double frac = std::min(elapsedSec / std::max(estimatedTotalSec, 1.0), 1.0);
        int pct = 55 + static_cast<int>(frac * 27.0);  // 55 → 82%
        reportProgress(pct, QString("Simulation running… (~%1% estimated)")
                                .arg(static_cast<int>(frac * 100)));
    };

    while (!m_stopRequested.load() && process->state() != QProcess::NotRunning) {
        if (process->waitForReadyRead(300)) {
            while (process->canReadLine()) {
                QString line = QString::fromUtf8(process->readLine()).trimmed();
                if (line.isEmpty()) continue;

                // Always write to log file
                if (m_logStream) {
                    QString timestamp = QDateTime::currentDateTime()
                                            .toString("yyyy-MM-dd HH:mm:ss.zzz");
                    *m_logStream << "[" << timestamp << "] [OUTPUT] " << line << "\n";
                    m_logStream->flush();
                }

                // Show only important lines in the UI panel (emoji stripped)
                if (isImportantLine(line)) {
                    QString displayLine = stripEmojis(line);
                    QMetaObject::invokeMethod(this, [this, displayLine]() {
                        if (m_resultsText) {
                            if (!m_resultsText->toPlainText().isEmpty() &&
                                m_resultsText->toPlainText().startsWith("<"))
                                m_resultsText->clear();
                            m_resultsText->append(displayLine);
                        }
                    }, Qt::QueuedConnection);
                }

                // --- Progress parsing ---
                if (isAllMode) {
                    QRegExp rxStep(R"(\[(\d)/7\])");
                    if (rxStep.indexIn(line) != -1) {
                        int step = rxStep.cap(1).toInt();
                        int pct = 35 + static_cast<int>((step - 1) * 55.0 / 7);
                        reportProgress(pct, QString("[%1/7] Running...").arg(step));
                    }
                } else {
                    if (line.contains("Starting simulation with parameters")) {
                        reportProgress(38, "Simulation initialising...");
                    } else if (line.contains("Configuring")) {
                        reportProgress(45, "Configuring devices...");
                    } else if (line.contains("ADR trace sources connected")) {
                        simulationStarted = true;
                        simStartMs = QDateTime::currentMSecsSinceEpoch();
                        reportProgress(55, "Simulation running...");
                    } else if (line.contains("Total packets transmitted")) {
                        reportProgress(85, "Collecting results...");
                    } else if (line.contains("Packet Delivery Ratio")) {
                        reportProgress(92, "Finalising...");
                    } else {
                        // Advance time-based estimate during the run phase
                        wallClockProgress();
                    }
                }
            }
        } else {
            // No new data — still nudge the wall-clock progress bar
            wallClockProgress();
        }
    }

    // Drain any remaining output after process exits — write to file only
    QByteArray remaining = process->readAll();
    if (!remaining.isEmpty() && m_logStream) {
        QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz");
        for (const QString& line : QString::fromUtf8(remaining).split('\n')) {
            QString trimmed = line.trimmed();
            if (!trimmed.isEmpty())
                *m_logStream << "[" << timestamp << "] [OUTPUT] " << trimmed << "\n";
        }
        m_logStream->flush();
    }

    int exitCode = process->exitCode();
    QProcess::ExitStatus exitStatus = process->exitStatus();
    m_simulationProcess = nullptr;
    delete process;

    QMetaObject::invokeMethod(this, [this]() {
        logMessage("----- End of Simulation Output -----", "INFO");
    }, Qt::QueuedConnection);

    if (m_stopRequested.load()) {
        m_lastSimulationSucceeded = false;
        return;
    }

    bool simulationSucceeded = (exitCode == 0 && exitStatus == QProcess::NormalExit);
    m_lastSimulationSucceeded = simulationSucceeded;

    QMetaObject::invokeMethod(this, [this, simulationSucceeded, exitCode]() {
        logMessage(QString("ns-3 process exited with code: %1").arg(exitCode),
                   simulationSucceeded ? "SUCCESS" : "ERROR");
    }, Qt::QueuedConnection);

    if (!simulationSucceeded) {
        QMetaObject::invokeMethod(this, [this, adrMode]() {
            logMessage(QString("Simulation failed for ADR mode '%1'").arg(adrMode), "ERROR");
        }, Qt::QueuedConnection);
        return;
    }

    reportProgress(78, "Collecting simulation outputs...");

    // ---- Run analysis.py while lorawan_datasets/ still has the new files ----
    {
        m_graphsDir = QString();
        reportProgress(83, "Analysing results...");
        QMetaObject::invokeMethod(this, [this]() {
            logMessage("Running analysis.py on simulation results...", "INFO");
        }, Qt::QueuedConnection);

        QProcess analysisProc;
        analysisProc.setWorkingDirectory(ns3Dir);
        analysisProc.setProcessChannelMode(QProcess::MergedChannels);
        analysisProc.start("python3", QStringList() << "analysis.py" << "lorawan_datasets");
        analysisProc.waitForFinished(-1);

        QString analysisOut = QString::fromUtf8(analysisProc.readAll());
        for (const QString& aline : analysisOut.split('\n')) {
            QString t = aline.trimmed();
            if (t.startsWith("GRAPHS_DIR:")) {
                m_graphsDir = t.mid(11).trimmed();
            }
            if (!t.isEmpty()) {
                QMetaObject::invokeMethod(this, [this, t]() {
                    if (m_resultsText) m_resultsText->append(t);
                }, Qt::QueuedConnection);
            }
        }

        // Fallback: most recently modified subdir of graphs/
        if (m_graphsDir.isEmpty()) {
            QDir graphsRoot(ns3Dir + "/graphs");
            if (graphsRoot.exists()) {
                QFileInfoList subs = graphsRoot.entryInfoList(
                    QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
                if (!subs.isEmpty())
                    m_graphsDir = subs.first().absoluteFilePath();
            }
        }

        int aExit = analysisProc.exitCode();
        QMetaObject::invokeMethod(this, [this, aExit]() {
            logMessage(aExit == 0 ? "Analysis completed successfully" :
                           QString("Analysis exited with code: %1").arg(aExit),
                       aExit == 0 ? "SUCCESS" : "WARNING");
        }, Qt::QueuedConnection);
    }
    reportProgress(91, "Collecting simulation outputs...");

    // Organize files
    if (adrMode == "all") {
        QStringList allModes = {"off", "on", "ddqn", "ppo", "marl", "mappo_gat", "marl_ppo"};
        for (const QString& mode : allModes) {
            organizeSimulationFiles(ns3Dir, csvDir, mode, csvFilename);
        }
    } else {
        organizeSimulationFiles(ns3Dir, csvDir, adrMode, csvFilename);
    }

    reportProgress(85, "Simulation completed, preparing analysis...");

    QMetaObject::invokeMethod(this, [this, simulationSucceeded, adrMode]() {
        if (simulationSucceeded) {
            logMessage("========================================", "SUCCESS");
            if (adrMode == "all") {
                logMessage("All 7 ADR mode simulations completed successfully!", "SUCCESS");
            } else {
                logMessage("Simulation completed successfully!", "SUCCESS");
            }
            logMessage("========================================", "SUCCESS");
        } else {
            logMessage("Simulation failed - check log for details", "ERROR");
        }
    }, Qt::QueuedConnection);
#endif
}

void MainWindow::runAnalysisScript() {
    // Analysis is now performed inside runSimulationThread before file organisation.
    // This stub is kept for compatibility but does nothing.
}

void MainWindow::showResultsDialog(const QString& graphsDir) {
    if (graphsDir.isEmpty()) {
        QMessageBox::information(this, "Simulation Complete",
            "Simulation completed.\n\n"
            "No analysis graphs were generated — check that analysis.py "
            "and its Python dependencies (matplotlib, scipy, pandas) are installed.");
        return;
    }

    QDialog* dialog = new QDialog(this);
    dialog->setWindowTitle("Simulation Results");
    dialog->resize(720, 560);
    dialog->setAttribute(Qt::WA_DeleteOnClose);

    QVBoxLayout* layout = new QVBoxLayout(dialog);

    QTabWidget* tabs = new QTabWidget();

    // Helper: add a tab for one PNG image
    auto addImageTab = [&](const QString& tabName, const QString& imgPath) {
        if (!QFile::exists(imgPath)) return;
        QWidget* tab = new QWidget();
        QVBoxLayout* tl = new QVBoxLayout(tab);
        tl->setContentsMargins(8, 8, 8, 8);

        QLabel* imgLabel = new QLabel();
        QPixmap px(imgPath);
        if (!px.isNull()) {
            imgLabel->setPixmap(px.scaled(660, 440, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        } else {
            imgLabel->setText("Could not load image:\n" + imgPath);
            imgLabel->setAlignment(Qt::AlignCenter);
        }
        imgLabel->setAlignment(Qt::AlignCenter);
        tl->addWidget(imgLabel);

        QPushButton* saveBtn = new QPushButton("Save As...");
        saveBtn->setMaximumWidth(120);
        connect(saveBtn, &QPushButton::clicked, [imgPath]() {
            QString dest = QFileDialog::getSaveFileName(
                nullptr, "Save Image", imgPath, "PNG Images (*.png)");
            if (!dest.isEmpty()) QFile::copy(imgPath, dest);
        });
        QHBoxLayout* bl = new QHBoxLayout();
        bl->addStretch();
        bl->addWidget(saveBtn);
        tl->addLayout(bl);

        tabs->addTab(tab, tabName);
    };

    addImageTab("PDR",      graphsDir + "/pdr_by_adr_type.png");
    addImageTab("Energy",   graphsDir + "/energy_by_adr_type.png");
    addImageTab("Fairness", graphsDir + "/fairness_by_adr_type.png");

    if (tabs->count() == 0) {
        QLabel* noData = new QLabel("No charts found in:\n" + graphsDir);
        noData->setAlignment(Qt::AlignCenter);
        layout->addWidget(noData);
    } else {
        layout->addWidget(tabs);
    }

    QPushButton* closeBtn = new QPushButton("Close");
    connect(closeBtn, &QPushButton::clicked, dialog, &QDialog::close);
    QHBoxLayout* cl = new QHBoxLayout();
    cl->addStretch();
    cl->addWidget(closeBtn);
    layout->addLayout(cl);

    dialog->show();
}

void MainWindow::onSimulationFinished() {
    m_simulationRunning = false;
    m_runButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    m_progressBar->setValue(100);
    
    if (m_simulationWatcher->isCanceled()) {
        logMessage("Simulation was canceled by user", "WARNING");
        updateStatusBar("Simulation canceled");
        return;
    }
    
    logMessage("========================================", "INFO");
    logMessage("Simulation thread finished", "INFO");
    logMessage("Processing results...", "INFO");

    if (!m_lastSimulationSucceeded) {
        updateStatusBar("Simulation failed");
        m_exportResultsButton->setEnabled(false);
        m_visualizeButton->setEnabled(false);
        logMessage("Simulation failed, skipping analysis", "ERROR");
        logMessage("Check simulation.log for the root cause and rerun.", "ERROR");
        logMessage("========================================", "ERROR");
        return;
    }
    
    updateStatusBar("Simulation completed!");
    
    // Show results dialog with analysis charts
    showResultsDialog(m_graphsDir);
    
    m_exportResultsButton->setEnabled(true);
    m_visualizeButton->setEnabled(true);
    
    logMessage("========================================", "SUCCESS");
    logMessage("Simulation run complete!", "SUCCESS");
    logMessage(QString("All results saved to: %1").arg(m_outputDirectory), "SUCCESS");
    logMessage("========================================", "SUCCESS");
}

void MainWindow::displayResults(const QString& csvPath) {
    logMessage("Reading results from: " + csvPath, "INFO");
    
    QFile file(csvPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QString errorMsg = "Error: Could not open results file: " + csvPath;
        m_resultsText->setPlainText(errorMsg);
        logMessage(errorMsg, "ERROR");
        return;
    }
    
    QTextStream in(&file);
    QString content = in.readAll();
    file.close();
    
    logMessage(QString("Results file size: %1 bytes").arg(content.size()), "INFO");
    
    // Parse CSV and compute statistics
    QStringList lines = content.split('\n');
    logMessage(QString("Total lines in CSV: %1").arg(lines.size()), "INFO");
    
    int totalDevices = 0;
    double totalPDR = 0.0;
    int sf7 = 0, sf8 = 0, sf9 = 0, sf10 = 0, sf11 = 0, sf12 = 0;
    int totalPacketsSent = 0;
    int totalPacketsReceived = 0;
    
    for (int i = 1; i < lines.size(); ++i) {  // Skip header
        QStringList fields = lines[i].split(',');
        if (fields.size() >= 5) {
            totalDevices++;
            double pdr = fields[4].toDouble();
            totalPDR += pdr;
            
            // Try to extract packet counts if available
            if (fields.size() >= 7) {
                totalPacketsSent += fields[5].toInt();
                totalPacketsReceived += fields[6].toInt();
            }
            
            int sf = fields[2].toInt();  // SF column
            switch(sf) {
                case 7: sf7++; break;
                case 8: sf8++; break;
                case 9: sf9++; break;
                case 10: sf10++; break;
                case 11: sf11++; break;
                case 12: sf12++; break;
            }
        }
    }
    
    double avgPDR = totalDevices > 0 ? totalPDR / totalDevices : 0.0;
    
    // Log statistics
    logMessage("=== Results Statistics ===", "INFO");
    logMessage(QString("Total Devices: %1").arg(totalDevices), "INFO");
    logMessage(QString("Average PDR: %1%").arg(avgPDR, 0, 'f', 2), "INFO");
    logMessage(QString("SF Distribution: SF7=%1, SF8=%2, SF9=%3, SF10=%4, SF11=%5, SF12=%6")
        .arg(sf7).arg(sf8).arg(sf9).arg(sf10).arg(sf11).arg(sf12), "INFO");
    
    if (totalPacketsSent > 0) {
        logMessage(QString("Total Packets Sent: %1").arg(totalPacketsSent), "INFO");
        logMessage(QString("Total Packets Received: %1").arg(totalPacketsReceived), "INFO");
        logMessage(QString("Overall Network PDR: %1%")
            .arg(100.0 * totalPacketsReceived / totalPacketsSent, 0, 'f', 2), "INFO");
    }
    
    QString summary = QString(
        "<h3>Simulation Summary</h3>"
        "<p><b>Total Devices:</b> %1</p>"
        "<p><b>Average PDR:</b> %2%</p>"
        "<p><b>Output Directory:</b> <a href=\"file://%3\">%3</a></p>"
        "<p><b>CSV File:</b> %4</p>"
        "<h4>Spreading Factor Distribution:</h4>"
        "<ul>"
        "<li>SF7: %5 devices</li>"
        "<li>SF8: %6 devices</li>"
        "<li>SF9: %7 devices</li>"
        "<li>SF10: %8 devices</li>"
        "<li>SF11: %9 devices</li>"
        "<li>SF12: %10 devices</li>"
        "</ul>"
    ).arg(totalDevices)
     .arg(avgPDR, 0, 'f', 2)
     .arg(m_outputDirectory)
     .arg(csvPath)
     .arg(sf7).arg(sf8).arg(sf9).arg(sf10).arg(sf11).arg(sf12);
    
    if (totalPacketsSent > 0) {
        summary += QString(
            "<h4>Packet Statistics:</h4>"
            "<ul>"
            "<li>Total Packets Sent: %1</li>"
            "<li>Total Packets Received: %2</li>"
            "<li>Overall Network PDR: %3%</li>"
            "</ul>"
        ).arg(totalPacketsSent)
         .arg(totalPacketsReceived)
         .arg(100.0 * totalPacketsReceived / totalPacketsSent, 0, 'f', 2);
    }
    
    m_resultsText->setHtml(summary);
    logMessage("Results displayed successfully", "SUCCESS");
}

// ============================================================================
// Helper Methods
// ============================================================================

bool MainWindow::validateParameters() {
    if (m_numDevicesSpin->value() < 1) {
        QMessageBox::warning(this, "Invalid Parameters", "Number of devices must be at least 1");
        return false;
    }
    
    return true;
}

void MainWindow::updateStatusBar(const QString& message) {
    statusBar()->showMessage(message);
    m_statusLabel->setText(message);
}

void MainWindow::updateMapInfo() {
    int numNodes = m_mapScene->getNodePositions().size() - 1;  // Exclude gateway
    int numObstacles = m_mapScene->getObstacles().size();
    m_mapInfoLabel->setText(QString("Nodes: %1 | Obstacles: %2").arg(numNodes).arg(numObstacles));
}

// ============================================================================
// Slot Implementations
// ============================================================================

void MainWindow::onNumDevicesChanged(int value) {
    // Update map if simulation not running
    if (!m_simulationRunning) {
        m_mapScene->createNodes(value, m_radiusSpin->value());
    }
}

void MainWindow::onRadiusChanged(double value) {
    m_mapScene->setRadius(value);
    if (!m_simulationRunning) {
        m_mapScene->createNodes(m_numDevicesSpin->value(), value);
    }
}

void MainWindow::onNodePositionChanged(int nodeId, const QPointF& pos) {
    updateMapInfo();
}

void MainWindow::onObstacleAdded(const ObstacleData& obstacle) {
    updateMapInfo();
}

void MainWindow::onObstacleRemoved(int index) {
    updateMapInfo();
}

void MainWindow::onClearMap() {
    m_mapScene->clearNodes();
    m_mapScene->clearObstacles();
    updateMapInfo();
}

void MainWindow::onExportConfig() {
    // TODO: Implement configuration export (JSON)
    QMessageBox::information(this, "Export Configuration",
        "Configuration export feature coming soon!");
}

void MainWindow::onImportConfig() {
    // TODO: Implement configuration import (JSON)
    QMessageBox::information(this, "Import Configuration",
        "Configuration import feature coming soon!");
}

void MainWindow::onSimulationProgress(int progress) {
    m_progressBar->setValue(progress);
}

// ============================================================================
// Logging and Output Management
// ============================================================================

void MainWindow::createOutputDirectory() {
    // Create output directories in lorawan-qt-gui root
    QString guiBaseDir = QDir(QCoreApplication::applicationDirPath()).absolutePath();
    // Go up one level from build directory to lorawan-qt-gui root
    QDir guiDir(guiBaseDir);
    guiDir.cdUp();
    QString guiRoot = guiDir.absolutePath();
    
    // Create timestamped output directory
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    m_outputDirectory = guiRoot + "/simulation_results/run_" + timestamp;
    
    QDir dir;
    if (!dir.mkpath(m_outputDirectory)) {
        QMessageBox::warning(this, "Directory Error", 
            "Could not create output directory: " + m_outputDirectory);
        m_outputDirectory = guiRoot;
        return;
    }
    
    // Create subdirectories with specific names requested
    dir.mkpath(m_outputDirectory + "/results-csv");
    dir.mkpath(m_outputDirectory + "/plots");
    dir.mkpath(m_outputDirectory + "/config");
    
    // Initialize log file
    QString logPath = m_outputDirectory + "/simulation.log";
    m_logFile = new QFile(logPath);
    if (m_logFile->open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Append)) {
        m_logStream = new QTextStream(m_logFile);
    }
}

QString MainWindow::getOutputDirectory() const {
    return m_outputDirectory;
}

void MainWindow::logMessage(const QString& message, const QString& level) {
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz");
    QString logLine = QString("[%1] [%2] %3").arg(timestamp).arg(level).arg(message);
    
    // Write to file
    if (m_logStream) {
        *m_logStream << logLine << "\n";
        m_logStream->flush();
    }
    
    // Write to console/terminal
    qDebug().noquote() << logLine;
    
    // Also append to results text if UI is visible
    if (m_resultsText) {
        QString currentText = m_resultsText->toPlainText();
        if (!currentText.isEmpty() && currentText.startsWith("<")) {
            // HTML content - clear it for log view
            m_resultsText->clear();
        }
        m_resultsText->append(logLine);
    }
}

void MainWindow::exportNodePositions(const QString& filepath) {
    QFile file(filepath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        logMessage("Failed to export node positions to: " + filepath, "ERROR");
        return;
    }
    
    QTextStream out(&file);
    out << "NodeID,X,Y,Z\n";  // Changed from NodeID,X,Y,Type
    
    QVector<QPointF> positions = m_mapScene->getNodePositions();
    for (int i = 0; i < positions.size(); ++i) {
        // Gateway at index 0 with height 15m, end devices at 1.5m
        double z = (i == 0) ? 15.0 : 1.5;
        out << i << "," 
            << positions[i].x() << "," 
            << positions[i].y() << "," 
            << z << "\n";
    }
    
    file.close();
    logMessage(QString("Exported %1 node positions to: %2")
        .arg(positions.size()).arg(filepath), "INFO");
}

void MainWindow::exportObstacles(const QString& filepath) {
    QVector<ObstacleData> obstacles = m_mapScene->getObstacles();
    
    if (obstacles.isEmpty()) {
        logMessage("No obstacles to export", "INFO");
        return;
    }
    
    QFile file(filepath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        logMessage("Failed to export obstacles to: " + filepath, "ERROR");
        return;
    }
    
    QTextStream out(&file);
    out << "X,Y,Width,Height,AttenuationDB\n";  // Simplified header
    
    for (int i = 0; i < obstacles.size(); ++i) {
        const ObstacleData& obs = obstacles[i];
        out << obs.x << ","   // Removed ObstacleID column
            << obs.y << "," 
            << obs.width << "," 
            << obs.height << "," 
            << obs.attenuationDb << "\n";
    }
    
    file.close();
    logMessage(QString("Exported %1 obstacles to: %2")
        .arg(obstacles.size()).arg(filepath), "INFO");
}

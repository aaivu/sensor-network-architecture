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

#ifdef ENABLE_NS3
#include "ns3/core-module.h"
#include "ns3/simulation-runner.h"
#endif

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_simulationRunning(false)
{
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
    
    updateStatusBar("Ready");
}

MainWindow::~MainWindow() {
    if (m_simulationRunning) {
        m_simulationWatcher->cancel();
        m_simulationWatcher->waitForFinished();
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
    m_numDevicesSpin->setValue(50);
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
    m_simTimeSpin->setValue(3600);
    m_simTimeSpin->setSingleStep(60);
    simLayout->addWidget(m_simTimeSpin, 0, 1);
    
    simLayout->addWidget(new QLabel("App Period (s):"), 1, 0);
    m_appPeriodSpin = new QDoubleSpinBox();
    m_appPeriodSpin->setRange(1.0, 3600.0);
    m_appPeriodSpin->setValue(60.0);
    m_appPeriodSpin->setSingleStep(10.0);
    simLayout->addWidget(m_appPeriodSpin, 1, 1);
    
    simLayout->addWidget(new QLabel("WiFi Interferers:"), 2, 0);
    m_wifiInterferersSpin = new QSpinBox();
    m_wifiInterferersSpin->setRange(0, 100);
    m_wifiInterferersSpin->setValue(5);
    simLayout->addWidget(m_wifiInterferersSpin, 2, 1);
    
    simLayout->addWidget(new QLabel("ADR Mode:"), 3, 0);
    m_adrModeCombo = new QComboBox();
    m_adrModeCombo->addItem("Off", "off");
    m_adrModeCombo->addItem("Standard", "on");
    m_adrModeCombo->addItem("DDQN-PER", "ddqn");
    m_adrModeCombo->setCurrentIndex(2);  // Default to DDQN
    simLayout->addWidget(m_adrModeCombo, 3, 1);
    
    simGroup->setLayout(simLayout);
    layout->addWidget(simGroup);
    
    // Output File Group
    QGroupBox* outputGroup = new QGroupBox("Output Settings");
    QVBoxLayout* outputLayout = new QVBoxLayout();
    
    QHBoxLayout* fileLayout = new QHBoxLayout();
    fileLayout->addWidget(new QLabel("CSV File:"));
    m_csvFileEdit = new QLineEdit("simulation_results.csv");
    fileLayout->addWidget(m_csvFileEdit, 1);
    m_browseButton = new QPushButton("Browse...");
    connect(m_browseButton, &QPushButton::clicked, [this]() {
        QString file = QFileDialog::getSaveFileName(this, "Output CSV File", 
                                                     m_csvFileEdit->text(),
                                                     "CSV Files (*.csv)");
        if (!file.isEmpty()) {
            m_csvFileEdit->setText(file);
        }
    });
    fileLayout->addWidget(m_browseButton);
    outputLayout->addLayout(fileLayout);
    
    outputGroup->setLayout(outputLayout);
    layout->addWidget(outputGroup);
    
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
        // TODO: Launch Python visualization script
        QMessageBox::information(this, "Visualize", 
            "Launch python_visualization.py with output CSV:\n" + m_currentOutputFile);
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

void MainWindow::onRunSimulation() {
    if (!validateParameters()) {
        return;
    }
    
    // Create nodes on map if not already done
    if (m_mapScene->getNodePositions().isEmpty()) {
        m_mapScene->createNodes(m_numDevicesSpin->value(), m_radiusSpin->value());
    }
    
    m_simulationRunning = true;
    m_runButton->setEnabled(false);
    m_stopButton->setEnabled(true);
    m_progressBar->setValue(0);
    updateStatusBar("Running simulation...");
    
    // Run in background thread
    QFuture<void> future = QtConcurrent::run([this]() {
        runSimulationThread();
    });
    
    m_simulationWatcher->setFuture(future);
}

void MainWindow::onStopSimulation() {
    m_simulationWatcher->cancel();
    m_simulationRunning = false;
    m_runButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    updateStatusBar("Simulation stopped by user");
}

void MainWindow::runSimulationThread() {
#ifdef ENABLE_NS3
    try {
        // Determine ADR method
        QString adrMode = m_adrModeCombo->currentData().toString();
        ns3::lorawan::ADRMethod adrMethod;
        if (adrMode == "off") {
            adrMethod = ns3::lorawan::ADRMethod::OFF;
        } else if (adrMode == "ddqn") {
            adrMethod = ns3::lorawan::ADRMethod::DDQN;
        } else {
            adrMethod = ns3::lorawan::ADRMethod::ON;
        }
        
        // Create simulation runner with parameters
        ns3::lorawan::SimulationRunner runner(
            m_numDevicesSpin->value(),              // nDevices
            m_simTimeSpin->value(),                 // simulationTime
            m_appPeriodSpin->value(),               // appPeriodSeconds
            m_radiusSpin->value(),                  // radius
            m_csvFileEdit->text().toStdString(),    // csvFileName
            adrMethod,                              // adrMethod
            m_wifiInterferersSpin->value(),         // nWifiInterferers
            m_environmentModelingCheck->isChecked() // enableEnvironmentalModeling
        );
        
        // Note: Node positions and obstacles from map editor are not currently
        // passed to the simulation runner as it generates random positions.
        // This could be extended in future versions.
        
        // Run simulation
        runner.Run();
        
        m_currentOutputFile = m_csvFileEdit->text();
        
    } catch (const std::exception& e) {
        // Error handling done in onSimulationFinished
        QString errorMsg = QString("Simulation error: %1").arg(e.what());
        QMetaObject::invokeMethod(this, [this, errorMsg]() {
            QMessageBox::critical(this, "Simulation Error", errorMsg);
        }, Qt::QueuedConnection);
    }
#else
    // Placeholder simulation when ns-3 is not enabled
    // Simulate a delay
    QThread::msleep(2000);
    m_currentOutputFile = m_csvFileEdit->text();
#endif
}

void MainWindow::onSimulationFinished() {
    m_simulationRunning = false;
    m_runButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    m_progressBar->setValue(100);
    
    if (m_simulationWatcher->isCanceled()) {
        updateStatusBar("Simulation canceled");
        return;
    }
    
    updateStatusBar("Simulation completed!");
    displayResults(m_csvFileEdit->text());
    
    m_exportResultsButton->setEnabled(true);
    m_visualizeButton->setEnabled(true);
}

void MainWindow::displayResults(const QString& csvPath) {
    QFile file(csvPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        m_resultsText->setPlainText("Error: Could not open results file");
        return;
    }
    
    QTextStream in(&file);
    QString content = in.readAll();
    file.close();
    
    // Parse CSV and compute statistics
    QStringList lines = content.split('\n');
    int totalDevices = 0;
    double totalPDR = 0.0;
    int sf7 = 0, sf8 = 0, sf9 = 0, sf10 = 0, sf11 = 0, sf12 = 0;
    
    for (int i = 1; i < lines.size(); ++i) {  // Skip header
        QStringList fields = lines[i].split(',');
        if (fields.size() >= 5) {
            totalDevices++;
            totalPDR += fields[4].toDouble();  // PDR column
            
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
    
    QString summary = QString(
        "<h3>Simulation Summary</h3>"
        "<p><b>Total Devices:</b> %1</p>"
        "<p><b>Average PDR:</b> %2%</p>"
        "<p><b>Output File:</b> %3</p>"
        "<h4>Spreading Factor Distribution:</h4>"
        "<ul>"
        "<li>SF7: %4 devices</li>"
        "<li>SF8: %5 devices</li>"
        "<li>SF9: %6 devices</li>"
        "<li>SF10: %7 devices</li>"
        "<li>SF11: %8 devices</li>"
        "<li>SF12: %9 devices</li>"
        "</ul>"
    ).arg(totalDevices)
     .arg(totalDevices > 0 ? totalPDR / totalDevices : 0.0, 0, 'f', 2)
     .arg(csvPath)
     .arg(sf7).arg(sf8).arg(sf9).arg(sf10).arg(sf11).arg(sf12);
    
    m_resultsText->setHtml(summary);
}

// ============================================================================
// Helper Methods
// ============================================================================

bool MainWindow::validateParameters() {
    if (m_numDevicesSpin->value() < 1) {
        QMessageBox::warning(this, "Invalid Parameters", "Number of devices must be at least 1");
        return false;
    }
    
    if (m_csvFileEdit->text().isEmpty()) {
        QMessageBox::warning(this, "Invalid Parameters", "Please specify output CSV filename");
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

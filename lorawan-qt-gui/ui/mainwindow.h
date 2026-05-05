#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsView>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QProgressBar>
#include <QLabel>
#include <QFuture>
#include <QFutureWatcher>
#include <QFile>
#include <QTextStream>
#include "mapscene.h"

// Forward declaration
class SimulationResult;

/**
 * Main application window for LoRaWAN network simulation
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    // UI Interactions
    void onRunSimulation();
    void onStopSimulation();
    void onClearMap();
    void onExportConfig();
    void onImportConfig();
    void onNumDevicesChanged(int value);
    void onRadiusChanged(double value);
    
    // Simulation callbacks
    void onSimulationFinished();
    void onSimulationProgress(int progress);
    
    // Map interactions
    void onNodePositionChanged(int nodeId, const QPointF& pos);
    void onObstacleAdded(const ObstacleData& obstacle);
    void onObstacleRemoved(int index);

private:
    // UI Setup
    void setupUi();
    void setupParameterPanel();
    void setupMapView();
    void setupControlPanel();
    void setupResultsPanel();
    void createActions();
    void createMenuBar();
    
    // Simulation
    void cleanupOldDatasets();
    void runSimulationThread();
    void organizeSimulationFiles(const QString& ns3Dir, const QString& csvDir, 
                                  const QString& adrMode, const QString& csvFilename);
    void displayResults(const QString& csvPath);
    void runAnalysisScript();
    void updateStatusBar(const QString& message);
    void updateMapInfo();
    
    // Data validation
    bool validateParameters();
    QString getOutputFilename() const;
    
    // Logging and output management
    void logMessage(const QString& message, const QString& level = "INFO");
    void createOutputDirectory();
    QString getOutputDirectory() const;
    void exportNodePositions(const QString& filepath);
    void exportObstacles(const QString& filepath);
    
    // UI Components - Parameter Panel
    QWidget* m_parameterPanel;
    QSpinBox* m_numDevicesSpin;
    QDoubleSpinBox* m_radiusSpin;
    QSpinBox* m_simTimeSpin;
    QDoubleSpinBox* m_appPeriodSpin;
    QSpinBox* m_wifiInterferersSpin;
    QComboBox* m_adrModeCombo;
    
    // UI Components - Map View
    QGraphicsView* m_mapView;
    MapScene* m_mapScene;
    QPushButton* m_addObstacleButton;
    QPushButton* m_clearMapButton;
    QPushButton* m_randomizeButton;
    QPushButton* m_zoomInButton;
    QPushButton* m_zoomOutButton;
    QPushButton* m_zoomResetButton;
    QLabel* m_mapInfoLabel;
    
    // UI Components - Control Panel
    QWidget* m_controlPanel;
    QPushButton* m_runButton;
    QPushButton* m_stopButton;
    QProgressBar* m_progressBar;
    QLabel* m_statusLabel;
    
    // UI Components - Results Panel
    QWidget* m_resultsPanel;
    QTextEdit* m_resultsText;
    QPushButton* m_exportResultsButton;
    QPushButton* m_visualizeButton;
    
    // Simulation state
    QFutureWatcher<void>* m_simulationWatcher;
    bool m_simulationRunning;
    bool m_lastSimulationSucceeded;
    QString m_currentOutputFile;
    QString m_outputDirectory;
    QFile* m_logFile;
    QTextStream* m_logStream;
    
    // Actions
    QAction* m_newAction;
    QAction* m_openAction;
    QAction* m_saveAction;
    QAction* m_exitAction;
    QAction* m_aboutAction;
};

/**
 * Structure to hold simulation results
 */
struct SimulationResult {
    bool success;
    QString errorMessage;
    QString outputFile;
    int totalPackets;
    int receivedPackets;
    double avgPDR;
    double avgLatency;
    
    SimulationResult() 
        : success(false)
        , totalPackets(0)
        , receivedPackets(0)
        , avgPDR(0.0)
        , avgLatency(0.0)
    {}
};

#endif // MAINWINDOW_H

#include "ui/mainwindow.h"
#include <QApplication>
#include <QStyleFactory>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    // Application metadata
    app.setApplicationName("LoRaWAN Network Simulator");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("ns-3 LoRaWAN");
    
    // Set modern style
    app.setStyle(QStyleFactory::create("Fusion"));
    
    // Create and show main window
    MainWindow window;
    window.show();
    
    return app.exec();
}

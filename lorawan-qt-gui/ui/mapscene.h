#ifndef MAPSCENE_H
#define MAPSCENE_H

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsRectItem>
#include <QGraphicsTextItem>
#include <QVector>
#include <memory>

// Forward declarations for ns-3 types
namespace ns3 {
    class Vector;
}

/**
 * @brief Represents an obstacle in the environment
 */
struct ObstacleData {
    double x;
    double y;
    double width;
    double height;
    double attenuationDb;
    
    ObstacleData(double x_ = 0, double y_ = 0, double w = 50, double h = 50, double atten = 20.0)
        : x(x_), y(y_), width(w), height(h), attenuationDb(atten) {}
};

/**
 * @brief Custom graphics item for a LoRaWAN node
 */
class NodeItem : public QGraphicsEllipseItem {
public:
    explicit NodeItem(int nodeId, bool isGateway = false, QGraphicsItem* parent = nullptr);
    
    int getNodeId() const { return m_nodeId; }
    bool isGateway() const { return m_isGateway; }
    
    void setHighlight(bool highlight);
    
protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    
private:
    int m_nodeId;
    bool m_isGateway;
    QGraphicsTextItem* m_label;
};

/**
 * @brief Custom graphics item for an obstacle
 */
class ObstacleItem : public QGraphicsRectItem {
public:
    explicit ObstacleItem(const QRectF& rect, QGraphicsItem* parent = nullptr);
    
    void setAttenuation(double db) { m_attenuationDb = db; updateTooltip(); }
    double getAttenuation() const { return m_attenuationDb; }
    
protected:
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    
private:
    void updateTooltip();
    double m_attenuationDb;
};

/**
 * @brief MapScene manages the 2D visualization of the LoRaWAN network
 * 
 * Features:
 * - Draggable nodes (end devices)
 * - Fixed gateway at center
 * - Editable obstacles
 * - Exports positions for simulation
 */
class MapScene : public QGraphicsScene {
    Q_OBJECT
    
public:
    explicit MapScene(QObject* parent = nullptr);
    ~MapScene() override;
    
    // Node management
    void createNodes(int numDevices, double radius);
    void clearNodes();
    int getNodeCount() const { return m_nodes.size(); }
    
    // Obstacle management
    void addObstacle(double x, double y, double width, double height, double attenuationDb = 20.0);
    void removeObstacle(ObstacleItem* obstacle);
    void clearObstacles();
    int getObstacleCount() const { return m_obstacles.size(); }
    
    // Export data for simulation
    QVector<QPointF> getNodePositions() const;
    QVector<ObstacleData> getObstacles() const;
    
    // Configuration
    void setRadius(double radius);
    double getRadius() const { return m_radius; }
    
    void setShowGrid(bool show);
    
signals:
    void nodePositionChanged(int nodeId, QPointF position);
    void obstacleAdded(ObstacleData obstacle);
    void obstacleRemoved(int index);
    
protected:
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    
private:
    void setupGateway();
    void drawGrid(QPainter* painter, const QRectF& rect);
    void generateRandomPositions(int numDevices, double radius);
    
    // Graphics items
    NodeItem* m_gateway;
    QVector<NodeItem*> m_nodes;
    QVector<ObstacleItem*> m_obstacles;
    
    // Configuration
    double m_radius;
    bool m_showGrid;
    
    // Visual elements
    QGraphicsEllipseItem* m_radiusCircle;
};

#endif // MAPSCENE_H

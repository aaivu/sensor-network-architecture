#include "mapscene.h"
#include <QGraphicsSceneContextMenuEvent>
#include <QMenu>
#include <QInputDialog>
#include <QPainter>
#include <QPen>
#include <QBrush>
#include <QRandomGenerator>
#include <QtMath>

// ============================================================================
// NodeItem Implementation
// ============================================================================

NodeItem::NodeItem(int nodeId, bool isGateway, QGraphicsItem* parent)
    : QGraphicsEllipseItem(parent)
    , m_nodeId(nodeId)
    , m_isGateway(isGateway)
{
    // Node appearance
    double size = isGateway ? 30.0 : 20.0;
    setRect(-size/2, -size/2, size, size);
    
    if (isGateway) {
        setBrush(QBrush(QColor(255, 0, 0, 200)));  // Red gateway
        setPen(QPen(Qt::darkRed, 3));
    } else {
        setBrush(QBrush(QColor(0, 120, 255, 180)));  // Blue nodes
        setPen(QPen(Qt::darkBlue, 2));
    }
    
    // Make nodes draggable (except gateway)
    if (!isGateway) {
        setFlag(QGraphicsItem::ItemIsMovable, true);
        setFlag(QGraphicsItem::ItemIsSelectable, true);
        setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    }
    
    // Add label
    m_label = new QGraphicsTextItem(this);
    if (isGateway) {
        m_label->setPlainText("GW");
    } else {
        m_label->setPlainText(QString::number(nodeId));
    }
    m_label->setDefaultTextColor(Qt::white);
    QFont font = m_label->font();
    font.setPointSize(8);
    font.setBold(true);
    m_label->setFont(font);
    
    // Center label
    QRectF textRect = m_label->boundingRect();
    m_label->setPos(-textRect.width()/2, -textRect.height()/2);
    
    setToolTip(isGateway ? "Gateway (Fixed)" : QString("Device %1 (Drag to move)").arg(nodeId));
}

void NodeItem::setHighlight(bool highlight) {
    if (highlight) {
        setPen(QPen(Qt::yellow, 3));
    } else {
        setPen(QPen(m_isGateway ? Qt::darkRed : Qt::darkBlue, 2));
    }
}

void NodeItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsEllipseItem::mouseMoveEvent(event);
    // Signal will be emitted in mouseReleaseEvent
}

void NodeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    QGraphicsEllipseItem::mouseReleaseEvent(event);
    // Notify scene of position change
    if (scene()) {
        auto mapScene = qobject_cast<MapScene*>(scene());
        if (mapScene) {
            emit mapScene->nodePositionChanged(m_nodeId, pos());
        }
    }
}

// ============================================================================
// ObstacleItem Implementation
// ============================================================================

ObstacleItem::ObstacleItem(const QRectF& rect, QGraphicsItem* parent)
    : QGraphicsRectItem(rect, parent)
    , m_attenuationDb(20.0)
{
    setBrush(QBrush(QColor(139, 69, 19, 150)));  // Brown buildings
    setPen(QPen(QColor(101, 50, 14), 2));
    
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    
    updateTooltip();
}

void ObstacleItem::updateTooltip() {
    setToolTip(QString("Obstacle\nAttenuation: %1 dB\nRight-click for options").arg(m_attenuationDb, 0, 'f', 1));
}

void ObstacleItem::contextMenuEvent(QGraphicsSceneContextMenuEvent* event) {
    QMenu menu;
    QAction* changeAttenAction = menu.addAction("Change Attenuation...");
    QAction* deleteAction = menu.addAction("Delete Obstacle");
    
    QAction* selected = menu.exec(event->screenPos());
    if (selected == changeAttenAction) {
        bool ok;
        double newAtten = QInputDialog::getDouble(
            nullptr, "Obstacle Attenuation",
            "Enter attenuation (dB):",
            m_attenuationDb, 0.0, 100.0, 1, &ok
        );
        if (ok) {
            setAttenuation(newAtten);
        }
    } else if (selected == deleteAction) {
        auto mapScene = qobject_cast<MapScene*>(scene());
        if (mapScene) {
            mapScene->removeObstacle(this);
        }
    }
}

// ============================================================================
// MapScene Implementation
// ============================================================================

MapScene::MapScene(QObject* parent)
    : QGraphicsScene(parent)
    , m_gateway(nullptr)
    , m_radius(1000.0)
    , m_showGrid(true)
    , m_radiusCircle(nullptr)
{
    // Set scene size
    setSceneRect(-1500, -1500, 3000, 3000);
    setBackgroundBrush(QBrush(QColor(240, 248, 255)));  // Light blue background
    
    // Create deployment radius circle
    m_radiusCircle = addEllipse(-m_radius, -m_radius, m_radius * 2, m_radius * 2,
                                 QPen(Qt::gray, 2, Qt::DashLine),
                                 QBrush(Qt::NoBrush));
    m_radiusCircle->setZValue(-1);
    
    // Setup gateway at center
    setupGateway();
}

MapScene::~MapScene() {
    clearNodes();
    clearObstacles();
}

void MapScene::setupGateway() {
    if (!m_gateway) {
        m_gateway = new NodeItem(-1, true);
        m_gateway->setPos(0, 0);
        addItem(m_gateway);
        m_gateway->setZValue(10);  // Gateway on top
    }
}

void MapScene::createNodes(int numDevices, double radius) {
    clearNodes();
    m_radius = radius;
    
    // Update radius circle
    if (m_radiusCircle) {
        m_radiusCircle->setRect(-m_radius, -m_radius, m_radius * 2, m_radius * 2);
    }
    
    generateRandomPositions(numDevices, radius);
}

void MapScene::generateRandomPositions(int numDevices, double radius) {
    for (int i = 0; i < numDevices; ++i) {
        // Random position within radius
        double angle = QRandomGenerator::global()->generateDouble() * 360.0 * M_PI / 180.0;
        double distance = radius * 0.3 + QRandomGenerator::global()->generateDouble() * (radius - radius * 0.3);
        
        double x = distance * qCos(angle);
        double y = distance * qSin(angle);
        
        NodeItem* node = new NodeItem(i, false);
        node->setPos(x, y);
        addItem(node);
        m_nodes.append(node);
    }
}

void MapScene::clearNodes() {
    for (auto* node : m_nodes) {
        removeItem(node);
        delete node;
    }
    m_nodes.clear();
}

void MapScene::addObstacle(double x, double y, double width, double height, double attenuationDb) {
    QRectF rect(x - width/2, y - height/2, width, height);
    ObstacleItem* obstacle = new ObstacleItem(rect);
    obstacle->setAttenuation(attenuationDb);
    addItem(obstacle);
    m_obstacles.append(obstacle);
    
    ObstacleData data(x, y, width, height, attenuationDb);
    emit obstacleAdded(data);
}

void MapScene::removeObstacle(ObstacleItem* obstacle) {
    int index = m_obstacles.indexOf(obstacle);
    if (index >= 0) {
        m_obstacles.removeAt(index);
        removeItem(obstacle);
        delete obstacle;
        emit obstacleRemoved(index);
    }
}

void MapScene::clearObstacles() {
    for (auto* obstacle : m_obstacles) {
        removeItem(obstacle);
        delete obstacle;
    }
    m_obstacles.clear();
}

QVector<QPointF> MapScene::getNodePositions() const {
    QVector<QPointF> positions;
    
    // Gateway first (at origin)
    positions.append(QPointF(0, 0));
    
    // Then end devices
    for (const auto* node : m_nodes) {
        positions.append(node->pos());
    }
    
    return positions;
}

QVector<ObstacleData> MapScene::getObstacles() const {
    QVector<ObstacleData> result;
    
    for (const auto* obstacle : m_obstacles) {
        QRectF rect = obstacle->rect();
        QPointF pos = obstacle->pos();
        
        ObstacleData data(
            pos.x() + rect.center().x(),
            pos.y() + rect.center().y(),
            rect.width(),
            rect.height(),
            obstacle->getAttenuation()
        );
        result.append(data);
    }
    
    return result;
}

void MapScene::setRadius(double radius) {
    m_radius = radius;
    if (m_radiusCircle) {
        m_radiusCircle->setRect(-m_radius, -m_radius, m_radius * 2, m_radius * 2);
    }
}

void MapScene::setShowGrid(bool show) {
    m_showGrid = show;
    invalidate(sceneRect(), QGraphicsScene::BackgroundLayer);
}

void MapScene::contextMenuEvent(QGraphicsSceneContextMenuEvent* event) {
    // Check if clicked on empty space
    QGraphicsItem* item = itemAt(event->scenePos(), QTransform());
    if (!item || item == m_radiusCircle) {
        QMenu menu;
        QAction* addObstacleAction = menu.addAction("Add Obstacle Here");
        QAction* clearObstaclesAction = menu.addAction("Clear All Obstacles");
        
        QAction* selected = menu.exec(event->screenPos());
        if (selected == addObstacleAction) {
            QPointF pos = event->scenePos();
            addObstacle(pos.x(), pos.y(), 80, 80, 20.0);
        } else if (selected == clearObstaclesAction) {
            clearObstacles();
        }
        event->accept();
    } else {
        QGraphicsScene::contextMenuEvent(event);
    }
}

void MapScene::drawBackground(QPainter* painter, const QRectF& rect) {
    QGraphicsScene::drawBackground(painter, rect);
    
    if (m_showGrid) {
        drawGrid(painter, rect);
    }
    
    // Draw axes
    painter->setPen(QPen(Qt::darkGray, 2));
    painter->drawLine(QLineF(rect.left(), 0, rect.right(), 0));  // X-axis
    painter->drawLine(QLineF(0, rect.top(), 0, rect.bottom()));  // Y-axis
    
    // Draw scale
    painter->setPen(Qt::black);
    QFont font = painter->font();
    font.setPointSize(10);
    painter->setFont(font);
    painter->drawText(10, -10, QString("%1m").arg(m_radius, 0, 'f', 0));
}

void MapScene::drawGrid(QPainter* painter, const QRectF& rect) {
    painter->setPen(QPen(QColor(200, 200, 200), 1, Qt::DotLine));
    
    // Grid spacing
    double gridSpacing = 100.0;
    
    // Vertical lines
    double left = qFloor(rect.left() / gridSpacing) * gridSpacing;
    double right = qCeil(rect.right() / gridSpacing) * gridSpacing;
    for (double x = left; x <= right; x += gridSpacing) {
        if (qAbs(x) > 1e-6) {  // Skip center axis
            painter->drawLine(QLineF(x, rect.top(), x, rect.bottom()));
        }
    }
    
    // Horizontal lines
    double top = qFloor(rect.top() / gridSpacing) * gridSpacing;
    double bottom = qCeil(rect.bottom() / gridSpacing) * gridSpacing;
    for (double y = top; y <= bottom; y += gridSpacing) {
        if (qAbs(y) > 1e-6) {  // Skip center axis
            painter->drawLine(QLineF(rect.left(), y, rect.right(), y));
        }
    }
}

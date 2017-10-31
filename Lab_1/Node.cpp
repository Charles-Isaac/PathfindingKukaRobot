#include "Node.h"
#include "NodeMap.h"
namespace Pathfinding
{

	class NodeMap;

	Node::Node()
	{
		position = new Point();
		vector = new Point();
	}

	Node::Node(float px, float py, float vx, float vy)
	{
		position = new Point(px, py);
		vector = new Point(vx, vy);
	}

	Node::Node(Point* p, Point* v)
	{
		position = p;
		vector = v;
		_Fcost = _Gcost = _Hcost = 0;
		_parent = nullptr;
		_visited = false;
	}


	Node::~Node()
	{
		//delete position;
		//delete vector;
		/*for (int i = 0; i < accessibleNodeVector.size(); i++)
		{
			delete accessibleNodeVector[i];
		}*/
		//accessibleNodeVector.clear();
	}
	float Node::getHCost() {
		return _Hcost;
	}
	float Node::getFCost() {
		return _Fcost;
	}
	float Node::getGCost() {
		return _Gcost;
	}
	Node* Node::getParent() {
		return _parent;
	}

	bool Node::isVisited() {
		return _visited;
	}
	void Node::setVisited(bool visited) {
		_visited = visited;
	}

	float Node::calculateCost(Node* end, Node::AccessibleNode* potentialNewParent) {
		float g, h, f;
		// means it's the first case, G = 0
		h = Point::distBetweenPoints(position, end->position);
		if (potentialNewParent == nullptr) {
			g = 0;
		}
		else {
			g = potentialNewParent->accessibleNode->getGCost() + Point::distBetweenPoints(potentialNewParent->accessibleNode->position, position);
		}
		f = g + h;
		if (_parent == nullptr || (f < _Fcost)) {
			if (potentialNewParent != nullptr) {
				_parent = potentialNewParent->accessibleNode;
			}
			_Hcost = h;
			_Fcost = f;
			_Gcost = g;
		}
		return _Fcost;
	}
	void Node::deleteNode()
	{
		accessibleNodeVector.clear();
	}

	void Node::deletePoint()
	{
		delete position;
		delete vector;
		for (int i = 0; i < accessibleNodeVector.size(); i++)
		{
			delete accessibleNodeVector[i];
		}
		accessibleNodeVector.clear();
	}

	void Node::createAccessibleNodeVect(NodeMap* map, float vehicleWidth)
	{
		if (accessibleNodeVector.size() == 0)
		{
			updateAccessibleNodeVect(map, vehicleWidth);
		}
	}
	void Node::updateAccessibleNodeVect(NodeMap* map, float vehicleWidth)
	{
		for (int i = 0; i < accessibleNodeVector.size(); i++)
		{
			delete accessibleNodeVector[i];
		}
		accessibleNodeVector.clear();
		float differentialWidth = vehicleWidth * 0.99f;
		for (int i = 0; i < map->polygonList.size(); i++)
		{

			for (int j = 0; j < map->polygonList[i]->nodeList.size(); j++)
			{
				bool intersect = false;
				for (int k = 0; k < map->polygonList.size() && !intersect; k++)
				{
					if (map->polygonList[k]->nodeList.size() <= 2)
					{
						continue;
					}
					Point* a = getPointAccordingToVehicleWidth(vehicleWidth);
					Point* b = map->polygonList[i]->nodeList[j]->getPointAccordingToVehicleWidth(vehicleWidth);
					Point* c = map->polygonList[k]->nodeList[map->polygonList[k]->nodeList.size() - 1]->getPointAccordingToVehicleWidth(differentialWidth);
					Point* d = map->polygonList[k]->nodeList[0]->getPointAccordingToVehicleWidth(differentialWidth);

					intersect = Point::IsIntersecting(a, b, c, d);
						
					delete a;
					delete b;
					delete c;
					delete d;

						
						
					for (int l = 1; l < map->polygonList[k]->nodeList.size() && !intersect; l++)
					{
						 a = getPointAccordingToVehicleWidth(vehicleWidth);
						 b = map->polygonList[i]->nodeList[j]->getPointAccordingToVehicleWidth(vehicleWidth);
						 c = map->polygonList[k]->nodeList[l - 1]->getPointAccordingToVehicleWidth(differentialWidth);
						 d = map->polygonList[k]->nodeList[l]->getPointAccordingToVehicleWidth(differentialWidth);

						intersect = Point::IsIntersecting(a, b, c, d);

						delete a;
						delete b;
						delete c;
						delete d;

					}
				}
				if (!intersect)
				{
					Point* a = map->polygonList[i]->nodeList[j]->getPointAccordingToVehicleWidth(vehicleWidth);
					Point* b = getPointAccordingToVehicleWidth(vehicleWidth);
					accessibleNodeVector.push_back(
						new AccessibleNode(map->polygonList[i]->nodeList[j],
							Point::distBetweenPoints(a, b)));
					delete a;
					delete b;
				}
			}
		}
	}

	Point* Node::getPointAccordingToVehicleWidth(float width)
	{
		return position->moveWithVect(vector, width);
	}


}
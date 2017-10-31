#pragma once
#include "stdafx.h"
#include "Point.h"
#include <vector>

namespace Pathfinding
{

	class NodeMap;
	class Node
	{
	public:
		struct AccessibleNode {
			AccessibleNode(Node *n, float dist)
			{
				accessibleNode = n;
				distanceFromSource = dist;
			}
			~AccessibleNode()
			{
				//delete accessibleNode;
			}
			Node* accessibleNode;
			float distanceFromSource;
		};
		Node();
		Node(float px, float py, float vx, float vy);
		Node(Point* p, Point* v);
		~Node();

		void deleteNode();
		void deletePoint();

		void createAccessibleNodeVect(NodeMap* map, float vehicleWidth);
		void updateAccessibleNodeVect(NodeMap* map, float vehicleWidth);

		Point* position;
		Point* vector;

		bool isVisited();
		void setVisited(bool);

		float getHCost();
		float getFCost();
		float getGCost();
		Node* getParent();
		float calculateCost(Node* end, Node::AccessibleNode* potentialNewParent = nullptr);
		
		

		std::vector<AccessibleNode*> accessibleNodeVector;
		//int accessibleNodeCount;

		Point* getPointAccordingToVehicleWidth(float width);

		

	private:

		float _Fcost, _Hcost, _Gcost;
		bool _visited;
		Node* _parent;
	};
}
#pragma once
#include "stdafx.h"
#include "Polygon.h"
#include "Node.h"
#include <vector>
namespace Pathfinding
{

	class NodeMap
	{
	public:
		NodeMap();
		void addPolygon(Polygon* polyg);
		void clearPolygonList();

		std::vector<Node*> getComputedNodeList(float vehicleWidth, Point* startPoint, Point* endPoint);
		void deleteMap();
		void deletePolygon();
		void deleteNode();
		void deletePoint();


		~NodeMap();
		std::vector<Polygon*> polygonList;
	private:

	};

}
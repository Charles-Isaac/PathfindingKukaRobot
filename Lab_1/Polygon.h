#pragma once
#include "stdafx.h"
#include <stdarg.h>
#include <initializer_list>
#include <vector>

#include "Node.h"

#define PI 3.14159265359

namespace Pathfinding
{
	class Polygon
	{
	public:
		Polygon(std::vector<Point*> pts);
		~Polygon();

		std::vector<Node*> nodeList;

		static std::vector<Point*> cleanupPointList(std::vector<Point*> pnodeList);

		static std::vector<Point*> removeNodes(std::vector<Point*> pnodeList, Point* newPoint, int startIndex, int endIndex);

		void deletePolygon();
		void deleteNode();
		void deletePoint();

	private:
		Point* getAngleVectorFromPoints(Point* previous, Point* current, Point* next);
	};

}
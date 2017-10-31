#include "Polygon.h"

namespace Pathfinding
{


	Polygon::Polygon(std::vector<Point*> pts)
	{
		pts = cleanupPointList(pts);
		if (pts.size() <= 2)
		{
			for (int i = 0; i < pts.size(); i++)
			{
				nodeList.push_back(new Node(pts[i], new Point(0,0)));
			}
			return;
		}
		for (int i = 0; i < pts.size(); i++)
		{
			Point* temp = getAngleVectorFromPoints(pts[i], pts[(i+1)% pts.size()], pts[(i + 2) % pts.size()]);
			nodeList.push_back(new Node(pts[(i + 1) % pts.size()], temp));
		}

	}
	
	
	Point* Polygon::getAngleVectorFromPoints(Point* previous, Point* current, Point* next)
	{
		/*previous = new Point(0, 0);
		current = new Point(1, 0);
		next = new Point(1.1f, 1);*/
		float vector1X = (previous->x - current->x);
		float vector1Y = (previous->y - current->y);
		float vector2X = (next->x - current->x);
		float vector2Y = (next->y - current->y);

		float normalizationFactorVector1 = sqrtf(vector1X*vector1X + vector1Y*vector1Y);
		float normalizationFactorVector2 = sqrtf(vector2X*vector2X + vector2Y*vector2Y);
		if (normalizationFactorVector1 == 0 || normalizationFactorVector2 == 0)
		{
			return new Point();
		}
		vector1X /= normalizationFactorVector1;
		vector1Y /= normalizationFactorVector1;
		vector2X /= normalizationFactorVector2;
		vector2Y /= normalizationFactorVector2;

		float dot = vector1X*vector2X + vector1Y*vector2Y;
		float det = vector1X*vector2Y - vector1Y*vector2X;
		float angle = atan2f(det, dot);
		angle += PI;
		if (angle ==-1)
		{
			int i = 0;
		}
		float angleAverage = atan2f(vector1Y + vector2Y, vector1X + vector2X) + PI * (angle <= PI && angle > 0);

		return new Point(cosf(angleAverage), sinf(angleAverage));
	}
	Polygon::~Polygon()
	{
		/*for (int i = 0; i < nodeList.size(); i++)
		{
			delete nodeList[i];
		}*/
		//nodeList.clear();

	}

	std::vector<Point*> Polygon::cleanupPointList(std::vector<Point*> pnodeList)
	{
		int size = pnodeList.size();
		if (pnodeList.size() >= 3)
		{
			for (int i = 0; i < pnodeList.size(); i++)
			{
				for (int j = i + 2; j < pnodeList.size() - (i == 0); j++)
				{
					if (Point::IsIntersecting(pnodeList[i], pnodeList[(i + 1) % size], pnodeList[(j) % size], pnodeList[(j + 1) % size]))
					{
						Point* tempPt = Point::getIntersectionPointOfTwoLines(pnodeList[i], pnodeList[(i + 1) % size], pnodeList[(j) % size], pnodeList[(j + 1) % size]);
						if ((float)(j - i) / (float)size > 0.5f)
						{
							pnodeList = removeNodes(pnodeList, tempPt, j + 1, (i + 1) % size);
						}
						else
						{
							pnodeList = removeNodes(pnodeList, tempPt, i + 1, (j + 1) % size);
						}
						size = pnodeList.size();
						if (j >= size)
						{
							j = size;
						}
						if (i >= size)
						{
							i = size;
						}
					}
				}
			}
		}
		return pnodeList;
	}
	std::vector<Point*> Polygon::removeNodes(std::vector<Point*> pnodeList,Point * newPoint, int startIndex, int endIndex)
	{
		if (startIndex>endIndex)
		{
			for (int i = startIndex; i < pnodeList.size(); i++)
			{
				delete pnodeList[i];
			}
			for (int i = 0; i < endIndex; i++)
			{
				delete pnodeList[i];
			}
			pnodeList.erase(pnodeList.begin() + startIndex, pnodeList.end());
			pnodeList.erase(pnodeList.begin(), pnodeList.begin() + endIndex);
			pnodeList.push_back(newPoint);
		}
		else
		{
			for (int i = startIndex; i < endIndex; i++)
			{
				delete pnodeList[i];
			}
			pnodeList.erase(pnodeList.begin() + startIndex, pnodeList.begin() + endIndex);
			pnodeList.insert(pnodeList.begin() + startIndex, newPoint);
		}
		return pnodeList;
	}
	void Polygon::deletePolygon()
	{
		nodeList.clear();
		delete this;
	}
	void Polygon::deleteNode()
	{
		for (int i = 0; i < nodeList.size(); i++)
		{
			nodeList[i]->deleteNode();
			delete nodeList[i];
		}
		nodeList.clear();
		delete this;
	}
	void Polygon::deletePoint()
	{
		for (int i = 0; i < nodeList.size(); i++)
		{
			nodeList[i]->deletePoint();
			delete nodeList[i];
		}
		nodeList.clear();
		delete this;
	}
}
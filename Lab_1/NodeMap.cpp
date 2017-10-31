#include "NodeMap.h"

namespace Pathfinding
{

	NodeMap::NodeMap()
	{
	}
	void NodeMap::addPolygon(Polygon* polyg)
	{
		polygonList.push_back(polyg);
	}
	void NodeMap::clearPolygonList()
	{
		polygonList.clear();
	}

	std::vector<Node*> NodeMap::getComputedNodeList(float vehicleWidth, Point* startPoint, Point* endPoint)
	{
		std::vector<Node*> tempVect;
		NodeMap nm;
		nm.addPolygon(new Polygon({ startPoint}));
		nm.addPolygon(new Polygon({ endPoint }));
		for (int i = 0; i < polygonList.size(); i++)
		{
			std::vector<Point*> subTempVect;

			for (int j = 0; j < polygonList[i]->nodeList.size(); j++)
			{
				subTempVect.push_back(polygonList[i]->nodeList[j]->getPointAccordingToVehicleWidth(vehicleWidth*0.97f));
				//subTempVect.push_back(polygonList[i]->nodeList[j]->getPointAccordingToVehicleWidth(vehicleWidth*0.03f));
			}
			nm.addPolygon(new Polygon(subTempVect));
			
		}

		for (int i = 0; i < nm.polygonList.size(); i++)
		{
			//std::vector<Point*> subTempVect;

			for (int j = 0; j < nm.polygonList[i]->nodeList.size(); j++)
			{
				nm.polygonList[i]->nodeList[j]->createAccessibleNodeVect(&nm, vehicleWidth*0.03f);
				tempVect.push_back(nm.polygonList[i]->nodeList[j]); //should return this instead
			}
		}
		nm.deletePolygon();
		return tempVect;
	}
	void NodeMap::deleteMap()
	{
		polygonList.clear();
	}
	void NodeMap::deletePolygon()
	{
		for (int i = 0; i < polygonList.size(); i++)
		{
			polygonList[i]->deletePolygon();
		}
		polygonList.clear();
	}
	void NodeMap::deleteNode()
	{
		for (int i = 0; i < polygonList.size(); i++)
		{
			polygonList[i]->deleteNode();
		}
		polygonList.clear();
	}
	void NodeMap::deletePoint()
	{
		for (int i = 0; i < polygonList.size(); i++)
		{
			polygonList[i]->deletePoint();
		}
		polygonList.clear();
	}
	NodeMap::~NodeMap()
	{
		/*for (int i = 0; i < polygonList.size(); i++)
		{
			delete polygonList[i];
		}*/
		//polygonList.clear();
	}
}
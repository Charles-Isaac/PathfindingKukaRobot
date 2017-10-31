#pragma once
#pragma once
#include "stdafx.h"
#include <math.h>
namespace Pathfinding
{
	class Point
	{
	public:
		Point();
		Point(float px, float py);
		~Point();


		Point* moveWithVect(Point* vector, float multiplier);


		float x;
		float y;

		static bool IsIntersecting(Point* VecteurA1, Point* VecteurA2, Point* VecteurB1, Point* VecteurB2); //Retourne un booléen indiquant si la ligne tracé entre VecteurA1 et VecteurA2 intersectionne la ligne tracé entre VecteurB1 et VecteurB2 
		static Point* getIntersectionPointOfTwoLines(Point* VecteurA1, Point* VecteurA2, Point* VecteurB1, Point* VecteurB2);
		static float distBetweenPoints(Point* a, Point* b);

	private:

	};

}
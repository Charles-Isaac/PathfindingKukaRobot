#include "Point.h"

namespace Pathfinding
{
	Point::Point()
	{
		x = 0;
		y = 0;
	}
	Point::Point(float px, float py)
	{
		x = px;
		y = py;
	}
	Point* Point::moveWithVect(Point* vector, float multiplier)
	{
		return new Point(x+vector->x*multiplier,y+vector->y*multiplier);
	}



	bool Point::IsIntersecting(Point* VecteurA1, Point* VecteurA2, Point* VecteurB1, Point* VecteurB2)
	{
		float Denominator = ((VecteurA2->x - VecteurA1->x) * (VecteurB2->y - VecteurB1->y)) - ((VecteurA2->y - VecteurA1->y) * (VecteurB2->x - VecteurB1->x));
		float Numerator1 = ((VecteurA1->y - VecteurB1->y) * (VecteurB2->x - VecteurB1->x)) - ((VecteurA1->x - VecteurB1->x) * (VecteurB2->y - VecteurB1->y));
		float Numerator2 = ((VecteurA1->y - VecteurB1->y) * (VecteurA2->x - VecteurA1->x)) - ((VecteurA1->x - VecteurB1->x) * (VecteurA2->y - VecteurA1->y));

		if (Denominator == 0) return Numerator1 == 0 && Numerator2 == 0;

		float R = Numerator1 / Denominator;
		float S = Numerator2 / Denominator;

		return (R >= 0 && R <= 1) && (S >= 0 && S <= 1);
	}
	Point * Point::getIntersectionPointOfTwoLines(Point * VecteurA1, Point * VecteurA2, Point * VecteurB1, Point * VecteurB2)
	{

		float dx12 = VecteurA2->x - VecteurA1->x;
		float dy12 = VecteurA2->y - VecteurA1->y;
		float dx34 = VecteurB2->x - VecteurB1->x;
		float dy34 = VecteurB2->y - VecteurB1->y;


		float denominator = (dy12 * dx34 - dx12 * dy34);
		if (denominator == 0)
		{
			return VecteurA1;
		}


		float t1 =
			((VecteurA1->x - VecteurB1->x) * dy34 + (VecteurB1->y - VecteurA1->y) * dx34)
			/ denominator;
		float t2 =
			((VecteurB1->x - VecteurA1->x) * dy12 + (VecteurA1->y - VecteurB1->y) * dx12)
			/ -denominator;



		return new Point(VecteurA1->x + dx12 * t1, VecteurA1->y + dy12 * t1);

		/*float denominator = (VecteurB2->y - VecteurB1->y) * (VecteurA2->x - VecteurA1->x) - (VecteurB2->x - VecteurB1->x) * (VecteurA2->y - VecteurA1->y);
		float vecteurA = (VecteurB2->x - VecteurB1->x) * (VecteurA1->y - VecteurB1->y) - (VecteurB2->y - VecteurB1->y) * (VecteurA1->x - VecteurB1->x);
		//float vecteurB = (VecteurA2->x - VecteurA1->x) * (VecteurA1->y - VecteurB1->y) - (VecteurA2->y - VecteurA1->y) * (VecteurA1->x - VecteurB1->x);
		float vecteurB = (VecteurA1->x - VecteurB1->x) * (VecteurB2->y - VecteurB1->y) + (VecteurB1->y - VecteurA1->y) * (VecteurB2->x - VecteurB1->x);

		
		return new Point(
			VecteurA1->x + vecteurA / denominator * (VecteurA2->x - VecteurA1->x),
			VecteurA1->y + vecteurA / denominator * (VecteurA2->y - VecteurA1->y)
			);*/
	}
	float Point::distBetweenPoints(Point* a, Point* b)
	{
		float deltaX = (a->x - b->x);
		float deltaY = (a->y - b->y);

		return sqrt(deltaX*deltaX+deltaY*deltaY);
	}
	Point::~Point()
	{

	}
}
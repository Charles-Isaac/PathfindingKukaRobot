#pragma once
#include "stdafx.h"
#include "NodeMap.h"
#include <vector>
#include <opencv2\core\core.hpp>
#include <set>
#include <unordered_set>
#include <iostream>
#include <queue>
#include <vector>

namespace Pathfinding {

	class Path {
	public:

		Path(std::vector<Node*> points, float width);

		void DrawOnMat(cv::Mat m);

		std::vector<Node*> getPath();

	private:

		void _compute();

		Node* _begin;
		Node* _end;

		std::vector<Node*> _path;
		bool compareNode(Node* a, Node* b);

		NodeMap _map;
		float _width;


	};

}
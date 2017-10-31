#include "Node.h"
#include "NodeMap.h"
#include "Path.h"


namespace Pathfinding
{

	Path::Path(std::vector<Node*> nm, float width) {
		_begin = nm[0]; // The begin and end points are added first...
		_end = nm[1];
		_width = width;
		// _map = [Call the method to generate a new map with begin, end and width]
		_compute();
	}

	std::vector<Node*> Path::getPath() {
		return _path;
	}

	struct comparator {
		bool operator () (Node::AccessibleNode *const a, Node::AccessibleNode *const b) {
			return a->accessibleNode->getFCost() < b->accessibleNode->getFCost();
		}
	};

	void Path::_compute() {

		// Comparateur de nodes, pour le sort ci-bas
		int iteration = 0;
		std::vector<Node::AccessibleNode*> opened;

		if (_begin->position->x == _end->position->x) {
			return;
		}

		// to free
		Node::AccessibleNode* beginNode = new Node::AccessibleNode(_begin, 0);
		opened.push_back(beginNode);
		_begin->calculateCost(_end);
		Node::AccessibleNode* currentNode = beginNode;
		std::vector<Node::AccessibleNode*> neighbors;
		bool didSomething = true;

		// Main pathfinding loop :D
		while (currentNode->accessibleNode != _end && !opened.empty() && currentNode != nullptr) {
			// get the best new node
			currentNode = *(opened.begin());
			// If there's a node...
			if (currentNode != nullptr) {
				// get its neighbors (nodes that can be accessed without crossing a shape)
				neighbors = currentNode->accessibleNode->accessibleNodeVector;
				// For each of those neighbors...
				for (std::vector<Node::AccessibleNode*>::iterator it = neighbors.begin(); it != neighbors.end(); ++it) {
					// Make sure that we're not reusing the same shape and that this shape is not visited or in the open list
					if ((*it)->accessibleNode != currentNode->accessibleNode && !(*it)->accessibleNode->isVisited() && 
						std::find_if(opened.begin(), opened.end(), [it](const Node::AccessibleNode* n) {
							return n->accessibleNode->position->x == (*it)->accessibleNode->position->x && n->accessibleNode->position->y == (*it)->accessibleNode->position->y;
						}) == opened.end()) {
						// if it's not, add it to the opened list
						(*it)->accessibleNode->calculateCost(_end, currentNode);
						opened.push_back((*it));
					}
				}
				// remove the current node from the current list and mark it as visited (add it to the closed list, which doesn't actually exist)
				opened.erase(std::find(opened.begin(), opened.end(), currentNode));
				currentNode->accessibleNode->setVisited(true);
				// sort them again, cuz why not... if only std::set worked properly :(
				std::sort(opened.begin(), opened.end(), comparator());
				iteration++;
				if (iteration > 1000) {
					int iwhgowrg = 0;
				}
			}
		}
		// Did we ever reach the end?
		if (currentNode != nullptr && _end->getParent() != nullptr) {
			// If we did, make the path (from the end to the start)
			Node* nodePath = currentNode->accessibleNode;
			while (nodePath != nullptr) {
				_path.push_back(nodePath);
				nodePath = nodePath->getParent();
			}
			// Reverse it
			std::reverse(_path.begin(), _path.end());
		}
		else {
			std::cout << "No path could be found" << std::endl;
		}
		delete beginNode;

	}

}
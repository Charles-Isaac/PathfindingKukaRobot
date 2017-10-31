// TP3 - SII
// Charles-Isaac Côté & Samuel Goulet
// 2017-10-24 ish

#include "stdafx.h"

// OpenCV
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2\imgproc\imgproc.hpp>
#include "AxisCommunication.h"

#include "Polygon.h"
#include "NodeMap.h"
#include "Path.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <thread>

#include <iostream>

extern "C" cudaError_t RunDilate(unsigned char* imageIn, unsigned char* imageOut, int width, int height, unsigned char* mask, int maskSize,
								int minHue, int maxHue, int minSat, int maxSat, int minVal, int maxVal, int closeSize);

using namespace std;

CamData CamInf;
Axis axis("10.128.3.4", "etudiant", "gty970");
float PanStep = 5.0, TiltStep = 5.0;
int ZoomStep = 10, FocusStep = 500, BrightnessStep = 500;

cv::Mat phil(int rad);
cv::VideoCapture vc;

Pathfinding::Point startPoint = Pathfinding::Point();
Pathfinding::Point endPoint = Pathfinding::Point();
std::vector<Pathfinding::Point*> testPoly;

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		startPoint = Pathfinding::Point(x, y);
	}
	else if (event == cv::EVENT_RBUTTONDOWN)
	{
		endPoint = Pathfinding::Point(x, y);
	}
	else if (event == cv::EVENT_MBUTTONDOWN)
	{
		testPoly.push_back(new Pathfinding::Point(x, y));
	}
}

cv::Mat getImage() {

	cv::Mat img1, img2;
	bool failed = true;
	while (failed) {
		failed = false;
		axis.AbsolutePan(-161.934402);
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		axis.AbsoluteTilt(-66.159401);
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		vc.read(img1);
		vc.read(img1);
		vc.read(img1);
		if (!vc.read(img1))
		{
			// Unable to retrieve frame from video stream
			std::cout << "Cannhot [sic] read image on Axis cam..." << std::endl << "Hit a key to try again." << std::endl;
			cv::waitKey(0);
			failed = true;
			vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
		}

		axis.AbsolutePan(16.4405994);
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		axis.AbsoluteTilt(-70.701599);
		std::this_thread::sleep_for(std::chrono::milliseconds(1500));
		vc.read(img2);
		vc.read(img2);
		vc.read(img2);
		if (!vc.read(img2))
		{
			// Unable to retrieve frame from video stream
			std::cout << "Cannhot [sic] read image on Axis cam..." << std::endl << "Hit a key to try again." << std::endl;
			cv::waitKey(0);
			failed = true;
			vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
		}
	}

	cv::Mat imgConcat;
	cv::flip(img2, img2, -1);
	cv::vconcat(img1, img2, imgConcat);
	cv::namedWindow("cat", CV_WINDOW_NORMAL);
	cv::imshow("cat", imgConcat);
	cv::waitKey(100);

	return imgConcat;
}

int main()
{
	float vehicleWidth;
	cv::Mat img;
	vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");

	cv::Mat imgOrig = getImage();

	cv::namedWindow("bin", cv::WINDOW_NORMAL);

	cv::namedWindow("Axis PTZ", cv::WINDOW_NORMAL);
	cv::setMouseCallback("Axis PTZ", MouseCallBackFunc, NULL);

	Pathfinding::NodeMap nm;


	vector<vector<cv::Point>> contours0;
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	cv::Mat hsv;
	cv::Mat bin;
	cv::Mat philippe = phil(125);
	cudaSetDevice(0);
	while (true) {
		vehicleWidth = 20; // 150.f;
		/*if (!vc.read(img))
		{
			// Unable to retrieve frame from video stream
			std::cout << "Cannhot [sic] read image on Axis cam..." << std::endl << "Hit a key to try again." << std::endl;
			cv::waitKey(0);
			vc.open("http://etudiant:gty970@10.128.3.4/axis-cgi/mjpg/video.cgi");
		}*/
		img = imgOrig.clone();
		//resize(img, img, cv::Size(1920 / 2, 1080 / 2));
		cv::GaussianBlur(img, img, cv::Size(5, 5), 2);
		cvtColor(img, hsv, CV_BGR2HSV);
		bin = cv::Mat::zeros(cv::Size(hsv.cols, hsv.rows), CV_8UC3);
		//RunDilate(img.data, bin.data, img.cols, img.rows, philippe.data, philippe.cols, 50, 82, 0, 255, 128, 255, 15);
		cvtColor(bin, bin, CV_BGR2GRAY);
		for (int i = 0; i < hsv.rows * hsv.cols; i++) {
			unsigned char h = hsv.data[i * 3], s = hsv.data[i * 3 + 1], v = hsv.data[i * 3 + 2];
			bin.data[i] = (h < 50 || h > 82 || v < 128) ? 255 : 0;
		}
		morphologyEx(bin, bin, cv::MORPH_OPEN, cv::Mat::ones(7, 7, CV_8UC1));
		//morphologyEx(bin, bin, cv::MORPH_CLOSE, cv::Mat::ones(7, 7, CV_8UC1));
		cv::dilate(bin, bin, philippe);

		findContours(bin, contours0, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_TC89_KCOS);
		contours.resize(contours0.size());
		std::cout << "New iteration! :D" << std::endl << std::endl;
		nm.clearPolygonList();
		if (testPoly.size() >= 3)
		{
			std::vector<Pathfinding::Point*> temp;
			for (int i = 0; i < testPoly.size(); i++)
			{
				temp.push_back(new Pathfinding::Point(testPoly[i]->x, testPoly[i]->y));
			}
			nm.addPolygon(new Pathfinding::Polygon(temp));
		}
		for (size_t k = 0; k < contours0.size(); k++) {
			approxPolyDP(cv::Mat(contours0[k]), contours[k], 6, false);
			std::vector<Pathfinding::Point*> points;
			for (int i = contours[k].size() - 1; i >= 0; i--)
			{
				points.push_back(new Pathfinding::Point(contours[k][i].x, contours[k][i].y));
			}
			nm.addPolygon(new Pathfinding::Polygon(points));
		}
		drawContours(img, contours, -1, cv::Scalar(128, 255, 255), 3, cv::LINE_AA, hierarchy, 3);

		std::vector<Pathfinding::Node*> points = nm.getComputedNodeList(vehicleWidth, new Pathfinding::Point(startPoint.x, startPoint.y), new Pathfinding::Point(endPoint.x, endPoint.y));


		for (int i = 0; i < points.size(); i++)
		{
			Pathfinding::Point* p = points[i]->position;
			cv::circle(img, cv::Point(p->x, p->y), 5, cv::Scalar(255.f, 0.f, 63.f), CV_FILLED);
		}

		

		std::vector<Pathfinding::Node::AccessibleNode*> neighbors;
		for (int i = 2; i < points.size(); i++) {
			Pathfinding::Node* p = points.at(i);
			cv::circle(img, cv::Point(p->position->x, p->position->y), 5, cv::Scalar(255.f, 0.f, 0.f));
			neighbors = p->accessibleNodeVector;
			for (int j = 0; j < neighbors.size(); j++) {
				Pathfinding::Node::AccessibleNode* neigh = neighbors.at(j);
				cv::line(img, cv::Point(p->position->x, p->position->y),
					cv::Point(neigh->accessibleNode->position->x, neigh->accessibleNode->position->y), cv::Scalar(127.f, 127.f, 127.f), 1);
			}
		}

		Pathfinding::Path* path = new Pathfinding::Path(points, vehicleWidth);
		std::vector<Pathfinding::Node*> pathNodes = path->getPath();

		if (pathNodes.size() > 0) {
			cv::Point beforeNode = cv::Point(pathNodes[0]->position->x, pathNodes[0]->position->y);
			for (int i = 1; i < pathNodes.size(); i++) {
				std::cout << pathNodes[i]->position->x << ", " << pathNodes[i]->position->y << std::endl;
				cv::Point nowNode = cv::Point(pathNodes[i]->position->x, pathNodes[i]->position->y);
				cv::line(img, beforeNode, nowNode, cv::Scalar(255.f, 255.f, 0.f), 5);
				beforeNode = nowNode;
			}
		}
		for (int i = 0; i < nm.polygonList.size(); i++)
		{
			std::vector<Pathfinding::Point*> subTempVect;

			for (int j = 0; j < nm.polygonList[i]->nodeList.size(); j++)
			{
				Pathfinding::Point* temp = nm.polygonList[i]->nodeList[j]->getPointAccordingToVehicleWidth(vehicleWidth*0.99f);
				Pathfinding::Point* temp2 = nm.polygonList[i]->nodeList[(j + 1) % nm.polygonList[i]->nodeList.size()]->getPointAccordingToVehicleWidth(vehicleWidth*0.99f);

				cv::circle(img, cv::Point(temp->x, temp->y), 3, cv::Scalar(234.f, 125.f, 56.f), CV_FILLED);

				cv::line(img, cv::Point(temp->x, temp->y), cv::Point(temp2->x, temp2->y), cv::Scalar(0, 0, 255.f), 1);
			}

		}
		cv::imshow("bin", bin);
		cv::waitKey(24);
		cv::imshow("Axis PTZ", img);
		cv::waitKey(24);

	}

	return 0;
}

cv::Mat phil(int rad) {
	cv::Mat phil(rad * 2 + 1, rad * 2 + 1, CV_8UC1);
	for (int x = -rad; x <= rad; x++) {
		for (int y = -rad; y <= rad; y++) {
			phil.data[(y + rad) * phil.cols + (x + rad)] = (-abs(x) + rad) + (-abs(y) + rad) < rad * 9 / 2; // (x * x + y * y) < (rad * rad);
		}
	}
	return phil;
}
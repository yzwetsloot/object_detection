#include <iostream>
#include <string>
#include <vector>
#include <zbar.h>
#include <opencv2/opencv.hpp>
#include "ZBarDetector.h"

using namespace std;
using namespace zbar;

void ZBarDetector::detectAndDecodeMulti(Mat& im, vector<string>& data, Mat& bbox)
{
	ImageScanner scanner;

	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 0); // disable all
	scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1); // enable QR Code

	Mat imGray;
	cvtColor(im, imGray, COLOR_BGR2GRAY);

	Image image(im.cols, im.rows, "Y800", (uchar*)imGray.data, im.cols * im.rows);

	int n = scanner.scan(image);

	for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol)
	{
		string text = symbol->get_data();
		data.push_back(text);

		float points[8];
		for (int i = 0; i < symbol->get_location_size(); i++)
		{
			points[i * 2] = static_cast<float>(symbol->get_location_x(i));
			points[i * 2 + 1] = static_cast<float>(symbol->get_location_y(i));
		}

		Mat row = Mat(1, 8, CV_32F, points);
		bbox.push_back(row);
	}
}

void ZBarDetector::display(Mat& im, vector<decodedObject>& decodedObjects)
{
	for (int i = 0; i < decodedObjects.size(); i++)
	{
		vector<Point> points = decodedObjects[i].location;
		vector<Point> hull;

		if (points.size() > 4)
			convexHull(points, hull);
		else
			hull = points;

		int n = hull.size();

		for (int j = 0; j < n; j++)
		{
			line(im, hull[j], hull[(j + 1) % n], Scalar(255, 0, 0), 3);
		}
	}

	imshow("Results", im);
	waitKey(0);
}

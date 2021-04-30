#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <zbar.h>
#include "QRDetector.h"

using namespace std;
using namespace cv;
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

string ZBarDetector::getName()
{
	return name;
}

void OpenCVDetector::detectAndDecodeMulti(Mat& im, vector<string>& data, Mat& bbox)
{
	QRCodeDetector detector = QRCodeDetector::QRCodeDetector();
	detector.detectAndDecodeMulti(im, data, bbox);
}

string OpenCVDetector::getName()
{
	return name;
}

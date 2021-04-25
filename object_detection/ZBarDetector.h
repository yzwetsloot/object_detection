#pragma once

#include <opencv2/core/types.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

typedef struct
{
	string type;
	string data;
	vector<Point> location;
} decodedObject;

class ZBarDetector {
public:
	void detectAndDecodeMulti(Mat& im, vector<string>& data, Mat& bbox);
	void display(Mat& im, vector<decodedObject>& decodedObjects);
};
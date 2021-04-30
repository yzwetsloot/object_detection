#pragma once

#include <opencv2/core/types.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

class QRDetector {
public:
	virtual void detectAndDecodeMulti(Mat& im, vector<string>& data, Mat& bbox) = 0;
	virtual string getName() = 0;
};

class ZBarDetector : public QRDetector {
public:
	void detectAndDecodeMulti(Mat& im, vector<string>& data, Mat& bbox);
	string getName();
private:
	string name = "ZBarDetector";
};

class OpenCVDetector : public QRDetector {
public:
	void detectAndDecodeMulti(Mat& im, vector<string>& data, Mat& bbox);
	string getName();
private:
	string name = "OpenCVDetector";
};

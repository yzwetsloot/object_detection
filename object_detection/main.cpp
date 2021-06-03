#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <time.h>
#include <cstdlib>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "QRDetector.h"

using namespace cv;
using namespace std;

string keys =
"{ help h      |       | Print help message. }"
"{ debug d     | false | Includes window with marked targets if true. }"
"{ camera c    | 0     | Camera device number. }"
"{ duration    | 3600  | Maximum program duration in seconds before program exit. }"
"{ targets     |       | Target data to be matched against QR-decoded text. Separate individual strings by a comma. }"
"{ detector    | 0     | Choose one of QR detector & decoder libaries: "
"0: ZBar (by default), "
"1: OpenCV }";

constexpr double FONT_SIZE = 0.3;

vector<string> parseTargetNames(string targets, char delimiter=',')
{
	vector<string> targetNames;

	size_t pos = 0;
	string token;
	while ((pos = targets.find(delimiter)) != string::npos) {
		token = targets.substr(0, pos);
		targetNames.push_back(token);
		targets.erase(0, pos + 1);
	}

	targetNames.push_back(targets);

	return targetNames;
}

void display(Mat& im, Mat& bbox, vector<string> data)
{
	int n = bbox.rows;
	for (int i = 0; i < n; i++)
	{
		// insert a rectangle around identified target using corner coordinates
		rectangle(im, Point_<float>(bbox.at<float>(i, 0), bbox.at<float>(i, 1)),
			Point_<float>(bbox.at<float>(i, 4), bbox.at<float>(i, 5)),
			Scalar(255, 0, 0), 3);

		// insert decoded text on top of bounding box
		putText(im, data[i], Point_<float>(bbox.at<float>(i, 0), bbox.at<float>(i, 1) - 10), 
			FONT_HERSHEY_SIMPLEX, FONT_SIZE, Scalar(255, 0, 0));
	}
}

QRDetector* getDetector(int id)
{
	if (id)
		return new OpenCVDetector();
	else
		return new ZBarDetector();
}

int main(int argc, char* argv[]) 
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("Lunar Zebro navigation - QR Code detection v1.0.0");

	if (parser.has("help"))
	{
		parser.printMessage();
		return EXIT_SUCCESS;
	}

	const bool DEBUG = parser.get<bool>("debug");
	cout << DEBUG << endl;

	const int MAX_DURATION = parser.get<int>("duration");

	const string targets = parser.get<string>("targets");
	const vector<string> targetNames = parseTargetNames(targets);
	
	cout << "Found " << targetNames.size() << " targets: ";
	for (auto targetName : targetNames) {
		cout << targetName << ' ';
	}
	cout << endl;

	cout << "Start object detection\n";

	Mat frame;

	VideoCapture cap;

	// open camera feed
	int deviceID = parser.get<int>("camera");
	int apiID = CAP_ANY;
	cap.open(deviceID, apiID);

	if (!cap.isOpened()) {
		cerr << "Unable to open camera\n";
		return EXIT_FAILURE;
	}

	cout << "Start grabbing frames\n";
	cout << "Press any key to terminate\n";

	// initialize variables for average FPS calculation
	int frameCounter = 0;
	int tick = 0;
	int fps = 0;
	time_t timeBegin = time(0);

	QRDetector* detector = getDetector(parser.get<int>("detector"));

	auto maxDuration = chrono::seconds(MAX_DURATION);

	chrono::time_point<chrono::system_clock> endTime;
	endTime = chrono::system_clock::now() + maxDuration;

	for (;;)
	{
		// if max duration has passed, exit from loop
		if (chrono::system_clock::now() >= endTime) {
			cout << "Maximum duration has passed. Exiting program..." << endl;
			break;
		}

		cap.read(frame);
		if (frame.empty()) {
			cerr << "Blank frame grabbed\n";
			break;
		}

		// calculate average FPS for camera video
		frameCounter++;
		time_t timeNow = time(0) - timeBegin;

		if (timeNow - tick >= 1)
		{
			tick++;
			fps = frameCounter;
			frameCounter = 0;
		}

		Mat bbox; 
		vector<string> data;

		detector->detectAndDecodeMulti(frame, data, bbox);
		if (!data.empty()) {
			for (string text : data)
				printf("[%s] Decoded data: %s\n", detector->getName().c_str(), text.c_str());
		}

		if (DEBUG) {
			// display bounding box around target
			if (!data.empty()) {
				display(frame, bbox, data);
			}

			// insert average FPS counter into image matrix
			putText(frame, format("Average FPS=%d", fps), Point(10, 10), FONT_HERSHEY_SIMPLEX, FONT_SIZE, Scalar(100, 255, 0));

			// show image frame and wait for key press
			namedWindow("Live", WINDOW_NORMAL);
			imshow("Live", frame);

			if (waitKey(5) >= 0)
				break;
		}
	}

	if (DEBUG) destroyAllWindows();

	return EXIT_SUCCESS;
}
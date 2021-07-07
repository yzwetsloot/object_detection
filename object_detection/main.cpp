#pragma warning(disable : 4996)

#include <algorithm>
#include <iostream>
#include <vector>
#include <set>
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
#include <iomanip>

using namespace cv;
using namespace std;

string keys =
"{ help h      |       | Print help message. }"
"{ debug d     | false | Includes window with marked targets if true. }"
"{ camera c    | 0     | Camera device number. }"
"{ targets     |       | Target data to be matched against QR-decoded text. Separate individual strings by a comma. }"
"{ width       | 1920  | Width component camera video resolution. }"
"{ height      | 1080  | Height component camera video resolution. }"
"{ detector    | 0     | Choose one of QR detector & decoder libaries: "
"0: ZBar (by default), "
"1: OpenCV }";

constexpr double FONT_SIZE = 0.3;
constexpr int MAX_COLOR_VALUE = 255;
constexpr double VARIANCE = 0.05; // % distance from middle vertical line

enum class State {
	detect,
	docking,
	sleep,
};

State EXEC_STATE = State::detect;

vector<string> parseStringArgument(string targets, char delimiter=',')
{
	vector<string> stringArguments;

	size_t pos = 0;
	string token;
	while ((pos = targets.find(delimiter)) != string::npos) {
		token = targets.substr(0, pos);
		stringArguments.push_back(token);
		targets.erase(0, pos + 1);
	}

	stringArguments.push_back(targets);

	return stringArguments;
}

set<string> convertVectorToSet(vector<string> elements)
{
	set<string> uniqueElements(elements.begin(), elements.end());
	return uniqueElements;
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

 string getCurrentTimeString()
{
	using namespace std::chrono;
	auto t = system_clock::now();
	auto ms = duration_cast<milliseconds>(t.time_since_epoch()) % 1000;
	auto timer = system_clock::to_time_t(t);
	auto tm = *localtime(&timer);

	ostringstream oss;

	oss << put_time(&tm, "%H:%M:%S");
	oss << '.' << setfill('0') << setw(3) << ms.count();

	return oss.str();
}

 bool checkTargetName(set<string>& targetNames, string text)
 {
	 if (targetNames.find(text) != targetNames.end()) {
		 targetNames.erase(text);
		 cout << "Decoded value corresponds to one of the targets" << endl;
		 return true;
	 }
	 return false;
 }

 Mat applyGrayscale(Mat& mat, const Scalar& low, const Scalar& high)
 {
	 // colored object detection
	 Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); // TODO: determine why?
	 Mat isolation;

	 inRange(mat, low, high, isolation);
	 bitwise_not(isolation, isolation);

	 erode(isolation, isolation, Mat());
	 dilate(isolation, isolation, element);

	 return isolation;
 }

 struct contour {
	 double area;
	 size_t index;
 };

 bool sortContour(contour c1, contour c2)
 {
	 return c1.area < c2.area;
 }

 vector<Point> getContour(Mat& mat)
 {
	 threshold(mat, mat, 128, 255, THRESH_BINARY); // TODO: determine why these values?

	 vector<vector<Point>> contours;
	 findContours(mat, contours, RETR_LIST, CHAIN_APPROX_SIMPLE); // TODO: use CHAIN_APPROX_NONE?

	 vector<contour> contourAreas;

	 if (contours.size() > 1) {
		 for (size_t idx = 0; idx < contours.size(); idx++) {
			 contourAreas.push_back(contour{ contourArea(contours[idx]), idx });
		 }

		 sort(contourAreas.begin(), contourAreas.end(), sortContour);
	 }
	 else if (contours.size() <= 1) return vector<Point>{}; // if single element or empty, there cannot be a second largest contour

	 size_t idxSecondBiggest = contourAreas[contourAreas.size() - 2].index; // get second largest
	 return contours[idxSecondBiggest];
 }

 Point2f getContourCenter(vector<Point> contour)
 {
	 auto mu = moments(contour);
	 auto mc = Point2f(static_cast<float>(mu.m10 / (mu.m00 + 1e-5)),
		static_cast<float>(mu.m01 / (mu.m00 + 1e-5)));

	 return mc;
 }

 void applyContourVisual(Mat& mat, vector<Point> contour, Point2f center)
 {
	 if (contour.empty()) return;
	 vector<vector<Point>> contours;
	 contours.push_back(contour);
	 drawContours(mat, contours, -1, Scalar(255,0,0));
	 circle(mat, center, 4, Scalar(255,0,0), -1);
 }

int main(int argc, char* argv[]) 
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("Lunar Zebro navigation - QR Code detection v1.0.0\nAuthor: Y. Zwetsloot\n");

	if (parser.has("help"))
	{
		parser.printMessage();
		return EXIT_SUCCESS;
	}

	cout << "Lunar Zebro navigation - QR Code detection v1.0.0\nAuthor: Y. Zwetsloot\n" << endl;

	const bool DEBUG = parser.get<bool>("debug");
	if (DEBUG) cout << "Running in debug mode" << endl;
	else cout << "Running in production mode" << endl;

	const string targets = parser.get<string>("targets");
	set<string> targetNames = convertVectorToSet(parseStringArgument(targets));

	if (!targetNames.size()) {
		cerr << "Unable to read in targets\n";
		return EXIT_FAILURE;
	}

	cout << "Found " << targetNames.size() << " target(s): ";
	for (auto targetName : targetNames) {
		cout << targetName << ' ';
	}
	cout << endl;

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

	const int fw = parser.get<int>("width");
	const int fh = parser.get<int>("height");

	cap.set(CAP_PROP_FRAME_WIDTH, fw);
	cap.set(CAP_PROP_FRAME_HEIGHT, fh);

	const double frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	const double frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
	const double middleCoordinateWidth = frameWidth / 2.0;

	cout << "\nCamera is open\nStart grabbing frames @ " << getCurrentTimeString() << " @ " << cap.get(CAP_PROP_FPS) << " frames/second" << 
		" with " << frameWidth << "x" << frameHeight << endl << endl;

	QRDetector* detector = getDetector(parser.get<int>("detector"));

	// define color range parameters
	int rh = 255, rl = 100, gh = 255, gl = 0, bh = 70, bl = 0;
	const string targetWindowName = "Target";

	for (;;)
	{
		if (EXEC_STATE == State::sleep) continue;

		cap.read(frame);
		if (frame.empty()) {
			cerr << "Blank frame grabbed\n";
			break;
		}

		if (EXEC_STATE == State::docking) {
			// apply grayscale
			Mat grayscaleImage = applyGrayscale(frame, Scalar(bl, gl, rl), Scalar(bh, gh, rh));

			// contours
			vector<Point> mainContour = getContour(grayscaleImage);
			Point2f center = getContourCenter(mainContour);

			if (center.x >= middleCoordinateWidth - VARIANCE * frameWidth &&
				center.x <= middleCoordinateWidth + VARIANCE * frameWidth) {
				cout << "Point is in the middle: " << center.x << ", " << center.y << endl;
			}

			if (DEBUG) {
				// show contours
				Mat contourImage(grayscaleImage.size(), CV_8UC3, Scalar(0, 0, 0));
				applyContourVisual(frame, mainContour, center);
				imshow("Flag", frame);
			}
		} else if (EXEC_STATE == State::detect) {
			Mat bbox;
			vector<string> data;

			detector->detectAndDecodeMulti(frame, data, bbox);
			if (!data.empty()) {

				// TODO: determine need for timestamp and printing
				for (string text : data) {
					cout << getCurrentTimeString() << " - ";
					printf("[%s] Decoded data: %s\n", detector->getName().c_str(), text.c_str());
					checkTargetName(targetNames, text);

					if (targetNames.empty()) {
						cout << "\nAll targets found. Exiting program..." << endl;
						goto endLoop;
					}
				}
			}
			if (DEBUG) {
				// display bounding box around target
				if (!data.empty()) {
					display(frame, bbox, data);
				}

				// show image frame and wait for key press
				namedWindow(targetWindowName, WINDOW_NORMAL);

				createTrackbar("rh", targetWindowName, &rh, MAX_COLOR_VALUE);
				createTrackbar("rl", targetWindowName, &rl, MAX_COLOR_VALUE);
				createTrackbar("gh", targetWindowName, &gh, MAX_COLOR_VALUE);
				createTrackbar("gl", targetWindowName, &gl, MAX_COLOR_VALUE);
				createTrackbar("bh", targetWindowName, &bh, MAX_COLOR_VALUE);
				createTrackbar("bl", targetWindowName, &bl, MAX_COLOR_VALUE);

				imshow(targetWindowName, frame);
			}
		}

		if (DEBUG) {
			if (waitKey(5) >= 0)
				break;
		}
	}

endLoop:
	if (DEBUG) destroyAllWindows();

	return EXIT_SUCCESS;
}

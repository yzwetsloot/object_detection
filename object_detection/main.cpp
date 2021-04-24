#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	cout << "Start object detection source\n";

	Mat frame;

	VideoCapture cap;

	int deviceID = 0;
	int apiID = CAP_ANY;
	cap.open(deviceID, apiID);

	if (!cap.isOpened()) {
		cerr << "Unable to open camera\n";
		return -1;
	}

	cout << "Start grabbing frames\n";
	cout << "Press any key to terminate\n";

	int frameCounter = 0;
	int tick = 0;
	int fps = 0;
	time_t timeBegin = time(0);

	for (;;)
	{
		cap.read(frame);
		if (frame.empty()) {
			cerr << "Blank frame grabbed\n";
			break;
		}

		frameCounter++;
		time_t timeNow = time(0) - timeBegin;

		if (timeNow - tick >= 1)
		{
			tick++;
			fps = frameCounter;
			frameCounter = 0;
		}

		putText(frame, format("Average FPS=%d", fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255));

		imshow("Live", frame);
		if (waitKey(5) >= 0)
			break;
	}

	return 0;
}
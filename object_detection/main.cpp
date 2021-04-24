#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

constexpr double FONT_SIZE = 0.3;

void display(Mat& im, Mat& bbox)
{
	int n = bbox.rows;
	for (int i = 0; i < n; i++)
	{
		rectangle(im, Point_<float>(bbox.at<float>(i, 0), bbox.at<float>(i, 1)),
			Point_<float>(bbox.at<float>(i, 4), bbox.at<float>(i, 5)),
			Scalar(255, 0, 0), 3);
	}
}

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

	QRCodeDetector qrDecoder = QRCodeDetector::QRCodeDetector();

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

		Mat bbox;

		std::string data = qrDecoder.detectAndDecode(frame, bbox);
		if (data.length() > 0)
		{
			cout << "Decoded data: " << data << endl;
			display(frame, bbox);
		}

		putText(frame, format("Average FPS=%d", fps), Point(10, 10), FONT_HERSHEY_SIMPLEX, FONT_SIZE, Scalar(100, 255, 0));

		namedWindow("Live", WINDOW_NORMAL);
		imshow("Live", frame);
		if (waitKey(5) >= 0)
			break;
	}

	destroyAllWindows();

	return 0;
}
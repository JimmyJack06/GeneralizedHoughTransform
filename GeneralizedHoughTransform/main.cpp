#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

int main()
{
	auto image = cv::imread("letters.png", cv::IMREAD_COLOR);
	cv::Mat intermediate;
	cv::Canny(image, intermediate, 80, 90);

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Canny", intermediate);
	cv::waitKey(0);
	return 0;
}
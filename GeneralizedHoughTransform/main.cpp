#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "GeneralizedHoughTransform.hpp"

using namespace std;
using namespace cv;

int main() 
{
	Mat tpl = imread("template_elephant.png");
	imshow("template", tpl);
	Mat src = imread("animals2.jpg");
	imshow("source", src);

	GeneralHoughTransform ght(tpl);

	/*Size s(src.size().width / 4, src.size().height / 4);
	resize(src, src, s, 0, 0, CV_INTER_AREA);
*/
	//imshow("debug - image", src);

	ght.accumulate(src);
	waitKey(0);
	return 0;
}


#pragma once
#include <cmath>
#include <opencv2\core\core.hpp>

const double PI = 4.0*std::atan(1.0);

cv::Mat gradientY(const cv::Mat &src);
cv::Mat gradientX(const cv::Mat &src);
cv::Mat gradientDirection(const cv::Mat& src);
void invertIntensities(const cv::Mat& src, cv::Mat& dst);
float gradientDirection(const cv::Mat& src, int x, int y);
float fastsqrt(float val);
int rad2SliceIndex(double angle, int nSlices);
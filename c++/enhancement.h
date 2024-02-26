#ifndef ENHANCEMENT_H
#define ENHANCEMENT_H

namespace cv{
    class Mat;
}

double degreesToRadians(double degrees);

cv::Mat equalizationAndCC(const cv::Mat& img, double percentage = 0.0005, int iterations = 100);

float calculateThreshold(const cv::Mat& value);

float calculateEuclidean(const double s_1, const double s_2, double h_1, double h_2);

void processChannel(cv::Mat& channel);

cv::Mat thresholdOrange(const cv::Mat& img, std::vector<float> referenceColor);

#endif
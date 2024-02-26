#ifndef SEGMENTATION_H
#define SEGMENTATION_H

namespace cv {
    class Mat;
}

// Declare functions using cv::Point and cv::Vec4i
void getContours(const cv::Mat& img, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i>& hierarchy);

void drawContours(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const std::vector<cv::Vec4i>& hierarchy);

#endif

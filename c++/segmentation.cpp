#include <math.h>
#include <opencv2/opencv.hpp>
#include "segmentation.h"

using namespace cv;

void getContours(const Mat& img, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i>& hierarchy){
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
}

void drawContours(Mat& img, const std::vector<std::vector<Point>>& contours, const std::vector<cv::Vec4i>& hierarchy){
    // draw contours straight onto img
    for (int i = 0; i < contours.size(); i++){
        if (hierarchy[i][2] == -1){
            // draw bounding box
            Rect rect = boundingRect(contours[i]);
            rectangle(img, rect, Scalar(0, 255, 0), 2);
        }
    }
}
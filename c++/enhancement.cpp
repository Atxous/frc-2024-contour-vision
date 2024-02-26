#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "enhancement.h"

using namespace cv;

const float HUE_SCALER = (360.0f / 255.0f);
const float SAT_SCALER = (100.0f / 255.0f);
const float VAL_SCALER = (100.0f / 255.0f);
const double PI = 3.14159265358979323846;

double degreesToRadians(double degrees){
    return degrees * PI / 180.0;
}

Mat equalizationAndCC(const cv::Mat& img, double percentage, int iterations){
    Mat c_i;
    equalizeHist(img, c_i);
    c_i.convertTo(c_i, CV_32F, 1.0 / 255.0);
    Mat a_i = c_i.clone();
    
    Mat kernel = (Mat_<float>(3, 3) << 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0);
    Mat avg;
    for(int i = 0; i < iterations; i++){
        filter2D(a_i, avg, -1, kernel);
        a_i = c_i * percentage + avg * (1 - percentage);
    }
    divide(c_i, (2 * a_i + 1e-6), c_i);
    c_i *= 255;

    // clip the values to 0-255
    c_i = cv::min(c_i, 255.0);
    c_i.convertTo(c_i, CV_8U);

    return c_i;
}

float calculateThreshold(const Mat& value){
    return exp(-1 * static_cast<float>((sum(value)[0] * VAL_SCALER)/(value.rows * value.cols * 256.0))) * 30;
}

float calculateEuclidean(const double s_1, const double s_2, double h_1, double h_2){
    float distance;
    h_1 = degreesToRadians(h_1);
    h_2 = degreesToRadians(h_2);
    distance = static_cast<float>(pow(s_2 * cos(h_2) - s_1 * cos(h_1), 2) + pow(s_2 * sin(h_2) - s_1 * sin(h_1), 2));
    return sqrt(distance);
}

void processChannel(Mat& channel){
    channel = equalizationAndCC(channel, 0.0005, 1000);
}

Mat thresholdOrange(const Mat& img, std::vector<float> referenceColor){
    Mat mask, hsv;
    cvtColor(img, mask, COLOR_BGR2GRAY);
    std::vector<Mat> channels;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    split(hsv, channels);
    float threshold = calculateThreshold(channels[2]);
    channels[0] = channels[0] * HUE_SCALER;
    channels[1] = channels[1] * SAT_SCALER;
    referenceColor[0] *= HUE_SCALER;
    referenceColor[1] *= SAT_SCALER;

    parallel_for_(Range(0, img.rows), [&](const Range& range){
        for (int i = range.start; i < range.end; i++){
            for (int j = 0; j < img.cols; j++){
                if(calculateEuclidean(channels[1].at<uchar>(i, j), referenceColor[1], channels[0].at<uchar>(i, j), referenceColor[0]) > threshold){
                    mask.at<uchar>(i, j) = 0;
                }
                else{
                    mask.at<uchar>(i, j) = 255;
                }
            }
        }
    });
    return mask;
}
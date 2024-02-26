#include <windows.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <enhancement.h>
#include <segmentation.h>

using namespace cv;

const bool PARALLEL = true;
const std::vector<float> BASE_ORANGE = {11.0, 180.0, 200.0};
/*
int main(){
    Mat img = imread("video_rings/0.jpg");
    
    // resize img to 128x128
    resize(img, img, Size(128, 128));
    
    std::vector<Mat> channels;
    split(img, channels);
    std::vector<std::thread> threads;
    // time the function
    auto start = std::chrono::high_resolution_clock::now();
    if (PARALLEL){
        for (int i = 0; i < 3; i++){
            threads.emplace_back(std::thread(processChannel, std::ref(channels[i])));
        }
        for (auto& t : threads){
            t.join();
        }
    }
    else{
        channels[0] = equalizationAndCC(channels[0]);
        channels[1] = equalizationAndCC(channels[1]);
        channels[2] = equalizationAndCC(channels[2]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    Mat img_cc;

    merge(channels, img_cc);
    resize(img_cc, img_cc, Size(1024, 1024));
    resize(img, img, Size(1024, 1024));
    Mat adjusted_mask = thresholdOrange(img_cc, BASE_ORANGE);
    morphologyEx(adjusted_mask, adjusted_mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(20, 20)));
    morphologyEx(adjusted_mask, adjusted_mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(25, 25)));

    imshow("Color Constancy", adjusted_mask);
    imshow("Original", img_cc);
    waitKey(0);
    return 0;
}
*/
void giveMask(const std::string& path, const std::vector<float>& referenceColor = BASE_ORANGE){
    Mat img = imread(path, IMREAD_COLOR);
    // resize img to 128x128
    resize(img, img, Size(64, 64));
    std::vector<Mat> channels;
    split(img, channels);
    std::vector<std::thread> threads;
    // time the function
    auto start = std::chrono::high_resolution_clock::now();
    if (PARALLEL){
        for (int i = 0; i < 3; i++){
            threads.emplace_back(std::thread(processChannel, std::ref(channels[i])));
        }
        for (auto& t : threads){
            t.join();
        }
    }
    else{
        for (int i = 0; i < 3; i++){
            processChannel(channels[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    Mat img_cc;

    merge(channels, img_cc);
    resize(img_cc, img_cc, Size(1024, 1024));
    resize(img, img, Size(1024, 1024));
    Mat adjusted_mask = thresholdOrange(img_cc, referenceColor);
    // Mat mask = thresholdOrange(img);
    // morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));
    morphologyEx(adjusted_mask, adjusted_mask, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(20, 20)));
    // morphologyEx(mask, mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));
    morphologyEx(adjusted_mask, adjusted_mask, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(25, 25)));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    getContours(adjusted_mask, contours, hierarchy);
    drawContours(img_cc, contours, hierarchy);

    // imshow("Original", mask);
    imshow("Color Constancy", adjusted_mask);
    imshow("Image", img_cc);
    if((waitKey(1) & 0xFF) == 27){
        destroyAllWindows();
    }
}

std::vector<std::string> loadFilesInDirectory(const std::string& directory) {
    std::vector<std::string> files;

    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile((directory + "\\*").c_str(), &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.push_back(directory + "\\" + findFileData.cFileName);
            }
        } while (FindNextFile(hFind, &findFileData) != 0);
        FindClose(hFind);
    }

    return files;
}

int main(){
    std::vector<std::string> files = loadFilesInDirectory("video_rings");

    for(const auto& file : files){
        giveMask(file);
    }
}
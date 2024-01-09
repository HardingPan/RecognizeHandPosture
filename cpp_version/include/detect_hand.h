#ifndef HAND_DETECT_H
#define HAND_DETECT_H

#include <opencv2/opencv.hpp>
#include <stack>

class HandDetect {
public:
    HandDetect();
    cv::Mat crop();
    cv::Mat skinEllipse(const cv::Mat& image);
    cv::Mat getHand();
    std::vector<cv::Point> extractLargestContour(const cv::Mat& inputImage);

private:
    cv::Mat image;
};

#endif
#ifndef HAND_DETECT_H
#define HAND_DETECT_H

#include <opencv2/opencv.hpp>

class HandDetect {
public:
    HandDetect();
    cv::Mat crop();
    cv::Mat skinEllipse(const cv::Mat& image);
    cv::Mat skinThreshold();
    cv::Mat getHand();

private:
    cv::Mat image;
};

#endif // HAND_DETECT_H
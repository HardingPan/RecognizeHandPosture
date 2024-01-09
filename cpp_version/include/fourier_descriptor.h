#ifndef FOURIER_DESCRIPTOR_H
#define FOURIER_DESCRIPTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class FourierDescriptors {
public:
    FourierDescriptors();
    std::vector<double> calculate(const cv::Mat& image);
    std::vector<double> GetAndTruncateDescriptors(const cv::Mat& image, int length);
private:
    cv::Mat image;
};

#endif
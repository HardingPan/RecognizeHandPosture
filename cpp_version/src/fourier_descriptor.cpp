#include "fourier_descriptor.h"

FourierDescriptors::FourierDescriptors(){}

std::vector<double> FourierDescriptors::calculate(const cv::Mat& image) {
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(image.rows);
    int n = cv::getOptimalDFTSize(image.cols);
    cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, \
                    n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);

    cv::dft(complexImage, complexImage);

    cv::split(complexImage, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);

    cv::Mat magnitudeImage = planes[0];

    // 检查并确保矩阵是连续的
    if (!magnitudeImage.isContinuous()) {
        magnitudeImage = magnitudeImage.clone();
    }
    // 将描述子展平为一维数组
    std::vector<double> flatDescriptor;
    magnitudeImage.reshape(0, 1).copyTo(flatDescriptor);
    // 归一化描述子
    cv::Scalar mean, stdDev;
    cv::meanStdDev(flatDescriptor, mean, stdDev);
    double meanValue = mean.val[0];
    double stdDevValue = stdDev.val[0];
    for (double& value : flatDescriptor) {
        value = (value - meanValue) / stdDevValue;
    }

    return flatDescriptor;
}

std::vector<double> FourierDescriptors::GetAndTruncateDescriptors(const cv::Mat& image, \
                                                                int length) {
    std::vector<double> descriptors = calculate(image);
    if (length >= descriptors.size()) {
        return descriptors; // 不需要截断，返回原始描述子
    } else {
        return std::vector<double>(descriptors.begin(), descriptors.begin() + length);
    }
}
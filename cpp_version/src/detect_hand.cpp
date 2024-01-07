#include "detect_hand.h"

HandDetect::HandDetect() {
    // 构造函数，可以在这里进行一些初始化操作
}

cv::Mat HandDetect::crop() {
    // 实现 crop 函数
}

cv::Mat HandDetect::skinEllipse(const cv::Mat& image) {
    // 创建椭圆mask
    cv::Mat skinCrCbHist = cv::Mat::zeros(256, 256, CV_8U);
    cv::ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 25), 43, 0, 360, cv::Scalar(255), -1);
    // 转换至YCrCb空间
    cv::Mat YCrCb;
    cv::cvtColor(image, YCrCb, cv::COLOR_BGR2YCrCb);
    // 拆分Y, Cr, Cb
    std::vector<cv::Mat> channels;
    cv::split(YCrCb, channels);
    cv::Mat Cr = channels[1];
    cv::Mat Cb = channels[2];
    // 创建皮肤掩膜
    cv::Mat skin = cv::Mat::zeros(Cr.size(), CV_8U);
    for (int i = 0; i < Cr.rows; ++i) {
        for (int j = 0; j < Cr.cols; ++j) {
            if (skinCrCbHist.at<uchar>(Cr.at<uchar>(i, j), Cb.at<uchar>(i, j)) > 0) {
                skin.at<uchar>(i, j) = 255; // 若不在椭圆区间中
    }}}
    cv::Mat result;
    cv::bitwise_and(image, image, result, skin);
    return result;
}


cv::Mat HandDetect::getHand() {
    // 实现 getHand 函数
}
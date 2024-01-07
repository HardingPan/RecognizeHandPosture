#include "include/detect_hand.h"

int main(int argc, char** argv){
    cv::Mat image;
    image = cv::imread("/home/harding/RecognizeHandPosture/1.jpg");
    HandDetect detector;
    cv::Mat image_out = detector.skinEllipse(image);
    std::cout << "Image width: " << image_out.cols << std::endl;
    std::cout << "Image height: " << image_out.rows << std::endl;
    std::cout << "Image channels: " << image_out.channels() << std::endl;
    cv::imshow("image", image_out);
    cv::waitKey(0);
}
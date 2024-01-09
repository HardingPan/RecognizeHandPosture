#include "include/detect_hand.h"

void InitText(){
    std::cout << "\n 欢迎使用基于图像处理和机器学习的手势识别系统" << std::endl;
    std::cout << "作者: Harding" << std::endl;
    std::cout << "项目仓库: https://github.com/HardingPan/RecognizeHandPosture.git" << std::endl;
}

int main(int argc, char** argv){
    InitText();
    cv::Mat image;
    image = cv::imread("/home/harding/RecognizeHandPosture/1.jpg");
    HandDetect detector;
    cv::Mat image_out = detector.skinEllipse(image);
    cv::imshow("image", image_out);
    cv::waitKey(0);
}
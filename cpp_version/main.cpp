#include "include/detect_hand.h"

void InitText(){
    std::cout << "\n 欢迎使用基于图像处理和机器学习的手势识别系统" << std::endl;
    std::cout << "作者: Harding" << std::endl;
    std::cout << "项目仓库: https://github.com/HardingPan/RecognizeHandPosture.git" << std::endl;
}

int main(int argc, char** argv){
    InitText();
    cv::VideoCapture cap(-1);
    HandDetect detector;
    cv::Mat image;
    // 设置字体
    std::string text;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    cv::Scalar fontColor(0, 255, 0); // 白色
    int thickness = 1;
    int lineType = cv::LINE_AA;
    cv::Point textOrg(10, 30);

    if (!cap.isOpened()) {
		return -1;
	}

    bool run_flag = true;
    while (run_flag){
        clock_t startTimeStamp, finishTimeStamp;
        cap >> image;
        startTimeStamp = clock();
        cv::Mat image_out = detector.skinEllipse(image);
        cv::cvtColor(image_out, image_out, cv::COLOR_GRAY2BGR);
        // 确保两个图像具有相同的行数并拼接
        if (image.rows != image_out.rows) {
            std::cerr << "Error: Images must have the same number of rows." << std::endl;
            return -1;
        }
        cv::Mat combinedImage;
        cv::hconcat(image, image_out, combinedImage);
        
        finishTimeStamp = clock();
        text = "FPS: " + std::to_string((double)CLOCKS_PER_SEC / (finishTimeStamp - startTimeStamp));
		cv::putText(combinedImage, text, textOrg, fontFace, \
                    fontScale, fontColor, thickness, lineType);
        cv::imshow("Recognize Hand Posture Window", combinedImage);
        if (cv::waitKey(1) >= 0)
			run_flag = true;
    }
}
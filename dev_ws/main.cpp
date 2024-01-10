#include "include/detect_hand.h"
#include "include/fourier_descriptor.h"
#include "include/knn_classify.h"

std::string dataPath = "/home/harding/RecognizeHandPosture/dataset/";
std::string datasetPath = "/home/harding/RecognizeHandPosture/dataset.yml";
int frameWidth = 640;
int frameHeight = 480;

void InitText(){
    std::cout << "\n 欢迎使用基于图像处理和机器学习的手势识别系统" << std::endl;
    std::cout << "作者: Harding" << std::endl;
    std::cout << "项目仓库: https://github.com/HardingPan/RecognizeHandPosture.git" << std::endl;
}

int main(int argc, char** argv){
    InitText();
    // 摄像头初始化
    cv::VideoCapture cap(-1);
    if (!cap.isOpened()) {
        std::cout << "can't find the camera successfully" << std::endl;
		return -1;
	}
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameWidth);
    // 字体初始化
    std::string text;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    cv::Scalar fontColor(0, 255, 0); // 白色
    int thickness = 1;
    int lineType = cv::LINE_AA;
    cv::Point textOrg(10, 30);

    HandDetect detector;
    FourierDescriptors fourier_descriptor;
    // 初始化KNN分类器
    KnnClassifier knnClassifier(3);
    
    knnClassifier.Train(dataPath);
    knnClassifier.SaveDataset("dataset.yml");
    knnClassifier.LoadDataset("dataset.yml");
    
    std::cout << "\nAll initialized successfully\n" << std::endl;

    cv::Mat image;
    bool run_flag = true;
    while (run_flag){
        clock_t startTimeStamp, finishTimeStamp;
        cap >> image;
        startTimeStamp = clock();
        
        cv::Mat imageHand = detector.getHand(image);
        // std::vector<double> descriptors = fourier_descriptor.calculate(imageHand);
        int predictedLabel = knnClassifier.Classify(imageHand);
        std::cout << "Predicted Label: " << predictedLabel << std::endl;
        
        cv::cvtColor(imageHand, imageHand, cv::COLOR_GRAY2BGR);
        // 确保两个图像具有相同的行数并拼接
        if (image.rows != imageHand.rows) {
            std::cerr << "Error: Images must have the same number of rows." << std::endl;
            return -1;
        }
        cv::Mat combinedImage;
        cv::hconcat(image, imageHand, combinedImage);
        
        finishTimeStamp = clock();
        text = "FPS: " + std::to_string((double)CLOCKS_PER_SEC / (finishTimeStamp - startTimeStamp));
		cv::putText(combinedImage, text, textOrg, fontFace, \
                    fontScale, fontColor, thickness, lineType);
        cv::imshow("Recognize Hand Posture Window", combinedImage);
        if (cv::waitKey(1) >= 0)
			run_flag = true;
    }
    // cv::Mat image;
    // bool run_flag = true;
    // while (run_flag){
    //     clock_t startTimeStamp, finishTimeStamp;
    //     cap >> image;
    //     startTimeStamp = clock();
        
    //     cv::Mat imageHand = detector.getHand(image);
    //     // std::vector<double> descriptors = fourier_descriptor.calculate(imageHand);
    //     std::vector<double> descriptors = fourier_descriptor.GetAndTruncateDescriptors(imageHand, 100);

    //     cv::cvtColor(imageHand, imageHand, cv::COLOR_GRAY2BGR);
        
    //     finishTimeStamp = clock();

    //     cv::imshow("Recognize Hand Posture Window", imageHand);
    //     if (cv::waitKey(1) >= 0)
	// 		run_flag = true;
    // }
}
#include "include/detect_hand.h"
#include "include/fourier_descriptor.h"
#include "include/knn_classify.h"

std::string dataPath;
std::string datasetPath = "/home/harding/RecognizeHandPosture/dataset.yml";
int frameWidth = 640;
int frameHeight = 480;

void InitText(){
    std::cout << "\n 欢迎使用基于图像处理和机器学习的手势识别系统" << std::endl;
    std::cout << "作者: Harding" << std::endl;
    std::cout << "项目仓库: https://github.com/HardingPan/RecognizeHandPosture.git" << std::endl;
}

std::string missionLabel(const int intpredictedLabel, const int missionClass) {
    switch (missionClass) {
        case 1:
            return std::to_string(intpredictedLabel);
        case 2:
            if (intpredictedLabel >= 1 && intpredictedLabel <= 26) {
                char alphabetChar = 'A' + intpredictedLabel - 1;
                return std::string(1, alphabetChar);
            } else {
                return "Invalid input for missionClass 2";
            }
        case 3:
            if (intpredictedLabel == 1) {
                return "up";
            } else if (intpredictedLabel == 2) {
                return "left";
            } else {
                return "right";
            }
        case 4:
            return std::to_string(intpredictedLabel);
        default:
            return "Invalid missionClass";
    }
}

constexpr size_t hash(const char* str, size_t h = 0) {
    return (str[h] == '\0') ? h : (hash(str, h + 1) ^ (size_t)str[h] << 7);
}

int main(int argc, char** argv){
    if (argc > 1) {
        std::string command = argv[1];
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
        std::string textFPS, textResult;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        cv::Scalar fontColor(0, 255, 0); // 白色
        int thickness = 1;
        int lineType = cv::LINE_AA;
        cv::Point fpsOrg(10, 30);
        cv::Point resOrg(10, 60);

        HandDetect detector;
        FourierDescriptors fourier_descriptor;
        cv::Mat image;
        bool run_flag = true;
        int missionClass = 0;

        switch (hash(command.c_str()))
        {
        case hash("c"):
            while (run_flag){
                cap >> image;
                cv::Mat imageHand = detector.getHand(image);
                cv::imshow("Recognize Hand Posture Window", imageHand);
                if (cv::waitKey(1) >= 0)
                    run_flag = true;
            }
            return -1;
            break;
        case hash("n"):
            dataPath = "/home/harding/RecognizeHandPosture/dataset/NumberLock";
            missionClass = 1;
            break;
        case hash("s"):
            dataPath = "/home/harding/RecognizeHandPosture/dataset/SignLanguage";
            missionClass = 2;
            break;
        case hash("d"):
            dataPath = "/home/harding/RecognizeHandPosture/dataset/Direction";
            missionClass = 3;
            break;
        case hash("o"):
            dataPath = "/home/harding/RecognizeHandPosture/dataset/OneOrZero";
            missionClass = 4;
            break;
        default:
            std::cout << "Wrong arg!" << std::endl;
            return -1;
        }
        // 初始化KNN分类器
        KnnClassifier knnClassifier(3);
        knnClassifier.Train(dataPath);
        knnClassifier.SaveDataset("dataset.yml");
        knnClassifier.LoadDataset("dataset.yml");

        std::string predictedLabel = "?";
        int unknowNum = 0;
        while (run_flag){
            clock_t startTimeStamp, finishTimeStamp;
            cap >> image;
            startTimeStamp = clock();
            
            cv::Mat imageHand = detector.getHand(image);
            // std::vector<double> descriptors = fourier_descriptor.calculate(imageHand);
            int intpredictedLabel = knnClassifier.Classify(imageHand);
            if (intpredictedLabel != 0){
                predictedLabel = missionLabel(intpredictedLabel, missionClass);
            } else {
                unknowNum = (unknowNum + 1) % 20;
                if (unknowNum == 0){
                    predictedLabel = "?";
                }
            }
            textResult = "Result: " + predictedLabel;
            // std::cout << "Predicted Label: " << predictedLabel << std::endl;
            
            cv::cvtColor(imageHand, imageHand, cv::COLOR_GRAY2BGR);
            // 确保两个图像具有相同的行数并拼接
            if (image.rows != imageHand.rows) {
                std::cerr << "Error: Images must have the same number of rows." << std::endl;
                return -1;
            }
            cv::Mat combinedImage;
            cv::hconcat(image, imageHand, combinedImage);
            
            finishTimeStamp = clock();
            textFPS = "FPS: " + std::to_string((double)CLOCKS_PER_SEC / (finishTimeStamp - startTimeStamp));
            cv::putText(combinedImage, textFPS, fpsOrg, fontFace, \
                        fontScale, fontColor, thickness, lineType);
            cv::putText(combinedImage, textResult, resOrg, fontFace, \
                        fontScale, fontColor, thickness, lineType);
            cv::imshow("Recognize Hand Posture Window", combinedImage);
            if (cv::waitKey(1) >= 0)
                run_flag = true;
        }
    } else {
        std::cerr << "please input argc" << std::endl;
    }
}


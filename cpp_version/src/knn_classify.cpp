#include "knn_classify.h"

KnnClassifier::KnnClassifier(int k) : k(k) {
    knn = cv::ml::KNearest::create();
}

void KnnClassifier::Train(const std::string& folderPath) {
    // 使用图像而不是描述子作为数据集, 在使用前遍历图像计算描述子
    std::vector<cv::String> imagePaths;
    cv::glob(folderPath, imagePaths);
    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cout << imagePath << " can't read this image";
            continue;
        }
        // 文件名就是图像的标签
        size_t startPos = imagePath.find_last_of('/') + 1;
        size_t endPos = imagePath.find_last_of('-');
        int label = std::stoi(imagePath.substr(startPos, endPos - startPos));
        std::vector<double> descriptor = \
            fourierCalculator.GetAndTruncateDescriptors(image, descriptorLength);
        dataset.push_back(descriptor);
        labels.push_back(label);
    }
    cv::Mat dataMat(dataset.size(), dataset[0].size(), CV_32F);
    cv::Mat labelsMat(this->labels.size(), 1, CV_32S);
    // 将计算得到的傅里叶描述子和相应的标签添加到数据集中
    for (size_t i = 0; i < dataset.size(); ++i) { // 填充数据矩阵
        float* dataPtr = dataMat.ptr<float>(static_cast<int>(i)); // 获取数据矩阵的行指针
        for (size_t j = 0; j < dataset[i].size(); ++j) {
            dataPtr[j] = static_cast<float>(dataset[i][j]);
        }
    }
    int* labelsPtr = labelsMat.ptr<int>(0); // 获取标签矩阵的指针
    for (size_t i = 0; i < this->labels.size(); ++i) { // 填充标签矩阵
        labelsPtr[i] = this->labels[i];
    }
    // 使用OpenCV的 train 方法, 对KNN分类器进行训练, 传递训练数据和标签
    knn->train(dataMat, cv::ml::ROW_SAMPLE, labelsMat);
}

void KnnClassifier::SaveDataset(const std::string& filename) {
    // cv::FileStorage 类用于将数据存储到文件中
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "Dataset" << "[";
    for (size_t i = 0; i < dataset.size(); ++i) {
        // 对于每个数据集条目, 使用 "{:" 表示一个组, 包含 "Descriptor" 和 "Label" 两个键
        fs << "{:" << "Descriptor" << dataset[i] << "Label" << labels[i] << "}";
    }
    fs << "]";
    fs.release();
}

void KnnClassifier::LoadDataset(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::FileNode datasetNode = fs["Dataset"];
    // 遍历数据集节点, 对于每个数据集条目从中提取傅里叶描述子和标签。
    for (cv::FileNodeIterator it = datasetNode.begin(); it != datasetNode.end(); ++it) {
        std::vector<double> descriptor;
        int label;
        (*it)["Descriptor"] >> descriptor;
        (*it)["Label"] >> label;
        dataset.push_back(descriptor);
        labels.push_back(label);
    }
    fs.release();
}

int KnnClassifier::Classify(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Error: Unable to read image " << std::endl;
        return -1;
    }
    std::vector<double> inputDescriptor = \
        fourierCalculator.GetAndTruncateDescriptors(image, descriptorLength);
    cv::Mat query(1, inputDescriptor.size(), CV_32F);
    for (size_t i = 0; i < inputDescriptor.size(); ++i) {
        query.at<float>(i) = static_cast<float>(inputDescriptor[i]);
    }
    cv::Mat response, dist;
    knn->findNearest(query, k, response, dist);

    return static_cast<int>(response.at<float>(0));
}

// int main() {
//     KnnClassifier knnClassifier(3);

//     std::string trainingFolderPath = "path/to/training/folder";
//     knnClassifier.Train(trainingFolderPath);

//     knnClassifier.SaveDataset("dataset.yml");
//     knnClassifier.LoadDataset("dataset.yml");

//     cv::Mat testImage;
//     int predictedLabel = knnClassifier.Classify(testImage);

//     std::cout << "Predicted Label: " << predictedLabel << std::endl;

//     return 0;
// }
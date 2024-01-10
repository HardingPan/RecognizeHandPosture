#ifndef KNN_CLASSIFY_H
#define KNN_CLASSIFY_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#include "fourier_descriptor.h"

class KnnClassifier {
public:
    KnnClassifier(int k);
    void Train(const std::string& folderPath);
    void SaveDataset(const std::string& filename);
    void LoadDataset(const std::string& filename);
    int Classify(const cv::Mat& image);

private:
    int k;
    cv::Ptr<cv::ml::KNearest> knn;
    std::vector<std::vector<double>> dataset;
    std::vector<int> labels;
    int descriptorLength;
    FourierDescriptors fourierCalculator;
};

#endif
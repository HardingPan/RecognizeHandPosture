#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// 计算傅里叶描述子
vector<double> calculateFourierDescriptors(const Mat& image) {
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols);
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImage;
    merge(planes, 2, complexImage);

    dft(complexImage, complexImage);

    split(complexImage, planes);
    magnitude(planes[0], planes[1], planes[0]);

    Mat magnitudeImage = planes[0];

    // 取中心区域作为傅里叶描述子
    int cx = magnitudeImage.cols / 2;
    int cy = magnitudeImage.rows / 2;

    Rect roi(cx - 50, cy - 50, 100, 100); // 以中心为中心取100x100的区域
    Mat descriptor = magnitudeImage(roi);

    // 将描述子展平为一维数组
    vector<double> flatDescriptor;
    descriptor.reshape(0, 1).copyTo(flatDescriptor);

    return flatDescriptor;
}

int main() {
    // 读取单通道图像
    Mat image = imread("your_image_path", 0);

    if (image.empty()) {
        cerr << "Error: Unable to load the image!" << endl;
        return -1;
    }

    // 计算傅里叶描述子
    vector<double> fourierDescriptors = calculateFourierDescriptors(image);

    // 打印傅里叶描述子
    cout << "Fourier Descriptors: ";
    for (double value : fourierDescriptors) {
        cout << value << " ";
    }
    cout << endl;

    return 0;
}
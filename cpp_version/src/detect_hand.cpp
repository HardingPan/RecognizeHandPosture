#include "detect_hand.h"

// 分水岭算法
class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMarkers(const cv::Mat& markerImage) {
		// 转换为整数图像
		markerImage.convertTo(markers, CV_32S);
	}
	cv::Mat process(const cv::Mat &image) {
		// 适用分水岭
		cv::watershed(image, markers);
		return markers;
	}
	// 以图像的形式返回结果
	cv::Mat getSegmentation() {
		cv::Mat tmp;
		// 标签高于255的所有段
		// 将被赋值为255
		markers.convertTo(tmp, CV_8U);
		return tmp;
	}
	// 以图像的形式返回分水岭
	cv::Mat getWatersheds() {
		cv::Mat tmp;
		markers.convertTo(tmp, CV_8U, 255, 255);
		return tmp;
	}
};

// 八邻接种子算法
void Seed_Filling(const cv::Mat& binImg, cv::Mat& labelImg, int& labelNum, int(&ymin)[20], int(&ymax)[20], int(&xmin)[20], int(&xmax)[20])
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1){
		return; // 如果图像是空或者格式不正确就返回
	}
	labelImg.release();
	binImg.convertTo(labelImg, CV_32SC1);// 矩阵数据类型转换
	int label = 0;
	int rows = binImg.rows;
	int cols = binImg.cols;
	for (int i = 1; i < rows - 1; i++){
		int* data = labelImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++){
			if (data[j] == 0){
                // std::pair主要的作用是将两个数据组合成一个数据，两个数据可以是同一类型或者不同类型。
				std::stack<std::pair<int, int>> neighborPixels;
				neighborPixels.push(std::pair<int, int>(j, i));// 向栈顶插入元素像素位置: <j,i>
				ymin[label] = i;
				ymax[label] = i;
				xmin[label] = j;
				xmax[label] = j;
				while (!neighborPixels.empty()){
                    // 如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它 
					std::pair<int, int> curPixel = neighborPixels.top();
					int curX = curPixel.first;
					int curY = curPixel.second;
					labelImg.at<int>(curY, curX) = 255;
					neighborPixels.pop();	// 出栈
					if ((curX > 0) && (curY > 0) && (curX < (cols - 1)) && (curY < (rows - 1))){
						if (labelImg.at<int>(curY - 1, curX) == 0){ //上
							neighborPixels.push(std::pair<int, int>(curX, curY - 1));
						}
						if (labelImg.at<int>(curY + 1, curX) == 0){ //下
							neighborPixels.push(std::pair<int, int>(curX, curY + 1));
							if ((curY + 1) > ymax[label])
								ymax[label] = curY + 1;
						}
						if (labelImg.at<int>(curY, curX - 1) == 0){ //左
							neighborPixels.push(std::pair<int, int>(curX - 1, curY));
							if ((curX - 1) < xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY, curX + 1) == 0){ //右
							neighborPixels.push(std::pair<int, int>(curX + 1, curY));
							if ((curX + 1) > xmax[label])
								xmax[label] = curX + 1;
						}
						if (labelImg.at<int>(curY - 1, curX - 1) == 0){ //左上
							neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1));
							if ((curX - 1) < xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY + 1, curX + 1) == 0){ //右下
							neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1));
							if ((curY + 1) > ymax[label])
								ymax[label] = curY + 1;
							if ((curX + 1) > xmax[label])
								xmax[label] = curX + 1;
						}
						if (labelImg.at<int>(curY + 1, curX - 1) == 0){ //左下
							neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1));
							if ((curY + 1) > ymax[label])
								ymax[label] = curY + 1;
							if ((curX - 1) < xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY - 1, curX + 1) == 0){ //右上
							neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX + 1) > xmax[label])
								xmax[label] = curX + 1;
						}
				}}
				++label; // 没有重复的团，开始新的标签
	}}}
	labelNum = label;
}

HandDetect::HandDetect() {
}

cv::Mat HandDetect::crop() {
    // 实现 crop 函数
}

cv::Mat HandDetect::skinEllipse(const cv::Mat& image) {
    // 创建椭圆mask
    cv::Mat skinCrCbHist = cv::Mat::zeros(256, 256, CV_8U);
    cv::ellipse(skinCrCbHist, cv::Point(113, 155), cv::Size(23, 25), 43, 0, 360, cv::Scalar(255), -1);
    // 转换至YCrCb空间
    cv::Mat YCrCb, HSV;
    cv::cvtColor(image, YCrCb, cv::COLOR_BGR2YCrCb);
    cv::cvtColor(image, HSV, cv::COLOR_BGR2HSV);
    // 拆分Cr, Cb
    std::vector<cv::Mat> channels_YCrCb, channels_HSV;
    cv::split(YCrCb, channels_YCrCb);
    cv::Mat Cr = channels_YCrCb[1];
    cv::Mat Cb = channels_YCrCb[2];
    cv::split(HSV, channels_HSV);
    cv::Mat H = channels_HSV[0];
    // 肤色检测，输出二值图像
    cv::Mat imageSkin = cv::Mat::zeros(Cr.size(), CV_8U);
    for (int j = 1; j < Cr.rows - 1; j++){
        uchar* currentCr = Cr.ptr< uchar>(j);
        uchar* currentCb = Cb.ptr< uchar>(j);
        uchar* currentH = H.ptr< uchar>(j);
        uchar* current = imageSkin.ptr< uchar>(j);
        for (int i = 1; i < Cb.cols - 1; i++){
            if ((currentCr[i] >= 135) && (currentCr[i] <= 170) && (currentCb[i] >= 94) && \
                (currentCb[i] <= 125) && (currentH[i] >= 1) && (currentH[i] <= 23))
                current[i] = 255;
            else
                current[i] = 0;
    }}
    // 进行一些修补
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(imageSkin, imageSkin, element);
    cv::dilate(imageSkin, imageSkin, element);
    // 基于标记的分水岭算法
    cv::Mat fg;
    cv::erode(imageSkin, fg, cv::Mat(), cv::Point(-1, -1), 6);	// 六次递归腐蚀
    // 识别没有对象的图像像素
    cv::Mat bg;
    cv::dilate(imageSkin, bg, cv::Mat(), cv::Point(-1, -1), 6); // 六次递归膨胀
    cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV); // 反转二值阈值化
    cv::Mat markers(imageSkin.size(), CV_8U, cv::Scalar(0));
    markers = fg + bg; // 显示标记图像
    // 使用分水岭进行分割
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
	segmenter.process(image);
    cv::Mat waterShed;
    waterShed = segmenter.getSegmentation();
    // 8向种子算法，给边框做标记
    cv::Mat labelImg;
    int label, ymin[20], ymax[20], xmin[20], xmax[20];
    // 对waterShed进行八邻接种子算法填充
    Seed_Filling(waterShed, labelImg, label, ymin, ymax, xmin, xmax);
    // cv::threshold(waterShed, waterShed, 1, 128, cv::THRESH_BINARY_INV);
    // 提取最大轮廓
    std::vector<cv::Point> largestContour = extractLargestContour(waterShed);
    std::vector<std::vector<cv::Point>> contours = {largestContour};
    // if (!largestContour.empty()) {
    //     cv::drawContours(image, contours, 0, cv::Scalar(0, 255, 0), 2);
    // } else {
    //     std::cout << "can't find hand posture" << std::endl;
    // }

    return waterShed;
}

std::vector<cv::Point> HandDetect::extractLargestContour(const cv::Mat& inputImage) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(inputImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 找到最大闭合轮廓
    int largestClosedContourIndex = -1;
    double largestClosedArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        // 检查轮廓是否是闭合的
        if (cv::isContourConvex(contours[i])) {
            double area = cv::contourArea(contours[i]);
            if (area > largestClosedArea) {
                largestClosedArea = area;
                largestClosedContourIndex = i;
    }}}
    if (largestClosedContourIndex != -1) {
        return contours[largestClosedContourIndex];
    }
    return {};
}


cv::Mat HandDetect::getHand() {
    // 实现 getHand 函数
}
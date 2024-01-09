/*----------------------------------【程序说明】---------------------------------------|
|	程序名称：基于视觉的手势识别交互系统											   |
|	程序功能：以手势代替鼠标进行人机交互											   |
|	程序时间：2019.1.14														           |
|	程序作者：黄俊															           |
|-------------------------------------------------------------------------------------*/

/*----------------------------------【头文件、命名空间包含部分】-----------------------|
|	描述：包含程序所使用的头文件和命名空间										       |
|-------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <map>
#include <stack>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <Windows.h>
#include<time.h>
using namespace std;
using namespace cv;
	
/*-----------------------------------【被调函数声明部分】-------------------------------|
|		描述：被调函数声明																|
|--------------------------------------------------------------------------------------*/
// 显示信息
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	cout << "\n      《基于视觉的手势识别交互系统设计》";
	cout << "\nDesign of vision-based gesture recognition interactive system";
	cout << "\n作者：黄俊	功能：以手势代替鼠标进行人机交互";
	cout << "\n---------------------------------------------------";
	cout << "\n手势1：鼠标移动";
	cout << "\n手势2：单击鼠标左键  手势3：单击鼠标右键";
	cout << "\n手势4：按下鼠标左键  手势5：松开鼠标左键";
	cout << "\n---------------------------------------------------\n";

}

// 八邻接种子算法，并返回每块区域的边缘框
void Seed_Filling(const cv::Mat& binImg, cv::Mat& labelImg, int& labelNum, int(&ymin)[20], int(&ymax)[20], int(&xmin)[20], int(&xmax)[20])  //种子填充法 
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)// 如果图像是空或者格式不正确就返回
	{
		return;
	}

	labelImg.release();
	binImg.convertTo(labelImg, CV_32SC1);// 矩阵数据类型转换
	int label = 0;
	int rows = binImg.rows;
	int cols = binImg.cols;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = labelImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			
			if (data[j] == 0)
			{
				std::stack<std::pair<int, int>> neighborPixels;// std::pair主要的作用是将两个数据组合成一个数据，两个数据可以是同一类型或者不同类型。
															   // pair是一个模板结构体，其主要的两个成员变量是first和second，这两个变量可以直接使用。
				neighborPixels.push(std::pair<int, int>(j, i));// 向栈顶插入元素像素位置: <j,i>
				ymin[label] = i;
				ymax[label] = i;
				xmin[label] = j;
				xmax[label] = j;
				while (!neighborPixels.empty())
				{
					std::pair<int, int> curPixel = neighborPixels.top();// 如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它 
					int curX = curPixel.first;
					int curY = curPixel.second;
					labelImg.at<int>(curY, curX) = 255;
					neighborPixels.pop();	// 出栈

					if ((curX > 0) && (curY > 0) && (curX < (cols - 1)) && (curY < (rows - 1)))
					{
						if (labelImg.at<int>(curY - 1, curX) == 0)                      //上
						{
							neighborPixels.push(std::pair<int, int>(curX, curY - 1));
							//ymin[label] = curY - 1;
						}

						if (labelImg.at<int>(curY + 1, curX) == 0)                      //下
						{
							neighborPixels.push(std::pair<int, int>(curX, curY + 1));
							if ((curY + 1) > ymax[label])
								ymax[label] = curY + 1;
						}

						if (labelImg.at<int>(curY, curX - 1) == 0)                      //左
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY));
							if ((curX - 1) < xmin[label])
								xmin[label] = curX - 1;
						}

						if (labelImg.at<int>(curY, curX + 1) == 0)                      //右
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY));
							if ((curX + 1) > xmax[label])
								xmax[label] = curX + 1;
						}

						if (labelImg.at<int>(curY - 1, curX - 1) == 0)                  //左上
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX - 1) < xmin[label])
								xmin[label] = curX - 1;
						}
						if (labelImg.at<int>(curY + 1, curX + 1) == 0)                  //右下
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY + 1));
							if ((curY + 1) > ymax[label])
								ymax[label] = curY + 1;
							if ((curX + 1) > xmax[label])
								xmax[label] = curX + 1;
						}

						if (labelImg.at<int>(curY + 1, curX - 1) == 0)                  //左下
						{
							neighborPixels.push(std::pair<int, int>(curX - 1, curY + 1));
							if ((curY + 1) > ymax[label])
								ymax[label] = curY + 1;
							if ((curX - 1) < xmin[label])
								xmin[label] = curX - 1;
						}

						if (labelImg.at<int>(curY - 1, curX + 1) == 0)                  //右上
						{
							neighborPixels.push(std::pair<int, int>(curX + 1, curY - 1));
							//ymin[label] = curY - 1;
							if ((curX + 1) > xmax[label])
								xmax[label] = curX + 1;
						}
					}
				}
				++label; // 没有重复的团，开始新的标签
			}
		}
	}
	labelNum = label;
}

// 镜像
void mirrorY(Mat src, Mat &dst)
{
	int row = src.rows;
	int col = src.cols;
	dst = src.clone();
	for (int i = 0; i < col; i++) {
		src.col(col - 1 - i).copyTo(dst.col(i));
	}
}

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
/*--------------------------------------【main()函数】-----------------------------------|
|		描述：控制台应用程序的入口函数，我们的程序从这里开始执行						 |
|---------------------------------------------------------------------------------------*/
int main()
{
	// 显示帮助信息
	ShowHelpText();

	// 设置视频读入，括号里面的数字是摄像头的选择，一般自带的是0
	cv::VideoCapture cap(0);

	if (!cap.isOpened())
	{
		return -1;
	}

	clock_t start, finish;
	double totaltime;

	string str1, str2, str3;
	Mat frame;
	Mat binImage, tmp, tmp1;
	Mat Y, Cr, Cb, H;
	vector<Mat> channels, channels1;

	//模板图片
	Mat mu[5];
	mu[0] = imread("m1.png", CV_8UC1); // 手势1
	mu[1] = imread("m2.png", CV_8UC1); // 手势2
	mu[2] = imread("m3.png", CV_8UC1); // 手势3
	mu[3] = imread("m4.png", CV_8UC1); // 手势4
	mu[4] = imread("m5.png", CV_8UC1); // 手势5
	float simi[5];	// 模板匹配
	float flag;		// 标志
	int flagx = 0, flagx1 = 0, flagx2 = 0;		// 中间变量
	float k = 1.5;	// 鼠标灵敏因子
	int curpointx = 0, curpointy = 0, prepointx = 367, prepointy = 80;	// 鼠标初始位置
	int xg_num = 0;

	bool stop = false;
	while (!stop)
	{
		cap >> frame;
		//imshow("The original image", frame);		// 测试显示【原始图像】
		//start = clock();
		

		// 镜像
		mirrorY(frame, frame);

		// 颜色空间变换（RGB to GRAY）
		cvtColor(frame, binImage, CV_BGR2GRAY);

		// 颜色空间变换（RGB to YCrCb）
		frame.copyTo(tmp);
		cvtColor(tmp, tmp, CV_BGR2YCrCb);

		// 颜色空间变换（RGB to HSV）
		frame.copyTo(tmp1);
		cvtColor(tmp1, tmp1, CV_BGR2HSV);

		// 通道分离
		split(tmp, channels);
		split(tmp1, channels1);
		Cr = channels.at(1);	// 分离出【色调Cr】
		Cb = channels.at(2);	// 分离出【饱和度Cb】
		H = channels1.at(0);	// 分离出【H】

		// 肤色检测，输出二值图像
		for (int j = 1; j < Cr.rows - 1; j++)	// 遍历图像像素点
		{
			uchar* currentCr = Cr.ptr< uchar>(j);
			uchar* currentCb = Cb.ptr< uchar>(j);
			uchar* currentH = H.ptr< uchar>(j);
			uchar* current = binImage.ptr< uchar>(j);

			for (int i = 1; i < Cb.cols - 1; i++)
			{
				if ((currentCr[i] >= 135) && (currentCr[i] <= 170) && (currentCb[i] >= 94) && (currentCb[i] <= 125) && (currentH[i] >= 1) && (currentH[i] <= 23))
					current[i] = 255;
				else
					current[i] = 0;
			}
		}

		// 灰度形态学处理
		erode(binImage, binImage, Mat());
		dilate(binImage, binImage, Mat());

		// 基于标记的分水岭算法
		cv::Mat fg;
		cv::erode(binImage, fg, cv::Mat(), cv::Point(-1, -1), 6);	// 六次递归腐蚀

		// 识别没有对象的图像像素
		cv::Mat bg;
		cv::dilate(binImage, bg, cv::Mat(), cv::Point(-1, -1), 6);	// 六次递归膨胀
		cv::threshold(bg, bg, 1, 128, cv::THRESH_BINARY_INV);	// 二进制阈值函数图像分割cv::THRESH_BINARY_INV 超过阈值 则 值变为 0,其他为 128 黑白二值反转(反转二值阈值化)

		// 显示标记图像
		cv::Mat markers(binImage.size(), CV_8U, cv::Scalar(0));
		markers = fg + bg;

		// 创建分水岭分割对象
		WatershedSegmenter segmenter;
		segmenter.setMarkers(markers);
		segmenter.process(frame);// 应用分水岭算法

		Mat waterShed;
		waterShed = segmenter.getSegmentation();

		// 8向种子算法，给边框做标记
		Mat labelImg;
		int label, ymin[20], ymax[20], xmin[20], xmax[20];
		Seed_Filling(waterShed, labelImg, label, ymin, ymax, xmin, xmax);

		// 统计一下区域中的肤色区域比例
		float fuseratio[20];
		for (int k = 0; k < label; k++)
		{
			fuseratio[k] = 1;
			if (((xmax[k] - xmin[k] + 1) > 50) && ((xmax[k] - xmin[k] + 1) < 300) && ((ymax[k] - ymin[k] + 1) > 150) && ((ymax[k] - ymin[k] + 1) < 450))
			{
				int fusepoint = 0;
				for (int j = ymin[k]; j < ymax[k]; j++)
				{
					uchar* current = waterShed.ptr< uchar>(j);
					for (int i = xmin[k]; i < xmax[k]; i++)
					{
						if (current[i] == 255)
							fusepoint++;
					}
				}
				fuseratio[k] = float(fusepoint) / ((xmax[k] - xmin[k] + 1) * (ymax[k] - ymin[k] + 1));
			}
		}

		Size dsize = Size(108, 128);
		// 给符合阈值条件的位置画框
		for (int i = 0; i < label; i++)
		{
			if ((fuseratio[i] < 0.58))
			{
				// 尺度调整
				Mat rROI = Mat(dsize, CV_8UC1);
				resize(waterShed(Rect(xmin[i], ymin[i], xmax[i] - xmin[i], ymax[i] - ymin[i])), rROI, dsize);
				imshow("手势区域", rROI);

				// 模板匹配
				Mat result;
				for (int mp = 0; mp < 5; mp++)
				{
					matchTemplate(rROI, mu[mp], result, CV_TM_SQDIFF_NORMED);
					simi[mp] = result.ptr<float>(0)[0];
				}
				
				// 寻找最佳匹配
				flagx2 = flagx1;
				flagx1 = flagx;
				flagx = 0;
				flag = simi[0];
				for (int j = 1; j < 5; j++)
				{
					if (simi[j] < flag)
					{
						flagx = j;
						flag = simi[j];
					}
				}

				cv::Point end = cv::Point(xmin[i], ymin[i] - 15);   // 加标签【位置】
				str1 = "(" + to_string(xmin[i]) + "," + to_string(ymin[i]) + ") " + " (" + to_string(xmax[i]) + "," + to_string(ymax[i]) + ")";
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
				putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(xmin[i] - 80, ymin[i] + 20);   // 加标签【手部标记】
				str1 = "Hand" + to_string(flagx + 1);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(20, 20);   // 加标签【肤色比例】
				str1 = "Skin area ratio: " + to_string(fuseratio[i]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(20, 40);   // 加标签【长宽】
				str1 = "Height: " + to_string(ymax[i] - ymin[i]) + "  Width: " + to_string(xmax[i] - xmin[i]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(20, 60);   // 加标签【数字 匹配度】
				str1 = "Suitability_1: " + to_string(simi[0]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(20, 80);   // 加标签【数字 匹配度】
				str1 = "Suitability_2: " + to_string(simi[1]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(20, 100);   // 加标签【数字 匹配度】
				str1 = "Suitability_3: " + to_string(simi[2]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);


				end = cv::Point(20, 120);   // 加标签【数字 匹配度】
				str1 = "Suitability_4: " + to_string(simi[3]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				end = cv::Point(20, 140);   // 加标签【数字 匹配度】
				str1 = "Suitability_5: " + to_string(simi[4]);
				putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				//putText(frame, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

				if (flagx == 0)
				{
					curpointx = xmin[i];
					curpointy = ymin[i];

					int dx = curpointx - prepointx;
					int dy = curpointy - prepointy;

					prepointx = curpointx;
					prepointy = curpointy;
					mouse_event(MOUSEEVENTF_MOVE, k * dx, k * dy, 0, 0);	// 鼠标移动
					
					end = cv::Point(20, 400);   // 加标签【数字 匹配度】
					str1 = "MOUSEEVENTF_MOVE";
					putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				}

				if (flagx == 1 && flagx1 != 1 && flagx2 != 1)
				{
					mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);			// 鼠标左键按下
					mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);			// 鼠标左键松开
					mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);			// 鼠标左键按下
					mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);			// 鼠标左键松开
					end = cv::Point(20, 440);   // 加标签【数字 匹配度】
					str1 = "MOUSEEVENTF_LEFT_CLICK";
					putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				}
				if (flagx == 2 && flagx1 != 2 && flagx2 != 2)
				{
					mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);			// 鼠标右键按下
					mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);			// 鼠标右键松开
					end = cv::Point(20, 440);   // 加标签【数字 匹配度】
					str1 = "MOUSEEVENTF_RIGHT_CLICK";
					putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				}
				if (flagx == 3)
				{
					mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);			// 鼠标左键按下
					end = cv::Point(20, 440);   // 加标签【数字 匹配度】
					str1 = "MOUSEEVENTF_LEFT_DOWN";
					putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				}
				if (flagx == 4)
				{
					mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);			// 鼠标左键松开
					end = cv::Point(20, 440);   // 加标签【数字 匹配度】
					str1 = "MOUSEEVENTF_LEFT_UP";
					putText(waterShed, str1, end, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
				}

				rectangle(waterShed, Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]), Scalar::all(255), 3, 8, 0);	// 加框
				rectangle(frame, Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]), cv::Scalar(0, 255, 0), 3, 8, 0);	// 加框
			}
		}
		imshow("信息显示", waterShed);
		imshow("功能显示", frame);

		// 时间显示
		//finish = clock();
		//totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
		//cout << "\n此程序的运行时间为" << totaltime << "秒！" << endl;
		
		if (waitKey(1) >= 0)
			stop = true;

	}
	
	return 0;
}
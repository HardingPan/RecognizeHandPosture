# RecognizeHandPosture
This project utilizes image processing and deep learning techniques to recognize human hand posture.   
本项目使用图像处理和机器学习技术实现手势识别任务    
## 功能实现 
**手势皮肤提取**
- H, Cr, Cb 多通道提取皮肤特征
- 风水岭算法得到手部皮肤轮廓
- 八邻接种子算法填充手部区域，得到二值化的皮肤图像及其轮廓  

**轮廓特征计算**
- 使用傅里叶描述子表示轮廓特征，并对描述子进行截断   

**手势分类**
- 使用KNN对傅里叶描述子进行分类。本项目中图像就是数据集，其文件名即为标签

## 使用说明
在PC端配置`OPENCV`进入`dev_ws/build`文件夹，进行编译：
```cmake
cmake ..
make
```
编译完成后会在`dev_ws/bin`文件夹内生成可执行文件`hand_detection`，运行命令：
```zsh
./hand_detection <arg>
```
**arg参数说明**
- n 识别数字
- s 识别手语（数据集内只存放了a、b、c三个手势）
- o 最简单的0和1分类
- d 三种方向手势的分类
- c 制作自己的数据集
## 文件说明
```
.
├── dataset
│   ├── Direction
│   ├── NumberLock
│   ├── OneOrZero
│   └── SignLanguage
├── dev_ws
│   ├── bin
│   ├── build
│   ├── CMakeLists.txt
│   ├── include
│   ├── main.cpp
│   └── src
├── LICENSE
└── README.md
```
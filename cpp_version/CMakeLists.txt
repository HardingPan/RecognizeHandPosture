cmake_minimum_required(VERSION 3.10)

project(hand_detection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 添加头文件目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    )

# 添加源文件
add_executable(
    hand_detection
    main.cpp
    src/detect_hand.cpp
)

# 链接 OpenCV 库
target_link_libraries(hand_detection ${OpenCV_LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
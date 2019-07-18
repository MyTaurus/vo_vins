/*
 * @Description: 这是采用本地图像模拟相机的一个类，用来测试数据集
 * @Author: Taurus.Lee
 * @Date: 2019-07-10 16:46:58
 * @LastEditTime: 2019-07-10 17:17:21
 */
#pragma once
#include "camera.h"

namespace TVins
{
#define ANALOG_CALI_FILE "./config/cali.yaml/"
class AnalogCamera : public Camera
{
public:
    AnalogCamera() : Camera(ANALOG_CALI_FILE)
    {
        m_count = 0;
    }

    int Open(const std::string &sn = "");
    int Close();
    cv::Mat GetOneFrame();

private:
    int m_count;
    std::vector<cv::String> image_path;
};
} // namespace TVins
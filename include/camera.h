/*
 * @Description: 这是相机的基类，希望相机有如下功能
 * @Author: Taurus.Lee
 * @Date: 2019-07-10 15:03:10
 * @LastEditTime: 2019-07-10 17:25:15
 */
#pragma once
#include <string>
#include <memory>
#include "calibration.h"

namespace TVins
{
class Camera
{
public:
    Camera(const std::string& cali_file = "./config/cali.yaml/")
    {
        m_calibration.Init(cali_file);
    }
    
    //打开相机
    virtual int Open(const std::string& sn = "") = 0;
    //关闭相机
    virtual int Close() = 0;
    //相机单帧采图
    virtual cv::Mat GetOneFrame() = 0;
    
    calibration& Calibration() {return m_calibration;};
protected:
    calibration m_calibration;
};
} // namespace TVins
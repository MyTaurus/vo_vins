/*
 * @Description: 一个纯视觉里程计计算部分
 * @Author: Taurus.Lee
 * @Date: 2019-07-15 14:20:01
 * @LastEditTime: 2019-07-15 17:28:12
 */
#pragma once
#include <opencv2/opencv.hpp>
namespace TVins
{
    class VisionEstimator
    {
        public:
        void EstimatorMotion(const cv::Mat& pre_image, const cv::Mat& cur_image, cv::Mat &R, cv::Mat& t);
    };

    void test();
}
/*
 * @Description:这里需要对当前程序用到的一些方法进行定义，比如：三角测量的方法，根据基本矩阵求RT的方法 
 * @Author: Taurus.Lee
 * @Date: 2019-07-11 16:37:34
 * @LastEditTime: 2019-07-15 20:07:32
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "calibration.h"

namespace TVins
{
    //根据点对求解两帧图像之间的RT关系，这里使用本质矩阵求解法
int SolveRelativeRT_CV(const std::vector<cv::Point2d> &pre_points,const std::vector<cv::Point2d> &cur_points,  calibration &cali, cv::Mat &R, cv::Mat &t);

int EssentialMatRT_Homography(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres, calibration &cali, cv::Mat &R, cv::Mat t);

/**
 * @brief 从特征点匹配求homography（normalized DLT）
 * 
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     单应矩阵
 */
cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);

// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX
// 它们都属于 x = aPX模型
//                         |X|
// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|     |0|
// |p0 -  xp2| X = |0|
// |xp1 - yp0|     |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|

/**
 * @brief 给定投影矩阵P1,P2和图像上的点kp1,kp2，从而恢复3D坐标
 *
 * @param kp1 特征点, in reference frame
 * @param kp2 特征点, in current frame
 * @param P1  投影矩阵P1
 * @param P2  投影矩阵P２
 * @param x3D 三维点
 * @see       Multiple View Geometry in Computer Vision - 12.2 Linear triangulation methods p312
 */
void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

/**
 * ＠brief 归一化特征点到同一尺度（作为normalize DLT的输入）
 *
 * [x' y' 1]' = T * [x y 1]' \n
 * 归一化后x', y'的均值为0，sum(abs(x_i'-0))=1，sum(abs((y_i'-0))=1
 * 
 * @param vKeys             特征点在图像上的坐标
 * @param vNormalizedPoints 特征点归一化后的坐标
 * @param T                 将特征点归一化的矩阵
 */
void Normalize(const std::vector<cv::Point2f> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

//ICP方法，如果有两个平面，该方法可以用来使用
void ICP(const std::vector<cv::Point3f>& pts1,const std::vector<cv::Point3f>& pts2, cv::Mat &R, cv::Mat& t);

//这是不用另外处理畸变的方法
void SolvePnp(const std::vector<cv::Point3f>& p3d,const std::vector<cv::Point2f>& p2d, cv::Mat &R, cv::Mat& t);

cv::Point3f  Rmat2Theta(const cv::Mat& rmat);
} // namespace TVins
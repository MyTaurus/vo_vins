/*
 * @Description: 
 * @Author: Taurus.Lee
 * @Date: 2019-07-10 15:07:03
 * @LastEditTime: 2019-07-15 20:04:12
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
namespace TVins
{
class CaliParam
{
public:
    void InitCaliParam(const std::string &calibration_file);
    void Load();
    void Save();

    void GenerateCameraMatrix(const double &fx, const double &fy, const double &cx, const double &cy);

    void GenerateDisCoffes(const double &k1, const double &k2, const double &p1, const double &p2, const double &k3 = 0);

    double k1(void) const;
    double k2(void) const;
    double p1(void) const;
    double p2(void) const;
    double fx(void) const;
    double fy(void) const;
    double cx(void) const;
    double cy(void) const; 

    cv::Size PatternSize;
    int ChessBoardWidth;
    std::string CaliImagePath;
    cv::Mat CameraMatrix;
    cv::Mat DisCoffes;
    int Image_W;
    int Image_H;
    int HasK3;

private:
    std::string m_calibration_file;
};

class calibration
{
public:
    void Init(const std::string &path);
    bool StartCali();
    //void Load();
    void Save();
    CaliParam &Param() { return m_param; };
    //生成畸变校正映射图
    void InitUndistortMap(cv::Mat &map1, cv::Mat &map2, double fScale = 1.0) const;
    //进行畸变校正
    cv::Mat InitUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2, float fx = -1.0f, float fy = -1.0f, cv::Size imageSize = cv::Size(0, 0), float cx = -1.0f, float cy = -1.0f, cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F));
     //对点进行畸变校正，求出来畸变量的差值
    void DistortionI(const cv::Point2d &p, cv::Point2d &p_u);
    //对点进行畸变校正，求出来畸变量的差值
    void Distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u) const;
    //根据当前映射关系，图像重定位
    void Distortion(const cv::Mat& in_image, cv::Mat& out_image);
    //相机坐标系到像素坐标，这里是带了畸变校正的
    void SpaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const;
    void LiftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P);
private:
    std::vector<std::vector<cv::Point2f>> findChessBoard(std::vector<cv::String> fn);

    //这里不会进行畸变矫正
    Eigen::Vector3d i2c(const Eigen::Vector2d &p);
    Eigen::Vector2d c2i(const Eigen::Vector3d &p);

private:
    // double fx, fy, cx, cy;
    // double k1, k2, p1, p2, k3;
    CaliParam m_param;
    cv::Mat m_map1, m_map2;
};
} // namespace TVins

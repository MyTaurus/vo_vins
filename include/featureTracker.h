/*
 * @Description: 这里定义一个求解特征点的类
 * @Author: Taurus.Lee
 * @Date: 2019-07-11 15:21:41
 * @LastEditTime: 2019-07-18 09:45:42
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "calibration.h"
#include <vector>

namespace TVins
{
class TrackPoint
{
    public:
        TrackPoint()
        {
            PointI_Old = cv::Point2d(0, 0);
            PointI = cv::Point2d(0, 0);
            PointC_Old = cv::Point3d(0, 0, 0);
            PointC = cv::Point3d(0, 0, 0);
        }

        TrackPoint(cv::Point2d pi_old, cv::Point2d pi, int ids, int trackCount = 0)
        {
            PointI_Old = pi_old;
            PointI = pi;
            PointC_Old = cv::Point3d(0, 0, 0);
            PointC = cv::Point3d(0, 0, 0);
            TrackCount = trackCount;
            Ids = ids;
        }
        
        TrackPoint& operator= (const TrackPoint& a)
        {
            PointI_Old = a.PointI_Old;
            PointI = a.PointI;
            PointC_Old = a.PointC_Old;
            PointC = a.PointC;
            TrackCount = a.TrackCount;
            Ids = a.Ids;
        }

    public:
        cv::Point2d PointI_Old;
        cv::Point2d PointI;
        cv::Point3d PointC_Old;
        cv::Point3d PointC;
        int TrackCount;
        int Ids;
};

class FeatureTracker
{
public:
    FeatureTracker(const calibration& cali);
    
    /** 
    *＠brief:每次读入新的图像,返回特定的值
    */
    std::vector<cv::Point3d> ReadImage(const cv::Mat &image, const bool& isFindFeatures = true, const double &cur_time = 0);

    std::vector<TrackPoint> GetTrackPoints() const;
private:
    /** 
    *＠brief:对于已经track到的点位，则不需要做补充，尽量的稀疏特征点
    */
    void getMask(std::vector<TrackPoint>& track_points, cv::Mat &mask, std::vector<uchar>& status);

    /** 
    *＠brief:根据F矩阵剔除多余的点
    */
    void rejectWithF(const std::vector<cv::Point2f>& vec_piold, const std::vector<cv::Point2f>& vec_pi, std::vector<uchar>& status);

    /** 
    *＠brief:对得到的点进行消畸变并归一化到相机坐标系
    */
    void undistortedPoints(std::vector<TrackPoint>& track_points);

    /** 
    *＠brief:判定当前点是否在视野内，不是则剔除
    */
    bool inBorder(const cv::Point2d &pt);

    /** 
    *＠brief:根据status删除模板
    */
    template <typename T>
    void reduceVector(std::vector<T> &v, std::vector<uchar> status);

    void trackPoint2pi(const std::vector<TrackPoint>& vec_trackPoint, std::vector<cv::Point2f>& vec_point);

    void pi2trackPoint(const std::vector<cv::Point2d>& vec_point, const int& id, const int& track_count,std::vector<TrackPoint>& vec_trackPoint);

    //存储点与图像等
    cv::Mat m_prev_img, m_cur_img, m_forw_img;

    std::vector<TrackPoint> m_track_points, m_pre_track_points;
    

    //速度计量，在有时间参数的时候有用！
    std::vector<cv::Point2d> m_pts_velocity;
    
    calibration m_cali;

    int m_num;

    double cur_time, prev_time;
};
} // namespace TVins
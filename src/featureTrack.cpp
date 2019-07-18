#include "featureTracker.h"
#include <algorithm>
using namespace std;
using namespace cv;

namespace TVins
{
const bool EQUALIZE = false;
const int MAX_CNT = 150;
const int MIN_DIST = 30;

FeatureTracker::FeatureTracker(const calibration &cali)
    : m_cali(cali),
      m_num(0)
{
}

std::vector<cv::Point3d> FeatureTracker::ReadImage(const cv::Mat &image, const bool &isFindFeatures, const double &cur_time)
{
    std::vector<cv::Point3d> cur_p3d;

    cv::Mat img;
    if (EQUALIZE)
    {
        //先做直方图均衡化，增加图像的稳定性
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(image, img);
    }
    else
    {
        img = image;
    }

    if (m_forw_img.empty())
    {
        m_prev_img = m_cur_img = m_forw_img = img;
    }
    else
    {
        m_forw_img = img;
    }

    vector<cv::Point2f> forw_points, cur_points;
    trackPoint2pi(m_track_points, cur_points);

    vector<TrackPoint> frow_track_points;

    if (cur_points.size() > 0)
    {
        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(m_cur_img, m_forw_img, cur_points, forw_points, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_points.size()); i++)
            if (status[i] && !inBorder(forw_points[i]))
                status[i] = 0;

        for (int i = 0; i < int(m_track_points.size()); i++)
        {
            if (status[i])
            {
                TrackPoint tp(cur_points[i], forw_points[i], m_track_points[i].Ids, m_track_points[i].TrackCount + 1);
                tp.PointC_Old = m_track_points[i].PointC;
                frow_track_points.push_back(tp);
            }
        }

        reduceVector<Point2f>(cur_points, status);
        reduceVector<Point2f>(forw_points, status);
    }

    if (isFindFeatures)
    {
        vector<uchar> status;
        rejectWithF(cur_points, forw_points, status);
        reduceVector<TrackPoint>(frow_track_points, status);
        reduceVector<Point2f>(cur_points, status);
        reduceVector<Point2f>(forw_points, status);

        status.clear();
        cv::Mat mask;
        getMask(frow_track_points ,mask, status);
        reduceVector<TrackPoint>(frow_track_points, status);

        std::vector<cv::Point2f> n_pts;
        int n_max_cnt = MAX_CNT - forw_points.size();
        if (n_max_cnt > 0)
        {
            cv::goodFeaturesToTrack(m_forw_img, n_pts, MAX_CNT - forw_points.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        
        for (auto &p : n_pts)
        {
            frow_track_points.push_back(TrackPoint(Point2d(0, 0), p, m_num++));
        }
    }

    m_prev_img = m_cur_img;
    m_cur_img = m_forw_img;

    undistortedPoints(frow_track_points);

    m_pre_track_points = m_track_points;
    m_track_points = frow_track_points;

    return cur_p3d;
}

std::vector<TrackPoint> FeatureTracker::GetTrackPoints() const
{
    return m_track_points;
}

/** 
    *＠brief:判定当前点是否在视野内，不是则剔除
    */
bool FeatureTracker::inBorder(const cv::Point2d &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < m_cali.Param().Image_W - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < m_cali.Param().Image_H - BORDER_SIZE;
}

/** 
    *＠brief:根据status删除模板
    */
template <typename T>
void FeatureTracker::reduceVector(std::vector<T> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTracker::trackPoint2pi(const std::vector<TrackPoint>& vec_trackPoint, std::vector<cv::Point2f>& vec_point)
{
    vec_point.clear();
    for(auto &p : vec_trackPoint)
    {
        vec_point.push_back(cv::Point2f(p.PointI.x, p.PointI.y));
    }
}

void FeatureTracker::pi2trackPoint(const std::vector<cv::Point2d>& vec_point, const int& id, const int& track_count,std::vector<TrackPoint>& vec_trackPoint)
{
    vec_trackPoint.clear();
    for(auto &p : vec_point)
    {
        TrackPoint tp;
        tp.Ids = id;
        tp.TrackCount = track_count;
        tp.PointI = p;
        vec_trackPoint.push_back(tp);
    }
}

/** 
    *＠brief:根据F矩阵剔除多余的点
    */
void FeatureTracker::rejectWithF(const std::vector<cv::Point2f>& vec_piold,const std::vector<cv::Point2f>& vec_pi, vector<uchar>& status)
{
    if (vec_pi.size() >= 8)
    {
        vector<cv::Point2d> un_cur_pts, un_forw_pts;
        //把当前点消畸变，然后求解
        for (unsigned int i = 0; i < vec_pi.size(); i++)
        {
            cv::Point2d p_u; 
            m_cali.DistortionI(vec_piold[i], p_u);
            un_cur_pts.push_back(p_u);
            m_cali.DistortionI(vec_pi[i], p_u);
            un_forw_pts.push_back(p_u);
        }

        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, 1.0, 0.99, status);
    }
}

void FeatureTracker::getMask(std::vector<TrackPoint>& track_points, cv::Mat &mask, std::vector<uchar>& status)
{   
    cv::Mat image = cv::Mat(m_cali.Param().Image_H, m_cali.Param().Image_W, CV_8UC1, cv::Scalar(255));

    // sort(track_points.begin(), track_points.end(), [](TrackPoint tp1, TrackPoint tp2) {
    //     return tp1.TrackCount > tp2.TrackCount;
    // });

    int i = 0;
    for (auto &it : track_points)
    {
        //如果图像在roi里面，则加入该点，注意，点的顺序在这里被打乱了
        if (image.at<uchar>(it.PointI) == 255)
        {
            cv::circle(image, it.PointI, MIN_DIST, 0, -1);
            status.push_back(1);
        }
        else
        {
            status.push_back(0);
        }
    }

    image.copyTo(mask);
}

void FeatureTracker::undistortedPoints(std::vector<TrackPoint>& track_points)
{
    for (unsigned int i = 0; i < track_points.size(); i++)
    {
        Eigen::Vector2d a(track_points[i].PointI.x, track_points[i].PointI.y);
        Eigen::Vector3d b;
        m_cali.LiftProjective(a, b);
        cv::Point2d point = cv::Point2d(b.x() / b.z(), b.y() / b.z());
        track_points[i].PointC = cv::Point3d(point.x, point.y, 1);
    }
}
} // namespace TVins
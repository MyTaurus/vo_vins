#include "calibration.h"

using namespace cv;
using namespace std;
namespace TVins
{
#pragma region calibratioan param
void CaliParam::InitCaliParam(const std::string &calibration_file)
{
    m_calibration_file = calibration_file;
    Load();
}

void CaliParam::Load()
{
    FileStorage fileStorage(m_calibration_file, cv::FileStorage::READ);
    Image_W = fileStorage["Image_W"];
    Image_H = fileStorage["Image_H"];
    ChessBoardWidth = fileStorage["ChessBoardWidth"];
    HasK3 = fileStorage["HasK3"];
    fileStorage["CaliImagePath"] >> CaliImagePath;
    fileStorage["PatternSize"] >> PatternSize;
    fileStorage["CameraMatrix"] >> CameraMatrix;
    fileStorage["DisCoffes"] >> DisCoffes;
    fileStorage.release();
}

void CaliParam::Save()
{
    FileStorage fileStorage(m_calibration_file, cv::FileStorage::WRITE);
    fileStorage << "Image_W" << Image_W;
    fileStorage << "Image_H" << Image_H;
    fileStorage << "PatternSize" << PatternSize;
    fileStorage << "ChessBoardWidth" << ChessBoardWidth;
    fileStorage << "CaliImagePath" << CaliImagePath;
    fileStorage << "CameraMatrix" << CameraMatrix;
    fileStorage << "DisCoffes" << DisCoffes;
    fileStorage << "HasK3" << HasK3;
    fileStorage.release();
}

void CaliParam::GenerateCameraMatrix(const double &fx, const double &fy, const double &cx, const double &cy)
{
    CameraMatrix.create(3, 3, cv::DataType<double>::type);
    CameraMatrix.zeros(3, 3, cv::DataType<double>::type);
    CameraMatrix.at<double>(0, 0) = fx;
    CameraMatrix.at<double>(1, 1) = fy;
    CameraMatrix.at<double>(0, 2) = cx;
    CameraMatrix.at<double>(1, 2) = cy;
    CameraMatrix.at<double>(2, 2) = 1;
}

void CaliParam::GenerateDisCoffes(const double &k1, const double &k2, const double &p1, const double &p2, const double &k3)
{
    DisCoffes.create(5, 1, cv::DataType<double>::type);
    DisCoffes.zeros(5, 1, cv::DataType<double>::type);
    DisCoffes.at<double>(0) = k1;
    DisCoffes.at<double>(1) = k2;
    DisCoffes.at<double>(2) = p1;
    DisCoffes.at<double>(3) = p2;
    DisCoffes.at<double>(4) = k3;
}

double CaliParam::k1(void) const
{
    return DisCoffes.at<double>(0);
}

double CaliParam::k2(void) const
{
    return DisCoffes.at<double>(1);
}

double CaliParam::p1(void) const
{
    return DisCoffes.at<double>(2);
}

double CaliParam::p2(void) const
{
    return DisCoffes.at<double>(3);
}

double CaliParam::fx(void) const
{
    return CameraMatrix.at<double>(0, 0);
}

double CaliParam::fy(void) const
{
    return CameraMatrix.at<double>(1, 1);
}

double CaliParam::cx(void) const
{
    return CameraMatrix.at<double>(0, 2);
}

double CaliParam::cy(void) const
{
    return CameraMatrix.at<double>(1, 2);
}
#pragma endregion

#pragma region calibration
void calibration::Init(const std::string &path)
{
    m_param.InitCaliParam(path);
}

vector<vector<Point2f>> calibration::findChessBoard(vector<cv::String> fn)
{
    cv::Size pattern_size(11, 8);
    vector<vector<Point2f>> corners;
    int count = fn.size();
    for (size_t i = 0; i < count; i++)
    {
        Mat image = imread(fn[i]);

        vector<Point2f> vec_point;
        bool is_found = cv::findChessboardCorners(image, pattern_size, vec_point, cv::CALIB_CB_ADAPTIVE_THRESH);

        cv::drawChessboardCorners(image, pattern_size, Mat(vec_point), is_found);

        corners.push_back(vec_point);
        imshow("show image", image);

        waitKey(0);
    }

    return corners;
}

Eigen::Vector3d calibration::i2c(const Eigen::Vector2d &p)
{
    Eigen::Vector3d P;
    double mx_d, my_d;

    double inv_K11 = 1.0 / m_param.fx();
    double inv_K13 = -m_param.cx() / m_param.fx();
    double inv_K22 = 1.0 / m_param.fy();
    double inv_K23 = -m_param.cy() / m_param.fy();
    mx_d = inv_K11 * p(0) + inv_K13;
    my_d = inv_K22 * p(1) + inv_K23;

    P << mx_d, my_d, 1.0;
    return P;
}

Eigen::Vector2d calibration::c2i(const Eigen::Vector3d &p)
{
    Eigen::Vector2d P;
    double mx_d, my_d;

    mx_d = m_param.fx() * p(0) + m_param.cx();
    my_d = m_param.fy() * p(1) + m_param.cy();

    P << mx_d, my_d;
    return P;
}

bool calibration::StartCali()
{
    cv::Size board_size = m_param.PatternSize;

    cv::String pattern = m_param.CaliImagePath + "*.png";

    vector<cv::String> fn;
    glob(pattern, fn, false);
    size_t count = fn.size();
    vector<vector<Point2f>> corners = findChessBoard(fn);

    vector<vector<Point3f>> objectPoints;

    for (int t = 0; t < count; t++)
    {
        vector<Point3f> vec_point3;
        for (int i = 0; i < board_size.height; i++)
        {
            for (int j = 0; j < board_size.width;
                 j++)
            {
                Point3f realPoint; /* 假设标定板放在世界坐标系中z=0的平面上 */
                realPoint.x = i * 10;
                realPoint.y = j * 10;
                realPoint.z = 0;
                vec_point3.push_back(realPoint);
            }
        }
        objectPoints.push_back(vec_point3);
    }

    cv::Mat cameraMatrix, disCoffes;
    std::vector<cv::Mat> rvecs, tvecs;
    bool isCalibrationSuccessed = cv::calibrateCamera(objectPoints, corners, Size(m_param.Image_W, m_param.Image_H), cameraMatrix, disCoffes, rvecs, tvecs);

    if (!isCalibrationSuccessed)
        return -1;
    cout << "开始评价标定结果………………";
    double total_err = 0.0; // 所有图像的平均误差的总和
    double err = 0.0;       // 每幅图像的平均误差

    double totalErr = 0.0;
    double totalPoints = 0.0;
    vector<Point2f> image_points_pro; // 保存重新计算得到的投影点
    for (int i = 0; i < count; i++)
    {
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, disCoffes, image_points_pro);
        //通过得到的摄像机内外参数，对角点的空间三维坐标进行重新投影计算
        err = norm(Mat(corners[i]), Mat(image_points_pro), NORM_L2);
        totalErr += err * err;
        totalPoints += objectPoints[i].size();
        err /= objectPoints[i].size();
        cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
        total_err += err;
    }
    cout << "重投影误差2：" << sqrt(totalErr / totalPoints) << "像素" << endl
         << endl;
    cout << "重投影误差3：" << total_err / count << "像素" << endl
         << endl;

    cameraMatrix.copyTo(m_param.CameraMatrix);
    disCoffes.copyTo(m_param.DisCoffes);
}

void calibration::Save()
{
}

//Xc = (1+k1*r^2+k2*r^4)*Xp+2p1*Xc*Yc+p2*(r^2+2Xc^2)，这里求出的是dx,dy
void calibration::Distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u) const
{
    double k1 = m_param.k1();
    double k2 = m_param.k2();
    double p1 = m_param.p1();
    double p2 = m_param.p2();

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
        p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

//对点进行畸变校正，求出来畸变量的差值
void calibration::DistortionI(const cv::Point2d &p, cv::Point2d &p_u)
{
    Eigen::Vector2d du;
    Eigen::Vector3d p3d = i2c(Eigen::Vector2d(p.x, p.y));
    Distortion(Eigen::Vector2d(p3d.x(), p3d.y()), du);

    p3d.x() = p3d.x() - du.x();
    p3d.y() = p3d.y() - du.y();

    Eigen::Vector2d pu = c2i(p3d);
    p_u.x = pu.x();
    p_u.y = pu.y();
}

void calibration::Distortion(const cv::Mat &in_image, cv::Mat &out_image)
{
    if (m_map1.cols == 0)
    {
        InitUndistortMap(m_map1, m_map2);
    }

    cv::remap(in_image, out_image, m_map1, m_map2, CV_INTER_LINEAR);
}

void calibration::InitUndistortMap(cv::Mat &map1, cv::Mat &map2, double fScale) const
{
    double inv_K11 = 1.0 / m_param.fx();
    double inv_K13 = -m_param.cx() / m_param.fx();
    double inv_K22 = 1.0 / m_param.fy();
    double inv_K23 = -m_param.cy() / m_param.fy();


    cv::Size imageSize(m_param.Image_W, m_param.Image_H);

    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            double mx_u = inv_K11 / fScale * u + inv_K13 / fScale;
            double my_u = inv_K22 / fScale * v + inv_K23 / fScale;

            Eigen::Vector3d P;
            P << mx_u, my_u, 1.0;

            Eigen::Vector2d p;
            SpaceToPlane(P, p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}

cv::Mat calibration::InitUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2, float fx, float fy, cv::Size imageSize, float cx, float cy, cv::Mat rmat)
{
if (imageSize == cv::Size(0, 0))
    {
        imageSize = cv::Size(m_param.Image_W, m_param.Image_H);
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    // assume no skew
    Eigen::Matrix3f K_rect;

    if (cx == -1.0f || cy == -1.0f)
    {
        K_rect << fx, 0, imageSize.width / 2,
                  0, fy, imageSize.height / 2,
                  0, 0, 1;
    }
    else
    {
        K_rect << fx, 0, cx,
                  0, fy, cy,
                  0, 0, 1;
    }

    if (fx == -1.0f || fy == -1.0f)
    {
        K_rect(0,0) = m_param.fx();
        K_rect(1,1) = m_param.fy();
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            Eigen::Vector3f xo;
            xo << u, v, 1;

            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            SpaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

//Z = 1.0，这里约束在z=1.0的空间平面
void calibration::LiftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P)
{
    double mx_d, my_d, mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    //double lambda;

    // Lift points to normalised plane
    Eigen::Vector3d p3d = i2c(p);
    mx_d = p3d.x();
    my_d = p3d.y();

    // Recursive distortion model???
    int n = 8;
    Eigen::Vector2d d_u;
    Distortion(Eigen::Vector2d(mx_d, my_d), d_u);
    // Approximate value
    mx_u = mx_d - d_u(0);
    my_u = my_d - d_u(1);

    //暂时不知道这里采用递归畸变模型的意义
    // for (int i = 1; i < n; ++i)
    // {
    //     Distortion(Eigen::Vector2d(mx_u, my_u), d_u);
    //     mx_u = mx_d - d_u(0);
    //     my_u = my_d - d_u(1);
    // }

    // Obtain a projective ray
    P << mx_u, my_u, 1.0;
}

void calibration::SpaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const
{
    Eigen::Vector2d p_u, p_d;

    // Project points to the normalised plane
    p_u << P(0) / P(2), P(1) / P(2);

    // Apply distortion
    Eigen::Vector2d d_u;
    Distortion(p_u, d_u);
    p_d = p_u + d_u;

    // Apply generalised projection matrix
    p << m_param.fx() * p_d(0) + m_param.cx(),
        m_param.fy() * p_d(1) + m_param.cy();
}
#pragma endregion
} // namespace TVins
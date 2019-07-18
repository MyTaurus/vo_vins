#include "solveVinsHelp.h"
using namespace std;
using namespace Eigen;
using namespace cv;

namespace TVins
{
#pragma region 2d - 2d方法求解Vins
//根据点对求解两帧图像之间的RT关系，这里使用本质矩阵求解法
int SolveRelativeRT_CV(const std::vector<cv::Point2d> &pre_points,const std::vector<cv::Point2d> &cur_points,  calibration &cali, cv::Mat &R, cv::Mat &t)
{
    if (pre_points.size() >= 9)
    {
        //求出基本矩阵
        cv::Mat E = cv::findFundamentalMat(pre_points, cur_points);
        cv::Mat_<double> R1, R2, t1, t2;

        //计算本质矩阵
        Mat essential_matrix;
        essential_matrix = cv::findEssentialMat(pre_points, cur_points, cali.Param().CameraMatrix, RANSAC);

        //根据本质矩阵换算成R、T矩阵
        cv::recoverPose(essential_matrix, pre_points, cur_points, cali.Param().CameraMatrix, R, t);
        
        return 0;
    }

    return -1;
}

int EssentialMatRT_Homography(const vector<pair<Vector3d, Vector3d>> &corres, calibration &cali, cv::Mat &R, cv::Mat t)
{
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
            Mat homography_matrix;
            homography_matrix = cv::findHomography(ll, rr, RANSAC, 3);
            cv::Mat n;
            cv::decomposeHomographyMat(homography_matrix, cali.Param().CameraMatrix, R, t, n);
            return 0;
        }

        return -1;
    }
}

/**
 * @brief 从特征点匹配求homography（normalized DLT）
 * 
 * @param  vP1 归一化后的点, in reference frame
 * @param  vP2 归一化后的点, in current frame
 * @return     单应矩阵
 */
cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F); // 2N*9

    for (int i = 0; i < N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2 * i, 0) = 0.0;
        A.at<float>(2 * i, 1) = 0.0;
        A.at<float>(2 * i, 2) = 0.0;
        A.at<float>(2 * i, 3) = -u1;
        A.at<float>(2 * i, 4) = -v1;
        A.at<float>(2 * i, 5) = -1;
        A.at<float>(2 * i, 6) = v2 * u1;
        A.at<float>(2 * i, 7) = v2 * v1;
        A.at<float>(2 * i, 8) = v2;

        A.at<float>(2 * i + 1, 0) = u1;
        A.at<float>(2 * i + 1, 1) = v1;
        A.at<float>(2 * i + 1, 2) = 1;
        A.at<float>(2 * i + 1, 3) = 0.0;
        A.at<float>(2 * i + 1, 4) = 0.0;
        A.at<float>(2 * i + 1, 5) = 0.0;
        A.at<float>(2 * i + 1, 6) = -u2 * u1;
        A.at<float>(2 * i + 1, 7) = -u2 * v1;
        A.at<float>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3); // v的最后一列
}

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
void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    // 在DecomposeE函数和ReconstructH函数中对t有归一化
    // 这里三角化过程中恢复的3D点深度取决于 t 的尺度，
    // 但是这里恢复的3D点并没有决定单目整个SLAM过程的尺度
    // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变

    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

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
void Normalize(const vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for (int i = 0; i < N; i++)
    {
        meanX += vKeys[i].x;
        meanY += vKeys[i].y;
    }

    meanX = meanX / N;
    meanY = meanY / N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 将所有vKeys点减去中心坐标，使x坐标和y坐标均值分别为0
    for (int i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].x - meanX;
        vNormalizedPoints[i].y = vKeys[i].y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX / N;
    meanDevY = meanDevY / N;

    float sX = 1.0 / meanDevX;
    float sY = 1.0 / meanDevY;

    // 将x坐标和y坐标分别进行尺度缩放，使得x坐标和y坐标的一阶绝对矩分别为1
    for (int i = 0; i < N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0      1    |
    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0, 0) = sX;
    T.at<float>(1, 1) = sY;
    T.at<float>(0, 2) = -meanX * sX;
    T.at<float>(1, 2) = -meanY * sY;
}
#pragma endregion

#pragma region 3d - 3d方法
//ICP方法，如果有两个平面，该方法可以用来使用
void ICP(const vector<Point3f> &pts1, const vector<Point3f> &pts2, cv::Mat &R, cv::Mat &t)
{
    Point3f p1, p2; // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3f(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3f(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f R_12 = U * (V.transpose());
    Eigen::Vector3f t_12 = Eigen::Vector3f(p1.x, p1.y, p1.z) - R_12 * Eigen::Vector3f(p2.x, p2.y, p2.z);
    cv::eigen2cv(R_12, R);
    cv::eigen2cv(t_12, t);
}
#pragma endregion

#pragma region 2d - 3d方法，即解决pnp问题
//这是不用另外处理畸变的方法
void SolvePnp(const vector<Point3f> &p3d, const vector<Point2f> &p2d, cv::Mat &R, cv::Mat &t)
{
    cv::Mat K(3, 3, cv::DataType<double>::type);
    K.eye(3, 3, cv::DataType<double>::type);
    cv::Mat D;

    bool pnp_succ;
    pnp_succ = cv::solvePnP(p3d, p2d, K, D, R, t, 1);
}

#pragma endregion

cv::Point3f  Rmat2Theta(const cv::Mat& rmat)
{

	Point3f theta;

	double r11 = rmat.ptr<double>(0)[0];
	double r12 = rmat.ptr<double>(0)[1];
	double r13 = rmat.ptr<double>(0)[2];
	double r21 = rmat.ptr<double>(1)[0];
	double r22 = rmat.ptr<double>(1)[1];
	double r23 = rmat.ptr<double>(1)[2];
	double r31 = rmat.ptr<double>(2)[0];
	double r32 = rmat.ptr<double>(2)[1];
	double r33 = rmat.ptr<double>(2)[2];

	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
    //旋转顺序为z、y、x
    double thetaz = atan2(r21, r11) / CV_PI * 180;
    double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / CV_PI * 180;
    double thetax = atan2(r32, r33) / CV_PI * 180;
 
    //相机系到世界系的三轴旋转欧拉角，相机坐标系照此旋转后可以与世界坐标系完全平行。
    //旋转顺序为z、y、x
    theta.z = thetaz;
    theta.y = thetay;
    theta.x = thetax;

	return theta;
}

} // namespace TVins
#include "visionEstimator.h"
#include "analogCamera.h"
#include "featureTracker.h"
#include "solveVinsHelp.h"

namespace TVins
{
using namespace std;
using namespace cv;

void VisionEstimator::EstimatorMotion(const cv::Mat &pre_image, const cv::Mat &cur_image, cv::Mat &R, cv::Mat &t)
{
}

void test()
{
    AnalogCamera cam;
    cam.Open();
    cv::Mat pre_image;
    cv::Mat cur_image = cam.GetOneFrame();

    FeatureTracker fTracker(cam.Calibration());

    fTracker.ReadImage(cur_image);
    cv::Mat R_max, t_max;
    R_max = Mat::eye(3, 3, cv::DataType<double>::type);
    t_max = Mat::zeros(3, 1, cv::DataType<double>::type);

    while (true)
    {
        pre_image = cur_image;
        cur_image = cam.GetOneFrame();

        fTracker.ReadImage(cur_image);

        vector<TrackPoint> vec_tp = fTracker.GetTrackPoints();
        std::vector<cv::Point2d> pre_points, cur_points;

        cv::Mat show_image;

        cam.Calibration().Distortion(cur_image, show_image);
        cvtColor(show_image, show_image, COLOR_GRAY2BGR);
        for (size_t i = 0; i < vec_tp.size(); i++)
        {
            if (vec_tp[i].TrackCount > 0)
            {
                Point2d pt1, pt2;
                cam.Calibration().DistortionI(vec_tp[i].PointI_Old, pt1);
                cam.Calibration().DistortionI(vec_tp[i].PointI, pt2);
                pre_points.push_back(pt1);
                cur_points.push_back(pt2);

                line(show_image, pt1, pt2, cv::Scalar(0, 255, 0), 2, 8);
                circle(show_image, pt2, 2, cv::Scalar(0, 0, 255), 2);
            }
        }

        calibration cali = cam.Calibration();
        cv::Mat R, t;

        int nRet = SolveRelativeRT_CV(pre_points, cur_points, cali, R, t);

        if (nRet == 0)
        {
            R_max *= R;
        }

        t_max = R * t_max + t;

        Point3f pd = Rmat2Theta(R_max);

        cout << "x_angle:" << pd.x << ",y_angle:" << pd.y << ",z_angle:" << pd.z << "\n";
        cout << "x:" << t_max.at<double>(0) << ",Y:" << t_max.at<double>(1) << ",Z:" << t_max.at<double>(2) << "\n";

        imshow("image", show_image);
        waitKey(10);
    }
}
} // namespace TVins
#include "analogCamera.h"
using namespace std;
using namespace cv;

//const string IMAGE_PATH = "./data/data/";
const string IMAGE_PATH = "/home/tl/datasets/data_odometry_gray(1)/dataset/sequences/00/image_0/";
const string CALI_FILE = "./data/sensor.yaml";

namespace TVins
{
int AnalogCamera::Open(const std::string &sn)
{
    //FileStorage fileStorage(CALI_FILE, cv::FileStorage::READ);
    // m_calibration.Param().Image_W = 752;
    // m_calibration.Param().Image_H = 480;

    // m_calibration.Param().GenerateCameraMatrix(458.654, 457.296, 367.215, 248.375);
    
    // m_calibration.Param().GenerateDisCoffes(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    m_calibration.Param().Image_W = 1241;
    m_calibration.Param().Image_H = 376;

    m_calibration.Param().GenerateCameraMatrix(718.856, 718.856, 607.192, 185.2157);
    
    m_calibration.Param().GenerateDisCoffes(0, 0, 0, 0);
    
    m_calibration.Param().HasK3 = -1;

    cv::String pattern = IMAGE_PATH + "*.png";
    glob(pattern, image_path, false);
    m_count = 0;
    return 0;
}

int AnalogCamera::Close()
{
    return 0;
}

cv::Mat AnalogCamera::GetOneFrame()
{
    cv::Mat image;
    if (m_count < image_path.size())
    {
        image = imread(image_path[m_count++], CV_8UC1);
    }

    return image;
}
} // namespace TVins
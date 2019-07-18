#include <iostream>
#include <string>
#include "analogCamera.h"
#include "featureTracker.h"
#include "visionEstimator.h"
using namespace std;
using namespace cv;

void tt()
{
    //string path = argv[1];
    TVins::AnalogCamera camera;
    camera.Open();
    TVins::FeatureTracker tracker(camera.Calibration());

    while (true)
    {
        cv::Mat image = camera.GetOneFrame();
        tracker.ReadImage(image);
        std::vector<TVins::TrackPoint> pts = tracker.GetTrackPoints();

        //std::vector<Point2d> points;
        Mat show_image;
        cvtColor(image,show_image, COLOR_GRAY2BGR);
        for (auto &p :pts)
        {
            circle(show_image, p.PointI, 2, Scalar(0, 0, 255), 2, 16);
        }
        
        //camera.Calibration().Distortion(image, image);
        imshow("show image", show_image);
        waitKey(100);
    }

    camera.Close();
}

int main(int argc, char **argv)
{
    TVins::test();
}
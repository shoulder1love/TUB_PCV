//============================================================================
// Name        : Pcv1.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for first PCV assignment
//============================================================================

#ifndef PCV1_INCLUDED
#define PCV1_INCLUDED

#include <opencv2/opencv.hpp>

#include <string>

namespace pcv1 {


void run(const std::string &imageFilename);

bool isPointOnLine(cv::Mat& point, cv::Mat& line, float eps = 1e-5f);

enum GeometryType {
    GEOM_TYPE_POINT,
    GEOM_TYPE_LINE,
};

cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type);
cv::Mat getH(cv::Mat& T, cv::Mat& R, cv::Mat& S);
cv::Mat getScaleMatrix(float lambda);
cv::Mat getRotMatrix(float phi);
cv::Mat getTranslMatrix(float dx, float dy);
cv::Mat getConnectingLine(cv::Mat& p1, cv::Mat& p2);

}


#endif // PCV1_INCLUDED

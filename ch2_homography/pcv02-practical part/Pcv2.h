//============================================================================
// Name        : Pcv2.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for second PCV assignment
//============================================================================
#ifndef PCV2_H
#define PCV2_H

#include <opencv2/opencv.hpp>

namespace pcv2 {

    

enum GeometryType {
    GEOM_TYPE_POINT,
    GEOM_TYPE_LINE,
};


    
// functions to be implemented
// --> please edit ONLY these functions!
    
    
cv::Mat homography2D(cv::Mat &base, cv::Mat &attach);
cv::Mat solve_dlt(cv::Mat &A);
void decondition(cv::Mat &T_base, cv::Mat &T_attach, cv::Mat &H);
cv::Mat getDesignMatrix_homography2D(cv::Mat &base, cv::Mat &attach);
cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type);
cv::Mat getCondition2D(cv::Mat &p);

}

#endif // PCV2_H

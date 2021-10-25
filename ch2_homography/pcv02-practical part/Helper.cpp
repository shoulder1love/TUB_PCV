#include "Helper.h"

#include <vector>
#include <string>


namespace pcv2 {

struct WinInfo { 
    cv::Mat img; 
    std::string name; 
    std::vector<cv::Point2f> pointList; 
};



// mouse call back to get points and draw circles
/*
event	specifies encountered mouse event
x,y	position of mouse pointer
flags	not used here
param	a struct containing used IplImage and window title
*/
void getPointsCB(int event, int x, int y, int flags, void* param){

    // cast to a structure
    WinInfo* win = (WinInfo*) param;

    switch(event){
        // if left mouse button was pressed
        case CV_EVENT_LBUTTONDOWN:{
            // create point representing mouse position
            cv::Point2f p(x,y);
            // draw green point
            cv::circle(win->img, p, 2, cv::Scalar(0, 255, 0), 2);
            // draw green circle
            cv::circle(win->img, p, 15, cv::Scalar(0, 255, 0), 2);
            // update image
            cv::imshow(win->name.c_str(), win->img);
            // add point to point list
            win->pointList.push_back(p);
        }break;
    }
}


int getPoints(const cv::Mat &baseImg, const cv::Mat &attachImg, cv::Mat &points_base, cv::Mat &points_attach)
{
    std::cout << std::endl;
    std::cout << "Please select at least four points by clicking at the corresponding image positions:" << std::endl;
    std::cout << "Firstly click at the point that shall be transformed (within the image to be attached), followed by a click on the corresponding point within the base image" << std::endl;
    std::cout << "Continue until you have collected as many point pairs as you wish" << std::endl;
    std::cout << "Stop the point selection by pressing any key" << std::endl << std::endl;

    
    WinInfo windowInfoBase = {
        baseImg.clone(),
        "Base image",
        {}
    };
    
    WinInfo windowInfoAttach = {
        attachImg.clone(),
        "Image to attach",
        {}
    };
    
    
    // show input images and install mouse callback
    cv::namedWindow( windowInfoBase.name.c_str(), 0 );
    cv::imshow( windowInfoBase.name.c_str(), windowInfoBase.img );
    cv::setMouseCallback(windowInfoBase.name.c_str(), getPointsCB, (void*) &windowInfoBase);
    

    cv::namedWindow( windowInfoAttach.name.c_str(), 0 );
    cv::imshow( windowInfoAttach.name.c_str(), windowInfoAttach.img );
    cv::setMouseCallback(windowInfoAttach.name.c_str(), getPointsCB, (void*) &windowInfoAttach);

    // wait until any key was pressed
    cv::waitKey(0);
    
    cv::destroyWindow( windowInfoBase.name.c_str() );
    cv::destroyWindow( windowInfoAttach.name.c_str() );

    // allocate memory for point-lists (represented as matrix)
    int numOfPoints = windowInfoBase.pointList.size();
    points_base = cv::Mat(3, numOfPoints, CV_32FC1);
    points_attach = cv::Mat(3, numOfPoints, CV_32FC1);
    // read points from global variable, transform them into homogeneous coordinates
    for(int p = 0; p<numOfPoints; p++){
        points_attach.at<float>(0, p) = windowInfoAttach.pointList.at(p).x;
        points_attach.at<float>(1, p) = windowInfoAttach.pointList.at(p).y;
        points_attach.at<float>(2, p) = 1;
        points_base.at<float>(0, p) = windowInfoBase.pointList.at(p).x;
        points_base.at<float>(1, p) = windowInfoBase.pointList.at(p).y;
        points_base.at<float>(2, p) = 1;
    }
    return numOfPoints;
}


cv::Mat stitch(const cv::Mat &base, const cv::Mat &attach, const cv::Mat &H)
{
    // compute corners of warped image
    cv::Mat corners(1, 4, CV_32FC2);
    corners.at<cv::Vec2f>(0, 0) = cv::Vec2f(0,0);
    corners.at<cv::Vec2f>(0, 1) = cv::Vec2f(0,attach.rows);
    corners.at<cv::Vec2f>(0, 2) = cv::Vec2f(attach.cols,0);
    corners.at<cv::Vec2f>(0, 3) = cv::Vec2f(attach.cols,attach.rows);
    perspectiveTransform(corners, corners, H);

    // compute size of resulting image and allocate memory
    float x_start = std::min( std::min( corners.at<cv::Vec2f>(0, 0)[0], corners.at<cv::Vec2f>(0, 1)[0]), (float)0);
    float x_end   = std::max( std::max( corners.at<cv::Vec2f>(0, 2)[0], corners.at<cv::Vec2f>(0, 3)[0]), (float)base.cols);
    float y_start = std::min( std::min( corners.at<cv::Vec2f>(0, 0)[1], corners.at<cv::Vec2f>(0, 2)[1]), (float)0);
    float y_end   = std::max( std::max( corners.at<cv::Vec2f>(0, 1)[1], corners.at<cv::Vec2f>(0, 3)[1]), (float)base.rows);

    // create translation matrix in order to copy both images to correct places
    cv::Mat T = cv::Mat::zeros(3,3,CV_32FC1);
    T.at<float>(0, 0) = 1;
    T.at<float>(1, 1) = 1;
    T.at<float>(2, 2) = 1;
    T.at<float>(0, 2) = -x_start;
    T.at<float>(1, 2) = -y_start;

    // change homography to take necessary translation into account
    T = T * H;
    // warp second image and copy it to output image
    cv::Mat panorama;
    cv::warpPerspective(attach, panorama, T, cv::Size(x_end - x_start + 1, y_end - y_start + 1), CV_INTER_LINEAR);

    // copy base image to correct position within output image
    cv::Mat roi(panorama, cv::Rect(-x_start,-y_start,base.cols, base.rows));
    base.copyTo(roi, base);

    return panorama;
}




}

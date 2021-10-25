#ifndef HELPER_H
#define HELPER_H


#include <opencv2/opencv.hpp>


namespace pcv4 {

/**
 * @brief Displays two images and catches the point pairs marked by left mouse clicks.
 * @details Points will be in homogeneous coordinates.
 * @param img1 The first image
 * @param img2 The second image
 * @param p1 Points within the first image (returned in the matrix by this method)
 * @param p2 Points within the second image (returned in the matrix by this method)
 */
int getPointsManual(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &p1, cv::Mat &p2);

/** 
 * @brief Draws line given in homogeneous representation into image
 * @param img the image to draw into
 * @param a The line parameters
 * @param b The line parameters
 * @param c The line parameters
 */
void drawEpiLine(cv::Mat& img, double a, double b, double c);

}


#endif // HELPER_H

//============================================================================
// Name        : Pcv4.h
// Author      : Ronny Haensch, Andreas Ley
// Version     : 2.0
// Copyright   : -
// Description : header file for the fourth PCV assignment
//============================================================================

#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <string>

namespace pcv4 {

enum GeometryType {
    GEOM_TYPE_POINT,
    GEOM_TYPE_LINE,
};


// functions to be implemented
// --> please edit ONLY these functions!


/**
 * @brief Compute the fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns	the estimated fundamental matrix
 */
cv::Mat getFundamentalMatrix(cv::Mat& p1, cv::Mat& p2);


/**
 * @brief Define the design matrix as needed to compute fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_fundamental(cv::Mat& p1, cv::Mat& p2); 



/**
 * @brief Enforce rank of 2 on fundamental matrix
 * @param F The matrix to be changed
 */
void forceSingularity(cv::Mat& F);

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated fundamental matrix
 */
cv::Mat solve_dlt(cv::Mat& A);

/**
 * @brief Decondition a fundamental matrix that was estimated from conditioned points
 * @param T1 Conditioning matrix of set of 2D image points
 * @param T2 Conditioning matrix of set of 2D image points
 * @param F Conditioned fundamental matrix that has to be un-conditioned (in-place)
 */
void decondition(cv::Mat& T1, cv::Mat& T2, cv::Mat& F);

/**
 * @brief Calculate geometric error of estimated fundamental matrix
 * @details Implement the mean "Sampson distance"
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @returns		geometric error
 */
double getError(cv::Mat p1, cv::Mat p2, cv::Mat& F);

/**
 * @brief Count the number of inliers of an estimated fundamental matrix
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @param threshold Maximal "Sampson distance" to sti9ll be countes as an inlier
 * @returns		Number of inliers
 */
unsigned countInliers(cv::Mat& p1, cv::Mat& p2, cv::Mat& F, float threshold);



cv::Mat estimateFundamentalRANSAC(cv::Mat& p1, cv::Mat& p2, unsigned numIterations);


/**
 * @brief Apply transformation to set of points
 * @param H Matrix representing the transformation
 * @param geomObj Matrix with input objects (one per column)
 * @param type The type of the geometric object (for now: only point and line)
 * @returns Transformed objects (one per column)
 */
cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type);

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix
 */
cv::Mat getCondition2D(cv::Mat& p);

/**
 * @brief Draw epipolar lines into both images
 * @param img1 Structure containing first image
 * @param img2 Structure containing second image
 * @param p1 First point set (points in first image)
 * @param p2 First point set (points in second image)
 * @param F Fundamental matrix (mapping from point in img1 to lines in img2)
 */
void visualize(cv::Mat& img1, cv::Mat& img2, cv::Mat& p1, cv::Mat& p2, cv::Mat& F);


/**
 * @brief Displays two images and catches the point pairs marked by left mouse clicks.
 * @details Points will be in homogeneous coordinates.
 * @param img1 The first image
 * @param img2 The second image
 * @param p1 Points within the first image (returned in the matrix by this method)
 * @param p2 Points within the second image (returned in the matrix by this method)
 */
int getPointsAutomatic(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &p1, cv::Mat &p2);



}

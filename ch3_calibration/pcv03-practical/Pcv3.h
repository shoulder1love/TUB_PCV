//============================================================================
// Name        : Pcv3.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for the third PCV assignment
//============================================================================

#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <string>

namespace pcv3 {

enum GeometryType {
    GEOM_TYPE_POINT,
    GEOM_TYPE_LINE,
};


// functions to be implemented
// --> please edit ONLY these functions!

/**
 * @brief Estimate projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The projection matrix to be computed
 */
cv::Mat calibrate(cv::Mat& points2D, cv::Mat& points3D);

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated projection matrix
 */
cv::Mat solve_dlt(cv::Mat& A);

/**
 * @brief Decondition a projection matrix that was estimated from conditioned point clouds
 * @param T_2D Conditioning matrix of set of 2D image points
 * @param T_3D Conditioning matrix of set of 3D object points
 * @param P Conditioned projection matrix that has to be un-conditioned (in-place)
 */
void decondition(cv::Mat& T_2D, cv::Mat& T_3D, cv::Mat& P);

/**
 * @brief Define the design matrix as needed to compute projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_camera(cv::Mat& points2D, cv::Mat& points3D);

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
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix 
 */
cv::Mat getCondition3D(cv::Mat& p);

/**
 * @brief Extract and prints information about interior and exterior orientation from camera
 * @param P The 3x4 projection matrix
 * @param K Matrix for returning the computed internal calibration
 * @param R Matrix for returning the computed rotation
 * @param info Structure for returning the interpretation such as principal distance
 */
void interprete(cv::Mat &P, cv::Mat &K, cv::Mat &R, ProjectionMatrixInterpretation &info);

}

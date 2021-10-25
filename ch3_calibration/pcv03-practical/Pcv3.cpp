//============================================================================
// Name        : Pcv3.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : Camera calibration
//============================================================================

#include "Pcv3.h"
#include<iostream>

namespace pcv3 {

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix
 */
cv::Mat getCondition2D(cv::Mat& p){
    // TO DO !!!
	// the number of columns and rows
	int cols = p.cols;
	int rows = p.rows;

	//get the coordinate of centre point
	float t[2] = { 0.0 };
	for (int r = 0; r < rows - 1; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			t[r] +=p.at<float>(r, c);
		}
		t[r] = t[r] / cols;
	}

	//get the scaling element
	float s[2] = { 0.0 };
	for (int r = 0; r < rows - 1; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			s[r] += abs(p.at<float>(r, c) - t[r]);
		}
		s[r] = s[r] / cols;
	}

	cv::Mat T = (cv::Mat_<float>(3, 3, CV_32FC1) <<
		1 / s[0], 0, -t[0] / s[0],
		0, 1 / s[1], -t[1] / s[1],
		0, 0, 1);

	return T;
}

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix 
 */
cv::Mat getCondition3D(cv::Mat& p){
    // TO DO !!!
	int cols = p.cols;
	int rows = p.rows;

	//get the coordinate of centre point
	float t[3] = { 0.0 };
	for (int r = 0; r < rows - 1; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			t[r] += p.at<float>(r, c);
		}
		t[r] = t[r] / cols;
	}

	//get the scaling element
	float s[3] = { 0.0 };
	for (int r = 0; r < rows - 1; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			s[r] += abs(p.at<float>(r, c) - t[r]);
		}
		s[r] = s[r] / cols;
	}

	cv::Mat T = (cv::Mat_<float>(4, 4, CV_32FC1) <<
		1 / s[0], 0, 0, -t[0] / s[0],
		0, 1 / s[1], 0, -t[1] / s[1],
		0, 0, 1 / s[2], -t[2] / s[2],
		0, 0, 0, 1);

    return T;
}

/**
 * @brief Apply transformation to set of points
 * @param H Matrix representing the transformation
 * @param geomObj Matrix with input objects (one per column)
 * @param type The type of the geometric object (for now: only point and line)
 * @returns Transformed objects (one per column)
 */
cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type){
    // TO DO !!!
    switch (type) {
        case GEOM_TYPE_POINT:
			return H * geomObj;
        case GEOM_TYPE_LINE:
			return (H.inv().t())*geomObj;
        default:
            throw std::runtime_error("Unhandled case!");
    }
}

/**
 * @brief Define the design matrix as needed to compute projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_camera(cv::Mat& points2D, cv::Mat& points3D){
    // TO DO !!!
	int N = points2D.cols;	//the number of point pairs
	//Create design matrix
	cv::Mat A = cv::Mat(2 * points2D.cols, 12, CV_32FC1);

	//Calculate the design matrix (the number of iteration: N)
	int i = 0;
	for (int j = 0; j < N; j++)
	{
		//W*XT
		A.at<float>(i, 0) = -points2D.at<float>(2, j)*points3D.at<float>(0, j);
		A.at<float>(i, 1) = -points2D.at<float>(2, j)*points3D.at<float>(1, j);
		A.at<float>(i, 2) = -points2D.at<float>(2, j)*points3D.at<float>(2, j);
		A.at<float>(i, 3) = -points2D.at<float>(2, j)*points3D.at<float>(3, j);

		A.at<float>(i, 4) = 0.0;
		A.at<float>(i, 5) = 0.0;
		A.at<float>(i, 6) = 0.0;
		A.at<float>(i, 7) = 0.0;

		A.at<float>(i, 8) = points2D.at<float>(0, j)*points3D.at<float>(0, j);
		A.at<float>(i, 9) = points2D.at<float>(0, j)*points3D.at<float>(1, j);
		A.at<float>(i, 10) = points2D.at<float>(0, j)*points3D.at<float>(2, j);
		A.at<float>(i, 11) = points2D.at<float>(0, j)*points3D.at<float>(3, j);

		A.at<float>(i+1, 0) = 0.0;
		A.at<float>(i+1, 1) = 0.0;
		A.at<float>(i+1, 2) = 0.0;
		A.at<float>(i+1, 3) = 0.0;

		A.at<float>(i+1, 4) = -points2D.at<float>(2, j)*points3D.at<float>(0, j);
		A.at<float>(i+1, 5) = -points2D.at<float>(2, j)*points3D.at<float>(1, j);
		A.at<float>(i+1, 6) = -points2D.at<float>(2, j)*points3D.at<float>(2, j);
		A.at<float>(i+1, 7) = -points2D.at<float>(2, j)*points3D.at<float>(3, j);

		A.at<float>(i+1, 8) = points2D.at<float>(1, j)*points3D.at<float>(0, j);
		A.at<float>(i+1, 9) = points2D.at<float>(1, j)*points3D.at<float>(1, j);
		A.at<float>(i+1, 10) = points2D.at<float>(1, j)*points3D.at<float>(2, j);
		A.at<float>(i+1, 11) = points2D.at<float>(1, j)*points3D.at<float>(3, j);

		i = i + 2;
	}
    return A;
}


/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated projection matrix
 */
cv::Mat solve_dlt(cv::Mat& A){
    // TO DO !!!
	cv::Mat U, W, VT, V;
	cv::SVD::compute(A, W, U, VT);

	V = VT.t();
	float vmin = W.at<float>(0, 0);
	int min_loc;
	for (int i = 1; i < 12; i++)
	{
		if (W.at<float>(i, 0) < vmin)
		{
			std::cout << W.at<float>(i, 0) << std::endl;
			vmin = W.at<float>(i, 0);			//record mininum of eigen value
			min_loc = i;						//record the location of min value (column)
		}
	}
	//Reshape
	cv::Mat P = cv::Mat::eye(3, 4, CV_32FC1);
	P.at<float>(0, 0) = V.at<float>(0, min_loc);
	P.at<float>(0, 1) = V.at<float>(1, min_loc);
	P.at<float>(0, 2) = V.at<float>(2, min_loc);
	P.at<float>(0, 3) = V.at<float>(3, min_loc);
	P.at<float>(1, 0) = V.at<float>(4, min_loc);
	P.at<float>(1, 1) = V.at<float>(5, min_loc);
	P.at<float>(1, 2) = V.at<float>(6, min_loc);
	P.at<float>(1, 3) = V.at<float>(7, min_loc);
	P.at<float>(2, 0) = V.at<float>(8, min_loc);
	P.at<float>(2, 1) = V.at<float>(9, min_loc);
	P.at<float>(2, 2) = V.at<float>(10, min_loc);
	P.at<float>(2, 3) = V.at<float>(11, min_loc);

    return P;
}

/**
 * @brief Decondition a projection matrix that was estimated from conditioned point clouds
 * @param T_2D Conditioning matrix of set of 2D image points
 * @param T_3D Conditioning matrix of set of 3D object points
 * @param P Conditioned projection matrix that has to be un-conditioned (in-place)
 */
void decondition(cv::Mat& T_2D, cv::Mat& T_3D, cv::Mat& P){
    // TO DO !!!
	P = (T_2D.inv())*P*T_3D;
}


/**
 * @brief Estimate projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The projection matrix to be computed
 */
cv::Mat calibrate(cv::Mat& points2D, cv::Mat& points3D){
    // TO DO !!!
	cv::Mat T_2D=getCondition2D(points2D);
	cv::Mat T_3D=getCondition3D(points3D);

	cv::Mat MT2D = applyH(points2D, T_2D, GEOM_TYPE_POINT);
	cv::Mat MT3D = applyH(points3D, T_3D, GEOM_TYPE_POINT);

	cv::Mat A = getDesignMatrix_camera(MT2D, MT3D);

	cv::Mat P = solve_dlt(A);

	decondition(T_2D, T_3D, P);

    return P;
}



/**
 * @brief Extract and prints information about interior and exterior orientation from camera
 * @param P The 3x4 projection matrix, only "input" to this function
 * @param K Matrix for returning the computed internal calibration
 * @param R Matrix for returning the computed rotation
 * @param info Structure for returning the interpretation such as principal distance
 */
void interprete(cv::Mat &P, cv::Mat &K, cv::Mat &R, ProjectionMatrixInterpretation &info){
    // TO DO !!!

    /*
     * **NOTE**
     * I changed things a little bit compared to what I presented in the tutorial.
     * To prevent trouple with returning things in the wrong order, 
     * the stuff you need to return is now explicitely named (see below).
     * Return the internal calibration in K, the rotation in R, and the 
     * the interpretation in the corresponding fields of the info struct.
     */
	K = cv::Mat(3, 3, CV_32FC1);
	R = cv::Mat(3, 3, CV_32FC1);

	//Create M matrix
	cv::Mat M = cv::Mat::zeros(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			M.at<float>(i, j) = P.at<float>(i, j);
		}
	}

	//RQ decomposition
	cv::RQDecomp3x3(M, K, R);
	K = K / (K.at<float>(2, 2));	//normalization
    
	//-------------------Calculating camera(K) matrix----------------------
    // Principal distance or focal length
	info.principalDistance = K.at<float>(0, 0);
    
    // Skew as an angle and in degrees
	info.skew = atan(1 / (-K.at<float>(0, 1) / K.at<float>(0, 0))) * 180 / 3.141592653;
    
    // Aspect ratio of the pixels
	info.aspectRatio = K.at<float>(1, 1) / K.at<float>(0, 0);
    
    // Location of principal point in image (pixel) coordinates
    info.principalPoint[0] = K.at<float>(0, 2);
    info.principalPoint[1] = K.at<float>(1, 2);
    //---------------------------end----------------------------------------

	//-------------------Calculating Rotation(R) matrix---------------------
	//Decomposition of P
	cv::Mat q1, q2, q3, q4;			//all are 3*1 matrixs
	q1 = cv::Mat::zeros(3, 1, CV_32FC1);
	q2 = cv::Mat::zeros(3, 1, CV_32FC1);
	q3 = cv::Mat::zeros(3, 1, CV_32FC1);
	q4 = cv::Mat::zeros(3, 1, CV_32FC1);

	//Assignment
	for (int i = 0; i < 3; i++)
	{
		q1.at<float>(i, 0) = P.at<float>(0, i);
		q2.at<float>(i, 0) = P.at<float>(1, i);
		q3.at<float>(i, 0) = P.at<float>(2, i);
		q4.at<float>(i, 0) = P.at<float>(i, 3);
	}

	//Calculating the camera location matrix
	cv::Mat C = -M.inv()*q4;
    // Camera rotation angle 1/3
	info.omega = atan2(-R.at<float>(2, 1), R.at<float>(2, 2)) * 180 / 3.141592653;
    
    // Camera rotation angle 2/3
	info.phi = asin(R.at<float>(2, 0)) * 180 / 3.141592653;
    
    // Camera rotation angle 3/3
	info.kappa = atan2(-R.at<float>(1, 0), R.at<float>(0, 0)) * 180 / 3.141592653;
    
    // 3D camera location in world coordinates
	info.cameraLocation[0] = C.at<float>(0, 0);
    info.cameraLocation[1] = C.at<float>(1, 0);
    info.cameraLocation[2] = C.at<float>(2, 0);
    
}




}

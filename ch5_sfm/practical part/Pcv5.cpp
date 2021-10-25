//============================================================================
// Name        : Pcv5.cpp
// Author      : Andreas Ley
// Version     : 1.0
// Copyright   : -
// Description : Bundle Adjustment
//============================================================================

#include "Pcv5.h"

#include <random>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

//#define VERBOSE_MODE

namespace pcv5 {
    
    
/**
 * @brief Apply transformation to set of points
 * @param H Matrix representing the transformation
 * @param geomObj Matrix with input objects (one per column)
 * @param type The type of the geometric object (for now: only point and line)
 * @returns Transformed objects (one per column)
 */
cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type)
{
    // TO DO !!!
    switch (type) {
        case GEOM_TYPE_POINT:
            return H*geomObj;
        case GEOM_TYPE_LINE:
            return H.inv().t()*geomObj;
        default:
            throw std::runtime_error("Unhandled case!");
    }
}

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix
 */
cv::Mat getCondition2D(cv::Mat& p)
{
	int cols = p.cols;
	int rows = p.rows;

	//get the coordinate of centre point
	float t[2] = { 0.0 };
	for (int r = 0; r < rows - 1; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			t[r] += p.at<float>(r, c);
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
 * @brief Compute the fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns	the estimated fundamental matrix
 */
cv::Mat getFundamentalMatrix(cv::Mat& p1, cv::Mat& p2) {

    // TO DO !!!
	cv::Mat p1_H = getCondition2D(p1);
	cv::Mat p2_H = getCondition2D(p2);

	cv::Mat p1_ = applyH(p1, p1_H, GEOM_TYPE_POINT);
	cv::Mat p2_ = applyH(p2, p2_H, GEOM_TYPE_POINT);

	cv::Mat A_design = getDesignMatrix_fundamental(p1_, p2_);

	cv::Mat F = solve_dlt_F(A_design);

	forceSingularity(F);
	decondition_F(p1_H, p2_H, F);

	return F;
}


/**
 * @brief Define the design matrix as needed to compute fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_fundamental(cv::Mat& p1, cv::Mat& p2)
{
	int N = p1.cols;	//the number of point pairs
	if (N == 8) {
		cv::Mat A = cv::Mat::zeros(9, 9, CV_32FC1);

		//Calculate the design matrix (the number of iteration:N)
		for (int i = 0; i < N; i++) {
			A.at<float>(i, 0) = p1.at<float>(0, i)*p2.at<float>(0, i);
			A.at<float>(i, 1) = p1.at<float>(1, i)*p2.at<float>(0, i);
			A.at<float>(i, 2) = p2.at<float>(0, i);
			A.at<float>(i, 3) = p1.at<float>(0, i)*p2.at<float>(1, i);
			A.at<float>(i, 4) = p1.at<float>(1, i)*p2.at<float>(1, i);
			A.at<float>(i, 5) = p2.at<float>(1, i);
			A.at<float>(i, 6) = p1.at<float>(0, i);
			A.at<float>(i, 7) = p1.at<float>(1, i);
			A.at<float>(i, 8) = 1;
		}
		return A;
	}
	else {
		cv::Mat A = cv::Mat::zeros(p1.cols, 9, CV_32F);
		for (int i = 0; i < N; i++) {
			A.at<float>(i, 0) = p1.at<float>(0, i)*p2.at<float>(0, i);
			A.at<float>(i, 1) = p1.at<float>(1, i)*p2.at<float>(0, i);
			A.at<float>(i, 2) = p2.at<float>(0, i);
			A.at<float>(i, 3) = p1.at<float>(0, i)*p2.at<float>(1, i);
			A.at<float>(i, 4) = p1.at<float>(1, i)*p2.at<float>(1, i);
			A.at<float>(i, 5) = p2.at<float>(1, i);
			A.at<float>(i, 6) = p1.at<float>(0, i);
			A.at<float>(i, 7) = p1.at<float>(1, i);
			A.at<float>(i, 8) = 1;
		}
		return A;
	}
}


/**
 * @brief Define the design matrix as needed to compute projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_camera(cv::Mat& points2D, cv::Mat& points3D){
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

		A.at<float>(i + 1, 0) = 0.0;
		A.at<float>(i + 1, 1) = 0.0;
		A.at<float>(i + 1, 2) = 0.0;
		A.at<float>(i + 1, 3) = 0.0;

		A.at<float>(i + 1, 4) = -points2D.at<float>(2, j)*points3D.at<float>(0, j);
		A.at<float>(i + 1, 5) = -points2D.at<float>(2, j)*points3D.at<float>(1, j);
		A.at<float>(i + 1, 6) = -points2D.at<float>(2, j)*points3D.at<float>(2, j);
		A.at<float>(i + 1, 7) = -points2D.at<float>(2, j)*points3D.at<float>(3, j);

		A.at<float>(i + 1, 8) = points2D.at<float>(1, j)*points3D.at<float>(0, j);
		A.at<float>(i + 1, 9) = points2D.at<float>(1, j)*points3D.at<float>(1, j);
		A.at<float>(i + 1, 10) = points2D.at<float>(1, j)*points3D.at<float>(2, j);
		A.at<float>(i + 1, 11) = points2D.at<float>(1, j)*points3D.at<float>(3, j);

		i = i + 2;
	}
	return A;
}

/**
 * @brief Enforce rank of 2 on fundamental matrix
 * @param F The matrix to be changed
 */
void forceSingularity(cv::Mat& F)
{
    // TO DO !!!
	cv::Mat U, V, VT, W;
	cv::SVD::compute(F, W, U, VT);

	//decrease rank of diagnal matrix
	W.at<float>(2, 0) = 0.0;
	cv::Mat W_new = cv::Mat::zeros(3, 3, CV_32FC1);
	W_new.at<float>(0, 0) = W.at<float>(0, 0);
	W_new.at<float>(1, 1) = W.at<float>(1, 0);
	W_new.at<float>(2, 2) = W.at<float>(2, 0);
	F = U * W_new*VT;
}

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated fundamental matrix
 */
cv::Mat solve_dlt_F(cv::Mat& A)
{
	cv::Mat U, W, VT, V;
	cv::SVD::compute(A, W, U, VT);

	V = VT.t();
	float vmin = W.at<float>(0, 0);
	int min_loc = 0;
	for (int i = 1; i < 9; i++)
	{
		if (W.at<float>(i, 0) < vmin)
		{
			vmin = W.at<float>(i, 0);			//record mininum of eigen value
			min_loc = i;						//record the location of min value (column)
		}
	}
	//Reshape
	cv::Mat F = cv::Mat::eye(3, 3, CV_32FC1);
	F.at<float>(0, 0) = V.at<float>(0, 8);
	F.at<float>(0, 1) = V.at<float>(1, 8);
	F.at<float>(0, 2) = V.at<float>(2, 8);
	F.at<float>(1, 0) = V.at<float>(3, 8);
	F.at<float>(1, 1) = V.at<float>(4, 8);
	F.at<float>(1, 2) = V.at<float>(5, 8);
	F.at<float>(2, 0) = V.at<float>(6, 8);
	F.at<float>(2, 1) = V.at<float>(7, 8);
	F.at<float>(2, 2) = V.at<float>(8, 8);

	return F;
}

/**
 * @brief Decondition a fundamental matrix that was estimated from conditioned points
 * @param T1 Conditioning matrix of set of 2D image points
 * @param T2 Conditioning matrix of set of 2D image points
 * @param F Conditioned fundamental matrix that has to be un-conditioned (in-place)
 */
void decondition_F(cv::Mat& T1, cv::Mat& T2, cv::Mat& F)
{
    // TO DO !!!
	F = (T2.t())*F*T1;
}

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated projection matrix
 */
cv::Mat solve_dlt_P(cv::Mat& A){

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
void decondition_P(cv::Mat& T_2D, cv::Mat& T_3D, cv::Mat& P){
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
	cv::Mat T_2D = getCondition2D(points2D);
	cv::Mat T_3D = getCondition3D(points3D);

	cv::Mat MT2D = applyH(points2D, T_2D, GEOM_TYPE_POINT);
	cv::Mat MT3D = applyH(points3D, T_3D, GEOM_TYPE_POINT);

	cv::Mat A = getDesignMatrix_camera(MT2D, MT3D);

	cv::Mat P = solve_dlt_P(A);

	decondition_P(T_2D, T_3D, P);

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


/**
 * @brief Calculate geometric error of estimated fundamental matrix
 * @details Implement the mean "Sampson distance"
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @returns		geometric error
 */
double getError(cv::Mat p1, cv::Mat p2, cv::Mat& F)
{
    // TO DO !!!
	double sum = 0;
	for (int i = 0; i < p1.cols; ++i)
	{
		const Mat& x = p1.col(i);
		const Mat& x_ = p2.col(i);
		const Mat F_x = F * x;
		const Mat F_T_x_ = F.t()*x_;
		//sampson distance
		sum = sum + pow(Mat(x_.t() * F_x).at<float>(0, 0), 2) /
			(pow(F_x.at<float>(0, 0), 2)
				+ pow(F_x.at<float>(1, 0), 2)
				+ pow(F_T_x_.at<float>(0, 0), 2)
				+ pow(F_T_x_.at<float>(1, 0), 2)
				);
	}
	int N = p1.cols;

	return 1.0 / N * sum;
}

/**
 * @brief Count the number of inliers of an estimated fundamental matrix
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @param threshold Maximal "Sampson distance" to sti9ll be countes as an inlier
 * @returns		Number of inliers
 */
unsigned countInliers(cv::Mat& p1, cv::Mat& p2, cv::Mat& F, float threshold)
{
    // TO DO !!!
	float d = 0.0;
	int count = 0;
	for (int i = 0; i < p1.cols; ++i)
	{
		const Mat& x = p1.col(i);
		const Mat& x_ = p2.col(i);
		const Mat F_x = F * x;
		const Mat F_T_x_ = F.t()*x_;

		d = pow(Mat(x_.t() * F_x).at<float>(0, 0), 2) /
			(pow(F_x.at<float>(0, 0), 2)
				+ pow(F_x.at<float>(1, 0), 2)
				+ pow(F_T_x_.at<float>(0, 0), 2)
				+ pow(F_T_x_.at<float>(1, 0), 2)
				);
		if (d <= threshold) {
			count = count + 1;
		}
	}
	return count;
}

/**
 * @brief Estimate the fundamental matrix robustly using RANSAC
 * @param p1 first set of points
 * @param p2 second set of points
 * @param numIterations How many subsets are to be evaluated
 * @returns The fundamental matrix
 */
cv::Mat estimateFundamentalRANSAC(cv::Mat& p1, cv::Mat& p2, unsigned numIterations)
{
    // TO DO !!!
	const unsigned subsetSize = 8;

	// TO DO !!!
	//Get Fundamental Matrix
	Mat F, F_;
	unsigned n = 0;
	float threshold = 2.0;		//Set a threshold: manual
	unsigned sum_in = 0;		//the number of inliers
	unsigned inliers = 0;		//initial value of inliers

	while (n < numIterations)
	{
		random_device rd;
		std::mt19937 rng(rd());
		std::uniform_int_distribution<unsigned> uniformDist(0, p1.cols - 1);

		//pick a random subset of 8 points
		Mat p1_ = Mat::zeros(3, subsetSize, CV_32FC1);		//subset of 8 points in image1
		Mat p2_ = Mat::zeros(3, subsetSize, CV_32FC1);		//subset of 8 points in image2

															//Traverse 8 points in first and second images
		for (int i = 0; i < subsetSize; i++)
		{
			// Draw a random point index with unsigned index = uniformDist(rng);
			unsigned index = uniformDist(rng);			//the index'th point
			p1_.at<float>(0, i) = p1.at<float>(0, index);
			p1_.at<float>(1, i) = p1.at<float>(1, index);
			p1_.at<float>(2, i) = 1.0;

			p2_.at<float>(0, i) = p2.at<float>(0, index);
			p2_.at<float>(1, i) = p2.at<float>(1, index);
			p2_.at<float>(2, i) = 1.0;
		}

		//estimate fundamental matrix of subset
		F = getFundamentalMatrix(p1_, p2_);

		//count total number of inliers
		sum_in = countInliers(p1, p2, F, threshold);
		if (sum_in > inliers)
		{
			F_ = F;
			inliers = sum_in;
		}
		n++;
	}

	return F_;
}



/**
 * @brief Estimate the fundamental matrix robustly using RANSAC
 * @param p1 first set of points
 * @param p2 second set of points
 * @param numIterations How many subsets are to be evaluated
 * @returns The fundamental matrix
 */
cv::Mat estimateProjectionRANSAC(cv::Mat& points2D, cv::Mat& points3D, unsigned numIterations)
{
    const unsigned subsetSize = 6;

    std::mt19937 rng;
    std::uniform_int_distribution<unsigned> uniformDist(0, points2D.cols-1);
    // Draw a random point index with unsigned index = uniformDist(rng);
    
    cv::Mat bestP;
    unsigned bestInliers = 0;
    
    cv::Mat p2D_subset(3, subsetSize, CV_32FC1);
    cv::Mat p3D_subset(4, subsetSize, CV_32FC1);
    for (unsigned iter = 0; iter < numIterations; iter++) {
        for (unsigned j = 0; j < subsetSize; j++) {
            unsigned index = uniformDist(rng);
            p2D_subset.at<float>(0, j) = points2D.at<float>(0, index);
            p2D_subset.at<float>(1, j) = points2D.at<float>(1, index);
            p2D_subset.at<float>(2, j) = points2D.at<float>(2, index);

            p3D_subset.at<float>(0, j) = points3D.at<float>(0, index);
            p3D_subset.at<float>(1, j) = points3D.at<float>(1, index);
            p3D_subset.at<float>(2, j) = points3D.at<float>(2, index);
            p3D_subset.at<float>(3, j) = points3D.at<float>(3, index);
        }
        
        cv::Mat P = calibrate(p2D_subset, p3D_subset);

        const float thresh = 20.0f;
        unsigned numInliers = 0;
        cv::Mat projected = P * points3D;
        for (unsigned i = 0; i < points2D.cols; i++) {
            if (projected.at<float>(2, i) > 0.0f) // in front
                if ((std::abs(points2D.at<float>(0, i) - projected.at<float>(0, i)/projected.at<float>(2, i)) < thresh) &&
                    (std::abs(points2D.at<float>(1, i) - projected.at<float>(1, i)/projected.at<float>(2, i)) < thresh))
                    numInliers++;
        }

        if (numInliers > bestInliers) {
            bestInliers = numInliers;
            bestP = P;
        }
    }
    
    return bestP;
}


// triangulates given set of image points based on projection matrices
/*
P1	projection matrix of first image
P2	projection matrix of second image
x1	image point set of first image
x2	image point set of second image
return	triangulated object points
*/
Mat linearTriangulation(const Mat& P1, const Mat& P2, const Mat& x1, const Mat& x2){

  Mat X = Mat(4, x1.cols, CV_32FC1);
  
  // allocate memory for design matrix
  Mat A = Mat(4, 4, CV_32FC1);
  
  for(int i=0; i < x1.cols; i++){

      // create design matrix
      // first row	x1(0, i) * P1(2, :) - P1(0, :)
      A.at<float>(0, 0) = x1.at<float>(0, i) * P1.at<float>(2, 0) - P1.at<float>(0, 0);
      A.at<float>(0, 1) = x1.at<float>(0, i) * P1.at<float>(2, 1) - P1.at<float>(0, 1);
      A.at<float>(0, 2) = x1.at<float>(0, i) * P1.at<float>(2, 2) - P1.at<float>(0, 2);
      A.at<float>(0, 3) = x1.at<float>(0, i) * P1.at<float>(2, 3) - P1.at<float>(0, 3);
      // second row	x1(1, i) * P1(2, :) - P1(1, :)
      A.at<float>(1, 0) = x1.at<float>(1, i) * P1.at<float>(2, 0) - P1.at<float>(1, 0);
      A.at<float>(1, 1) = x1.at<float>(1, i) * P1.at<float>(2, 1) - P1.at<float>(1, 1);
      A.at<float>(1, 2) = x1.at<float>(1, i) * P1.at<float>(2, 2) - P1.at<float>(1, 2);
      A.at<float>(1, 3) = x1.at<float>(1, i) * P1.at<float>(2, 3) - P1.at<float>(1, 3);
      // third row	x2(0, i) * P2(3, :) - P2(0, :)
      A.at<float>(2, 0) = x2.at<float>(0, i) * P2.at<float>(2, 0) - P2.at<float>(0, 0);
      A.at<float>(2, 1) = x2.at<float>(0, i) * P2.at<float>(2, 1) - P2.at<float>(0, 1);
      A.at<float>(2, 2) = x2.at<float>(0, i) * P2.at<float>(2, 2) - P2.at<float>(0, 2);
      A.at<float>(2, 3) = x2.at<float>(0, i) * P2.at<float>(2, 3) - P2.at<float>(0, 3);
      // first row	x2(1, i) * P2(3, :) - P2(1, :)
      A.at<float>(3, 0) = x2.at<float>(1, i) * P2.at<float>(2, 0) - P2.at<float>(1, 0);
      A.at<float>(3, 1) = x2.at<float>(1, i) * P2.at<float>(2, 1) - P2.at<float>(1, 1);
      A.at<float>(3, 2) = x2.at<float>(1, i) * P2.at<float>(2, 2) - P2.at<float>(1, 2);
      A.at<float>(3, 3) = x2.at<float>(1, i) * P2.at<float>(2, 3) - P2.at<float>(1, 3);

      cv::SVD svd(A);
      Mat tmp = svd.vt.row(3).t();

      // set triangulated object point
      X.at<float>(0, i) = tmp.at<float>(0)/tmp.at<float>(3);
      X.at<float>(1, i) = tmp.at<float>(1)/tmp.at<float>(3);
      X.at<float>(2, i) = tmp.at<float>(2)/tmp.at<float>(3);
      X.at<float>(3, i) = 1;

  }
  
  return X;
  
}


/**
 * @brief Given an internal calibration and point pairs, estimate the camera pose of the second camera if the first is in the world space origin.
 * @param K Internal calibration of both cameras
 * @param p1 Points of first camera
 * @param p2 Points of second camera
 * @returns External calibration of second camera
 */
cv::Mat computeCameraPose(const cv::Mat &K, const cv::Mat &p1, const cv::Mat &p2)
{
    // TO DO !!!

    // Compute "calibrated" versions of p1 and p2
	Mat p1_ = K.inv() * p1;
	Mat p2_ = K.inv() * p2;

    // Compute E' (E=K2^T*F*K1)
	Mat E = getFundamentalMatrix(p1_, p2_);
    
    // In the "calibrated" space, P1 is just the identity matrix and P2 is just the external orientation
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
    cv::Mat P2[4];

    // Compute 4 possible versions of P2
	Mat U, D, VT;
	SVD::compute(E, D, U, VT);		//SVD decomposition

	Mat t = Mat::zeros(3, 1, CV_32FC1);		//B in .pdf
	t.at<float>(0, 0) = U.at<float>(0, 2);
	t.at<float>(1, 0) = U.at<float>(1, 2);
	t.at<float>(2, 0) = U.at<float>(2, 2);
	Mat W = Mat::zeros(3, 3, CV_32FC1);
	W.at<float>(0, 1) = -1.0;
	W.at<float>(1, 0) = 1.0;
	W.at<float>(2, 2) = 1.0;
	//Calculate R = U*W*VT [Rotation matrix]
	Mat R1 = U * W*VT;
	Mat R2 = U * (W.t())*VT;

	//4 cases of P matrix
	hconcat(R1, -t, P2[0]);
	hconcat(R1, t, P2[1]);
	hconcat(R2, -t, P2[2]);
	hconcat(R2, t, P2[3]);
    
    unsigned best = 0;
	int most = 0;
	Mat X;
    // Find the variant of P2, which has the most points in front of both cameras
	for (int i = 0; i < 4; i++)
	{
		unsigned n = 0;
		Mat P_;
		invert(P2[i], P_, cv::DECOMP_SVD);		//Get the Pseudo-inverse of P matrix
												//X=P.inv()*x
		Mat X0 = P_ * p1_;
		//Calculate the most points
		for (int j = 0; j < X0.cols; j++)
		{
			//Get the number of points which is the in front of image
			if (X0.at<float>(2, j) > 0)
			{
				n++;
			}


		}
		if (n > most)
		{
			most = n;
			X = X0.clone();
			best ++;
		}
	}

    cv::Mat H = cv::Mat::eye(4, 4, CV_32F);
    H(cv::Range(0, 3), cv::Range(0, 4)) = P2[best] * 1.0f;
    return H;
}


void BundleAdjustment::BAState::computeResiduals(float *residuals) const
{
    unsigned rIdx = 0;
    for (unsigned camIdx = 0; camIdx < m_cameras.size(); camIdx++) {
        const auto &calibState = m_internalCalibs[m_scene.cameras[camIdx].internalCalibIdx];
        const auto &cameraState = m_cameras[camIdx];
        
		// TO DO !!!
		// Compute 3x4 camera matrix (composition of internal and external calibration)
        // Internal calibration is calibState.K
        // External calibration is dropLastRow(cameraState.H)
		Matrix<3, 4> P = calibState.K*dropLastRow(cameraState.H);
        
        for (const KeyPoint &kp : m_scene.cameras[camIdx].keypoints) {
            const auto &trackState = m_tracks[kp.trackIdx];
			// TO DO !!!
			// Using P, compute the homogeneous position of the track in the image (world space position is trackState.location)
			Vector<3> projection = P * trackState.location;			//[3,4]*[4,1]=[3,1]
            
			// TO DO !!!
			// Compute the euclidean position of the track
            // Elements of projection can be accessed with projection(0), projection(1), ...
			Matrix<2, 1> eucl_track;		//euclidean position of the track
			eucl_track(0, 0) = projection(0) / projection(2);
			eucl_track(1, 0) = projection(1) / projection(2);
            
			// TO DO !!!
			// Compute the residuals: the difference between computed position and real position (kp.location(0) and kp.location(1))
            // Compute and store the residual in x direction multiplied by kp.weight
			residuals[rIdx++] = kp.weight*(kp.location(0) - eucl_track(0, 0));
            // Compute and store the residual in y direction multiplied by kp.weight
			residuals[rIdx++] = kp.weight*(kp.location(1) - eucl_track(1, 0));
        }
    }
}

void BundleAdjustment::BAState::computeJacobiMatrix(JacobiMatrix *dst) const
{
    BAJacobiMatrix &J = dynamic_cast<BAJacobiMatrix&>(*dst);
    
    unsigned rIdx = 0;
    for (unsigned camIdx = 0; camIdx < m_cameras.size(); camIdx++) {
        const auto &calibState = m_internalCalibs[m_scene.cameras[camIdx].internalCalibIdx];	//第几张影像的相机矩阵的第几组内方位元素
        const auto &cameraState = m_cameras[camIdx];											//第几个相机矩阵
        
        for (const KeyPoint &kp : m_scene.cameras[camIdx].keypoints) {				//第几张影像的相机关键点
            const auto &trackState = m_tracks[kp.trackIdx];
            
            // dropLastRow(cameraState.H) is the upper 3x4 part of the external calibration
            // calibState.K is the internal calbration
            // trackState.location is the 3D location of the track in homogeneous coordinates

            // TO DO !!!
            // Compute the positions in before and after the internal calibration.
            // The multiplication operator "*" works as one would suspect.
			
			Matrix<3,4> H = dropLastRow(cameraState.H);		//H(4*4->3*4)

			Vector<3> v = H * trackState.location;
			
			Vector<3> u = calibState.K*v;
            
            
            Matrix<2, 3> J_hom2eucl;
            // TO DO !!!
            // How do the euclidean image positions change when the homogeneous image positions change?
            
			//Jocobi of conversion to Euclidean
			J_hom2eucl(0, 0) = 1.0f / u(2);
			J_hom2eucl(0, 1) = 0.0f;
			J_hom2eucl(0, 2) = -u(0) / (u(2)*u(2));
			J_hom2eucl(1, 0) = 0.0f;
			J_hom2eucl(1, 1) = 1.0f / u(2);
			J_hom2eucl(1, 2) = -u(1) / (u(2)*u(2));
            
            
            Matrix<3, 3> du_dDeltaK;
 
            // TO DO !!!
            // How do homogeneous image positions change when the internal calibration is changed (the 3 update parameters)?
			du_dDeltaK(0, 0) = v(0)*calibState.K(0, 0);
			du_dDeltaK(0, 1) = v(2)*calibState.K(0, 2);
			du_dDeltaK(0, 2) = 0.0;
			du_dDeltaK(1, 0) = v(1)*calibState.K(1, 1);
			du_dDeltaK(1, 1) = 0.0;
            du_dDeltaK(1, 2) = v(2)*calibState.K(1, 2);
			du_dDeltaK(2, 0) = 0.0;
			du_dDeltaK(2, 1) = 0.0;
			du_dDeltaK(2, 2) = 0.0;

            
            
            // TO DO !!!
            // Using the above, how do the euclidean image positions change when the internal calibration is changed (the 3 update parameters)?
            // Remember to include the weight of the keypoint (kp.weight)
            // The multiplication operator "*" works as one would suspect
			J.m_rows[rIdx].J_internalCalib = J_hom2eucl * du_dDeltaK * kp.weight;		//[2,3]*[3,3]=[2,3]
            
            // TO DO !!!
            // How do the euclidean image positions change when the tracks are moving in eye space/camera space (the vector "v" in the slides)?
            // The multiplication operator "*" works as one would suspect

			Matrix<2, 3> J_v2eucl = J_hom2eucl * calibState.K;	//自己改的！！！！！

            Matrix<3, 6> dv_dDeltaH;
            // TO DO !!!
            // How do tracks move in eye space (vector "v" in slides) when the parameters of the camera are changed?
            
			dv_dDeltaH(0, 0) = 0.0;
			dv_dDeltaH(0, 1) = v(2);
			dv_dDeltaH(0, 2) = -v(1);
			dv_dDeltaH(0, 3) = trackState.location(3);
			dv_dDeltaH(0, 4) = 0.0;
			dv_dDeltaH(0, 5) = 0.0;
			dv_dDeltaH(1, 0) = -v(2);
			dv_dDeltaH(1, 1) = 0.0;
			dv_dDeltaH(1, 2) = v(0);
            dv_dDeltaH(1, 3) = 0.0;
			dv_dDeltaH(1, 4) = trackState.location(3);
			dv_dDeltaH(1, 5) = 0.0;
			dv_dDeltaH(2, 0) = v(1);
			dv_dDeltaH(2, 1) = -v(0);
			dv_dDeltaH(2, 2) = 0.0;
			dv_dDeltaH(2, 3) = 0.0;
			dv_dDeltaH(2, 4) = 0.0;
            dv_dDeltaH(2, 5) = trackState.location(3);
            
            
            // TO DO !!!
            // How do the euclidean image positions change when the external calibration is changed (the 6 update parameters)?
            // Remember to include the weight of the keypoint (kp.weight)
            // The multiplication operator "*" works as one would suspect
			J.m_rows[rIdx].J_camera = J_v2eucl * dv_dDeltaH * kp.weight;	//[2,3]
            
            
            // TO DO !!!
            // How do the euclidean image positions change when the tracks are moving in world space (the x, y, z, and w before the external calibration)?
            // The multiplication operator "*" works as one would suspect. You can use dropLastRow(...) to drop the last row of a matrix.
			Matrix<2, 4> J_worldSpace2eucl = J_v2eucl * dropLastRow(cameraState.H);

            // TO DO !!!
            // How do the euclidean image positions change when the tracks are changed. 
            // This is the same as above, except it should also include the weight of the keypoint (kp.weight)
			J.m_rows[rIdx].J_track = J_worldSpace2eucl * kp.weight;
            
            rIdx++;
        }
    }
}

void BundleAdjustment::BAState::update(const float *update, State *dst) const
{
    BAState &state = dynamic_cast<BAState &>(*dst);
    state.m_internalCalibs.resize(m_internalCalibs.size());
    state.m_cameras.resize(m_cameras.size());
    state.m_tracks.resize(m_tracks.size());
    
    unsigned intCalibOffset = 0;
    for (unsigned i = 0; i < m_internalCalibs.size(); i++) {
        state.m_internalCalibs[i].K = m_internalCalibs[i].K;

		//TO DO !!!
		/*
		 * Modify the new matrix K
		 * 
		 * m_internalCalibs[i].K is the old matrix, state.m_internalCalibs[i].K is the new matrix.
		 * 
		 * update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 0] is how much the focal length is supposed to change (scaled by the old focal length)
		 * update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 1] is how much the principal point is supposed to shift in x direction (scaled by the old x position of the principal point)
		 * update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 2] is how much the principal point is supposed to shift in y direction (scaled by the old y position of the principal point)
		 */
		state.m_internalCalibs[i].K(0, 0) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 0] * m_internalCalibs[i].K(0, 0);
		state.m_internalCalibs[i].K(0, 2) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 1] * m_internalCalibs[i].K(0, 2);
		state.m_internalCalibs[i].K(1, 1) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 0] * m_internalCalibs[i].K(1, 1);
		state.m_internalCalibs[i].K(1, 2) += update[intCalibOffset + i * NumUpdateParams::INTERNAL_CALIB + 2] * m_internalCalibs[i].K(1, 2);
		state.m_internalCalibs[i].K(2, 2) = 1;
	}
    unsigned cameraOffset = intCalibOffset + m_internalCalibs.size() * NumUpdateParams::INTERNAL_CALIB;
    for (unsigned i = 0; i < m_cameras.size(); i++) {
		// TO DO !!!
		/*
		 * Compose the new matrix H
		 * 
		 * m_cameras[i].H is the old matrix, state.m_cameras[i].H is the new matrix.
		 * 
		 * update[cameraOffset + i * NumUpdateParams::CAMERA + 0] rotation increment around the camera X axis (not world X axis)
		 * update[cameraOffset + i * NumUpdateParams::CAMERA + 1] rotation increment around the camera Y axis (not world Y axis)
		 * update[cameraOffset + i * NumUpdateParams::CAMERA + 2] rotation increment around the camera Z axis (not world Z axis)
		 * update[cameraOffset + i * NumUpdateParams::CAMERA + 3] translation increment along the camera X axis (not world X axis)
		 * update[cameraOffset + i * NumUpdateParams::CAMERA + 4] translation increment along the camera Y axis (not world Y axis)
		 * update[cameraOffset + i * NumUpdateParams::CAMERA + 5] translation increment along the camera Z axis (not world Z axis)
		 * 
		 * use rotationMatrixX(...), rotationMatrixY(...), rotationMatrixZ(...), and translationMatrix
		 * 
		 * The "*" multiplication operator works as one would expect it to.
		 */

		state.m_cameras[i].H = rotationMatrixZ(update[cameraOffset + i * NumUpdateParams::CAMERA + 2])
				*rotationMatrixY(update[cameraOffset + i * NumUpdateParams::CAMERA + 1])
				*rotationMatrixX(update[cameraOffset + i * NumUpdateParams::CAMERA + 0])
				*translationMatrix(
				update[cameraOffset + i * NumUpdateParams::CAMERA + 3],
				update[cameraOffset + i * NumUpdateParams::CAMERA + 4],
				update[cameraOffset + i * NumUpdateParams::CAMERA + 5])*m_cameras[i].H;
    }
    unsigned trackOffset = cameraOffset + m_cameras.size() * NumUpdateParams::CAMERA;
    for (unsigned i = 0; i < m_tracks.size(); i++) {
        state.m_tracks[i].location = m_tracks[i].location;
        
		// TO DO !!!
		/*
		 * Modify the new track location
		 * 
		 * m_tracks[i].location is the old location, state.m_tracks[i].location is the new location.
		 * 
		 * update[trackOffset + i * NumUpdateParams::TRACK + 0] increment of X
		 * update[trackOffset + i * NumUpdateParams::TRACK + 1] increment of Y
		 * update[trackOffset + i * NumUpdateParams::TRACK + 2] increment of Z
		 * update[trackOffset + i * NumUpdateParams::TRACK + 3] increment of W
		 */
        
        
		state.m_tracks[i].location(0) += update[trackOffset + i * NumUpdateParams::TRACK + 0];
        state.m_tracks[i].location(1) += update[trackOffset + i * NumUpdateParams::TRACK + 1];
        state.m_tracks[i].location(2) += update[trackOffset + i * NumUpdateParams::TRACK + 2];
        state.m_tracks[i].location(3) += update[trackOffset + i * NumUpdateParams::TRACK + 3];


		// Renormalization to length one
        float len = std::sqrt(innerProd(state.m_tracks[i].location, state.m_tracks[i].location));
        state.m_tracks[i].location *= 1.0f / len;
    }
}






/************************************************************************************************************/
/************************************************************************************************************/
/***************************                                     ********************************************/
/***************************    Nothing to do below this point   ********************************************/
/***************************                                     ********************************************/
/************************************************************************************************************/
/************************************************************************************************************/




BundleAdjustment::BAJacobiMatrix::BAJacobiMatrix(const Scene &scene)
{
    unsigned numResidualPairs = 0;
    for (const auto &camera : scene.cameras)
        numResidualPairs += camera.keypoints.size();
    
    m_rows.reserve(numResidualPairs);
    for (unsigned camIdx = 0; camIdx < scene.cameras.size(); camIdx++) {
        const auto &camera = scene.cameras[camIdx];
        for (unsigned kpIdx = 0; kpIdx < camera.keypoints.size(); kpIdx++) {
            m_rows.push_back({});
            m_rows.back().internalCalibIdx = camera.internalCalibIdx;
            m_rows.back().cameraIdx = camIdx;
            m_rows.back().keypointIdx = kpIdx;
            m_rows.back().trackIdx = camera.keypoints[kpIdx].trackIdx;
        }
    }
    
    m_internalCalibOffset = 0;
    m_cameraOffset = m_internalCalibOffset + scene.numInternalCalibs * NumUpdateParams::INTERNAL_CALIB;
    m_trackOffset = m_cameraOffset + scene.cameras.size() * NumUpdateParams::CAMERA;
    m_totalUpdateParams = m_trackOffset + scene.numTracks * NumUpdateParams::TRACK;
}

void BundleAdjustment::BAJacobiMatrix::multiply(float * __restrict dst, const float * __restrict src) const
{
    for (unsigned r = 0; r < m_rows.size(); r++) {
        float sumX = 0.0f;
        float sumY = 0.0f;
        for (unsigned i = 0; i < NumUpdateParams::INTERNAL_CALIB; i++) {
            sumX += src[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] * 
                        m_rows[r].J_internalCalib(0, i);
            sumY += src[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] * 
                        m_rows[r].J_internalCalib(1, i);
        }
        for (unsigned i = 0; i < NumUpdateParams::CAMERA; i++) {
            sumX += src[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] * 
                        m_rows[r].J_camera(0, i);
            sumY += src[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] * 
                        m_rows[r].J_camera(1, i);
        }
        for (unsigned i = 0; i < NumUpdateParams::TRACK; i++) {
            sumX += src[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] * 
                        m_rows[r].J_track(0, i);
            sumY += src[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] * 
                        m_rows[r].J_track(1, i);
        }
        dst[r*2+0] = sumX;
        dst[r*2+1] = sumY;
    }
}

void BundleAdjustment::BAJacobiMatrix::transposedMultiply(float * __restrict dst, const float * __restrict src) const
{
    memset(dst, 0, sizeof(float) * m_totalUpdateParams);
    // This is super ugly...
    for (unsigned r = 0; r < m_rows.size(); r++) {
        for (unsigned i = 0; i < NumUpdateParams::INTERNAL_CALIB; i++) {
            float elem = dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i];
            elem += src[r*2+0] * m_rows[r].J_internalCalib(0, i);
            elem += src[r*2+1] * m_rows[r].J_internalCalib(1, i);
            dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] = elem;
        }
        
        for (unsigned i = 0; i < NumUpdateParams::CAMERA; i++) {
            float elem = dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i];
            elem += src[r*2+0] * m_rows[r].J_camera(0, i);
            elem += src[r*2+1] * m_rows[r].J_camera(1, i);
            dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] = elem;
        }
        for (unsigned i = 0; i < NumUpdateParams::TRACK; i++) {
            float elem = dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i];
            elem += src[r*2+0] * m_rows[r].J_track(0, i);
            elem += src[r*2+1] * m_rows[r].J_track(1, i);
            dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] = elem;
        }
    }
}

void BundleAdjustment::BAJacobiMatrix::computeDiagJtJ(float * __restrict dst) const
{
    memset(dst, 0, sizeof(float) * m_totalUpdateParams);
    // This is super ugly...
    for (unsigned r = 0; r < m_rows.size(); r++) {
        for (unsigned i = 0; i < NumUpdateParams::INTERNAL_CALIB; i++) {
            float elem = dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i];
            elem += m_rows[r].J_internalCalib(0, i) * m_rows[r].J_internalCalib(0, i);
            elem += m_rows[r].J_internalCalib(1, i) * m_rows[r].J_internalCalib(1, i);
            dst[m_internalCalibOffset + m_rows[r].internalCalibIdx * NumUpdateParams::INTERNAL_CALIB + i] = elem;
        }
        for (unsigned i = 0; i < NumUpdateParams::CAMERA; i++) {
            float elem = dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i];
            elem += m_rows[r].J_camera(0, i) * m_rows[r].J_camera(0, i);
            elem += m_rows[r].J_camera(1, i) * m_rows[r].J_camera(1, i);
            dst[m_cameraOffset + m_rows[r].cameraIdx * NumUpdateParams::CAMERA + i] = elem;
        }
        for (unsigned i = 0; i < NumUpdateParams::TRACK; i++) {
            float elem = dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i];
            elem += m_rows[r].J_track(0, i) * m_rows[r].J_track(0, i);
            elem += m_rows[r].J_track(1, i) * m_rows[r].J_track(1, i);
            dst[m_trackOffset + m_rows[r].trackIdx * NumUpdateParams::TRACK + i] = elem;
        }
    }
}



BundleAdjustment::BAState::BAState(const Scene &scene) : m_scene(scene)
{
    m_tracks.resize(m_scene.numTracks);
    m_internalCalibs.resize(m_scene.numInternalCalibs);
    m_cameras.resize(m_scene.cameras.size());
}

OptimizationProblem::State* BundleAdjustment::BAState::clone() const
{
    return new BAState(m_scene);
}


BundleAdjustment::BundleAdjustment(Scene &scene) : m_scene(scene)
{
    m_numResiduals = 0;
    for (const auto &camera : m_scene.cameras)
        m_numResiduals += camera.keypoints.size()*2;
    
    m_numUpdateParameters = 
                m_scene.numInternalCalibs * NumUpdateParams::INTERNAL_CALIB +
                m_scene.cameras.size() * NumUpdateParams::CAMERA +
                m_scene.numTracks * NumUpdateParams::TRACK;
}

OptimizationProblem::JacobiMatrix* BundleAdjustment::createJacobiMatrix() const
{
    return new BAJacobiMatrix(m_scene);
}


void BundleAdjustment::downweightOutlierKeypoints(BAState &state)
{
    std::vector<float> residuals;
    residuals.resize(m_numResiduals);
    state.computeResiduals(residuals.data());
    
    std::vector<float> distances;
    distances.resize(m_numResiduals/2);
    
    unsigned residualIdx = 0;
    for (auto &c : m_scene.cameras) {
        for (auto &kp : c.keypoints) {
            distances[residualIdx/2] = 
                std::sqrt(residuals[residualIdx+0]*residuals[residualIdx+0] + 
                          residuals[residualIdx+1]*residuals[residualIdx+1]);
            residualIdx+=2;
        }
    }

    std::vector<float> sortedDistances = distances;
    std::sort(sortedDistances.begin(), sortedDistances.end());
    
    std::cout << "min, max, median distances (weighted): " << sortedDistances.front() << " " << sortedDistances.back() << " " << sortedDistances[sortedDistances.size()/2] << std::endl;
    
    float thresh = sortedDistances[sortedDistances.size() * 2 / 3] * 2.0f;
    
    residualIdx = 0;
    for (auto &c : m_scene.cameras)
        for (auto &kp : c.keypoints) 
            if (distances[residualIdx++] > thresh) 
                kp.weight *= 0.5f;
}


Scene buildScene(const std::vector<std::string> &imagesFilenames)
{
    struct Image {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        std::vector<std::vector<std::pair<unsigned, unsigned>>> matches;
    };
    
    std::vector<Image> allImages;
    allImages.resize(imagesFilenames.size());
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(10000);
    for (unsigned i = 0; i < imagesFilenames.size(); i++) {
        std::cout << "Extracting keypoints from " << imagesFilenames[i] << std::endl;
        cv::Mat img = cv::imread(imagesFilenames[i].c_str());
        orb->detectAndCompute(img, cv::noArray(), allImages[i].keypoints, allImages[i].descriptors);
        allImages[i].matches.resize(allImages[i].keypoints.size());
    }
    
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    for (unsigned i = 0; i < allImages.size(); i++)
        for (unsigned j = i+1; j < allImages.size(); j++) {
            std::cout << "Matching " << imagesFilenames[i] << " against " << imagesFilenames[j] << std::endl;
            
            std::vector<std::vector<cv::DMatch>> matches;
            matcher->knnMatch(allImages[i].descriptors, allImages[j].descriptors, matches, 2);
            for (unsigned k = 0; k < matches.size(); ) {
                if (matches[k][0].distance > matches[k][1].distance * 0.75f) {
                    matches[k] = std::move(matches.back());
                    matches.pop_back();
                } else k++;
            }
            cv::Mat p1 = cv::Mat::zeros(3, matches.size(), CV_32FC1);
            cv::Mat p2 = cv::Mat::zeros(3, matches.size(), CV_32FC1);
            for (unsigned k = 0; k < matches.size(); k++) {
                p1.at<float>(0, k) = allImages[i].keypoints[matches[k][0].queryIdx].pt.x;
                p1.at<float>(1, k) = allImages[i].keypoints[matches[k][0].queryIdx].pt.y;
                p1.at<float>(2, k) = 1.0f;
                p2.at<float>(0, k) = allImages[j].keypoints[matches[k][0].trainIdx].pt.x;
                p2.at<float>(1, k) = allImages[j].keypoints[matches[k][0].trainIdx].pt.y;
                p2.at<float>(2, k) = 1.0f;
            }
            std::cout << "RANSACing " << imagesFilenames[i] << " against " << imagesFilenames[j] << std::endl;
            
            cv::Mat F = estimateFundamentalRANSAC(p1, p2, 1000);
            
            const float threshold = 20.0f;
            
            std::vector<std::pair<unsigned, unsigned>> inlierMatches;
            for (unsigned k = 0; k < matches.size(); k++) 
                if (getError(p1.colRange(k, k+1), p2.colRange(k, k+1), F) < threshold) 
                    inlierMatches.push_back({
                        matches[k][0].queryIdx,
                        matches[k][0].trainIdx
                    });
            const unsigned minMatches = 400;
                
            std::cout << "Found " << inlierMatches.size() << " valid matches!" << std::endl;
            if (inlierMatches.size() >= minMatches)
                for (const auto p : inlierMatches) {
                    allImages[i].matches[p.first].push_back({j, p.second});
                    allImages[j].matches[p.second].push_back({i, p.first});
                }
        }
    
    
    Scene scene;
    scene.numInternalCalibs = 1;
    scene.cameras.resize(imagesFilenames.size());
    for (auto &c : scene.cameras)
        c.internalCalibIdx = 0;
    scene.numTracks = 0;
    
    std::cout << "Finding tracks " << std::endl;
    {
        std::set<std::pair<unsigned, unsigned>> handledKeypoints;
        std::set<unsigned> imagesSpanned;
        std::vector<std::pair<unsigned, unsigned>> kpStack;
        std::vector<std::pair<unsigned, unsigned>> kpList;
        for (unsigned i = 0; i < allImages.size(); i++) {
            for (unsigned kp = 0; kp < allImages[i].keypoints.size(); kp++) {
                if (allImages[i].matches[kp].empty()) continue;
                if (handledKeypoints.find({i, kp}) != handledKeypoints.end()) continue;
                
                bool valid = true;
                
                kpStack.push_back({i, kp});
                while (!kpStack.empty()) {
                    auto kp = kpStack.back();
                    kpStack.pop_back();
                    
                    
                    if (imagesSpanned.find(kp.first) != imagesSpanned.end()) // appearing twice in one image -> invalid
                        valid = false;
                    
                    handledKeypoints.insert(kp);
                    kpList.push_back(kp);
                    imagesSpanned.insert(kp.first);
                    
                    for (const auto &matchedKp : allImages[kp.first].matches[kp.second])
                        if (handledKeypoints.find(matchedKp) == handledKeypoints.end()) 
                            kpStack.push_back(matchedKp);
                }
                
                if (valid) {
                    //std::cout << "Forming track from group of " << kpList.size() << " keypoints over " << imagesSpanned.size() << " images" << std::endl;
                    
                    for (const auto &kp : kpList) {
                        Vector<2> pixelPosition;
                        pixelPosition(0) = allImages[kp.first].keypoints[kp.second].pt.x;
                        pixelPosition(1) = allImages[kp.first].keypoints[kp.second].pt.y;
                        
                        unsigned trackIdx = scene.numTracks;
                        
                        scene.cameras[kp.first].keypoints.push_back({
                            pixelPosition,
                            trackIdx,
                            1.0f
                        });
                    }
                    
                    scene.numTracks++;
                } else {
                    //std::cout << "Dropping invalid group of " << kpList.size() << " keypoints over " << imagesSpanned.size() << " images" << std::endl;
                }
                kpList.clear();
                imagesSpanned.clear();
            }
        }
        std::cout << "Formed " << scene.numTracks << " tracks" << std::endl;
    }
    
    for (auto &c : scene.cameras)
        if (c.keypoints.size() < 100)
            std::cout << "Warning: One camera is connected with only " << c.keypoints.size() << " keypoints, this might be too unstable!" << std::endl;

    return scene;
}

void produceInitialState(const Scene &scene, const Matrix<3, 3> &initialInternalCalib, BundleAdjustment::BAState &state)
{
    state.m_internalCalibs[0].K = initialInternalCalib;
    
    std::set<unsigned> triangulatedPoints;
    
    const unsigned image1 = 0;
    const unsigned image2 = 1;
    // Find stereo pose of first two images
    {
        
        std::map<unsigned, Vector<2>> track2keypoint;
        for (const auto &kp : scene.cameras[image1].keypoints)
            track2keypoint[kp.trackIdx] = kp.location;
        
        std::vector<std::pair<Vector<2>, Vector<2>>> matches;
        std::vector<unsigned> matches2track;
        for (const auto &kp : scene.cameras[image2].keypoints) {
            auto it = track2keypoint.find(kp.trackIdx);
            if (it != track2keypoint.end()) {
                matches.push_back({it->second, kp.location});
                matches2track.push_back(kp.trackIdx);
            }
        }
        
        std::cout << "Initial pair has " << matches.size() << " matches" << std::endl;
        
        cv::Mat p1 = cv::Mat::zeros(3, matches.size(), CV_32FC1);
        cv::Mat p2 = cv::Mat::zeros(3, matches.size(), CV_32FC1);
        for (unsigned i = 0; i < matches.size(); i++) {
            p1.at<float>(0, i) = matches[i].first(0);
            p1.at<float>(1, i) = matches[i].first(1);
            p1.at<float>(2, i) = 1.0f;
            p2.at<float>(0, i) = matches[i].second(0);
            p2.at<float>(1, i) = matches[i].second(1);
            p2.at<float>(2, i) = 1.0f;
        }
        
        auto nonConstInternalCalib = initialInternalCalib;
        cv::Mat K(3, 3, CV_32F, &nonConstInternalCalib(0, 0));
        cv::Mat H = computeCameraPose(K, p1, p2);
        
        state.m_cameras[image1].H.setIdentity();
        for (unsigned i = 0; i < 4; i++)
            for (unsigned j = 0; j < 4; j++)
                state.m_cameras[image2].H(i, j) = H.at<float>(i, j);
            
        cv::Mat X = linearTriangulation(K * cv::Mat::eye(3, 4, CV_32F), K * H(cv::Range(0, 3), cv::Range(0, 4)), p1, p2);
        for (unsigned i = 0; i < X.cols; i++) {
            auto &t = state.m_tracks[matches2track[i]].location;
            t(0) = X.at<float>(0, i);
            t(1) = X.at<float>(1, i);
            t(2) = X.at<float>(2, i);
            t(3) = X.at<float>(3, i);
            
            t /= std::sqrt(innerProd(t, t));
            
            triangulatedPoints.insert(matches2track[i]);
        }
    }
    

    for (unsigned c = 0; c < scene.cameras.size(); c++) {
        if (c == image1) continue;
        if (c == image2) continue;
        
        std::vector<KeyPoint> triangulatedKeypoints;
        for (const auto &kp : scene.cameras[c].keypoints) 
            if (triangulatedPoints.find(kp.trackIdx) != triangulatedPoints.end()) 
                triangulatedKeypoints.push_back(kp);

        if (triangulatedKeypoints.size() < 100)
            std::cout << "Warning: Camera " << c << " is only estimated from " << triangulatedKeypoints.size() << " keypoints" << std::endl;
        
        cv::Mat points2D(3, triangulatedKeypoints.size(), CV_32F);
        cv::Mat points3D(4, triangulatedKeypoints.size(), CV_32F);
        
        for (unsigned i = 0; i < triangulatedKeypoints.size(); i++) {
            points2D.at<float>(0, i) = triangulatedKeypoints[i].location(0);
            points2D.at<float>(1, i) = triangulatedKeypoints[i].location(1);
            points2D.at<float>(2, i) = 1.0f;
            Vector<3> tp = hom2eucl(state.m_tracks[triangulatedKeypoints[i].trackIdx].location);
            points3D.at<float>(0, i) = tp(0);
            points3D.at<float>(1, i) = tp(1);
            points3D.at<float>(2, i) = tp(2);
            points3D.at<float>(3, i) = 1.0f;
        }
        
        std::cout << "Estimating camera " << c << " from " << triangulatedKeypoints.size() << " keypoints" << std::endl;
        //cv::Mat P = calibrate(points2D, points3D);
        cv::Mat P = estimateProjectionRANSAC(points2D, points3D, 1000);
        cv::Mat K, R;
        ProjectionMatrixInterpretation info;
        interprete(P, K, R, info);
        
        state.m_cameras[c].H.setIdentity();
        for (unsigned i = 0; i < 3; i++)
            for (unsigned j = 0; j < 3; j++)
                state.m_cameras[c].H(i, j) = R.at<float>(i, j);
            
        state.m_cameras[c].H = state.m_cameras[c].H * translationMatrix(-info.cameraLocation[0], -info.cameraLocation[1], -info.cameraLocation[2]);
    }
    // Triangulate remaining points
    for (unsigned c = 0; c < scene.cameras.size(); c++) {
        
        Matrix<3, 4> P1 = state.m_internalCalibs[scene.cameras[c].internalCalibIdx].K * dropLastRow(state.m_cameras[c].H);
        cv::Mat cvP1(3, 4, CV_32F, &P1(0, 0));
            
        for (unsigned otherC = 0; otherC < c; otherC++) {
            Matrix<3, 4> P2 = state.m_internalCalibs[scene.cameras[otherC].internalCalibIdx].K * dropLastRow(state.m_cameras[otherC].H);
            cv::Mat cvP2(3, 4, CV_32F, &P2(0, 0));
            for (const auto &kp : scene.cameras[c].keypoints) {
                if (triangulatedPoints.find(kp.trackIdx) != triangulatedPoints.end()) continue;
                
                for (const auto &otherKp : scene.cameras[otherC].keypoints) {
                    if (kp.trackIdx == otherKp.trackIdx) {
                        cv::Mat X = linearTriangulation(
                            cvP1, cvP2,
                            (cv::Mat_<float>(3, 1) << kp.location(0), kp.location(1), 1.0f),
                            (cv::Mat_<float>(3, 1) << otherKp.location(0), otherKp.location(1), 1.0f)
                        );
                        X /= X.at<float>(3, 0);
                        
                        auto &t = state.m_tracks[kp.trackIdx].location;
                        t(0) = X.at<float>(0, 0);
                        t(1) = X.at<float>(1, 0);
                        t(2) = X.at<float>(2, 0);
                        t(3) = X.at<float>(3, 0);
                        t /= std::sqrt(innerProd(t, t));
                        
                        triangulatedPoints.insert(kp.trackIdx);
                    }
                }
            }
        }
    }
    if (triangulatedPoints.size() != state.m_tracks.size())
        std::cout << "Warning: Some tracks were not triangulated. This should not happen!" << std::endl;
}


}

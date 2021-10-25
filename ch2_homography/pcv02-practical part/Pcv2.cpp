//============================================================================
// Name        : Pcv2test.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Pcv2.h"
#include<iostream>

namespace pcv2 {

// compute the homography
/*
base		first set of points x'
attach		second set of points x
return		homography H, so that x' = Hx
*/
cv::Mat homography2D(cv::Mat &base, cv::Mat &attach){
	// TO DO !!!
	//Conditioning T
	//Translate the centroid of all points to the origin
	cv::Mat baseT = getCondition2D(base);
	cv::Mat attachT = getCondition2D(attach);

	//Transform points
	cv::Mat base2D = applyH(base, baseT, GEOM_TYPE_POINT);
	cv::Mat attach2D = applyH(attach, attachT, GEOM_TYPE_POINT);

	//Create Design Matrix A (size:2N*9)
	cv::Mat A = getDesignMatrix_homography2D(base2D, attach2D);
	
	//SVD
	cv::Mat h = solve_dlt(A);

	//Reverse Conditioning
	decondition(baseT, attachT, h);

    return h;
}

// solve homogeneous equation system by usage of SVD
/*
A		the design matrix
return	solution of the homogeneous equation system
*/
cv::Mat solve_dlt(cv::Mat &A){
	// TO DO !!!

	//Create a one row Matrix which values are all 0
	cv::Mat B = cv::Mat(1, 9, CV_32FC1);
	for (int j = 0; j < 9; j++)
	{
		B.at<float>(0, j) = 0.0;
	}

	//Add a one more rows to A
	cv::vconcat(A, B, A);

	//Calculate eigen values and eigen vectors
	cv::Mat U, W, VT, V;
	cv::SVD::compute(A, W, U, VT);

	V = VT.t();

	//Compute the mininum of eigen value
	float vmin = W.at<float>(0,0);
	int min_loc;
	for (int i = 1; i < 9; i++)
	{
		if (W.at<float>(i,0) < vmin)
		{
			std::cout << W.at<float>(i,0) << std::endl;
			vmin = W.at<float>(i,0);	//record mininum of eigen value
			min_loc = i;						//record the location of min value (column)
		}
	}

	//Reshape
	//to get the homography for the conditioned coordinates
	cv::Mat h = cv::Mat::eye(3, 3, CV_32FC1);
	h.at<float>(0, 0) = V.at<float>(0, min_loc);
	h.at<float>(0, 1) = V.at<float>(1, min_loc);
	h.at<float>(0, 2) = V.at<float>(2, min_loc);
	h.at<float>(1, 0) = V.at<float>(3, min_loc);
	h.at<float>(1, 1) = V.at<float>(4, min_loc);
	h.at<float>(1, 2) = V.at<float>(5, min_loc);
	h.at<float>(2, 0) = V.at<float>(6, min_loc);
	h.at<float>(2, 1) = V.at<float>(7, min_loc);
	h.at<float>(2, 2) = V.at<float>(8, min_loc);

	for (auto i = 0; i < 3; i++)
	{
		for (auto j = 0; j < 3; j++)
		{
			std::cout << h.at<float>(i, j)<<" ";
		}
		std::cout << std::endl;
	}

    return h;
}

// decondition a homography that was estimated from conditioned point clouds
/*
T_base		conditioning matrix T' of first set of points x'
T_attach	conditioning matrix T of second set of points x
H			conditioned homography that has to be un-conditioned (in-place)
*/
void decondition(cv::Mat &T_base, cv::Mat &T_attach, cv::Mat &H){
  	// TO DO !!!
	H = (T_base.inv())*H*T_attach;
}

// define the design matrix as needed to compute 2D-homography
/*
base	first set of points x' --> x' = H * x
attach	second set of points x --> x' = H * x
return	the design matrix to be computed
*/
cv::Mat getDesignMatrix_homography2D(cv::Mat &base, cv::Mat &attach){
	// TO DO !!!
	int N = base.cols;	//the number of corresponding point pairs
	cv::Mat A = cv::Mat(2 * N, 9, CV_32FC1);

	//Calculate the design matrix (2N points'iterations: N)
	int i = 0;
	for (int j = 0; j < N; j++)
	{
		A.at<float>(i, 0) = -base.at<float>(2, j)*attach.at<float>(0, j);
		A.at<float>(i, 1) = -base.at<float>(2, j)*attach.at<float>(1, j);
		A.at<float>(i, 2) = -base.at<float>(2, j)*attach.at<float>(2, j);
		A.at<float>(i, 3) = 0.0;
		A.at<float>(i, 4) = 0.0;
		A.at<float>(i, 5) = 0.0;
		A.at<float>(i, 6) = base.at<float>(0, j)*attach.at<float>(0, j);
		A.at<float>(i, 7) = base.at<float>(0, j)*attach.at<float>(1, j);
		A.at<float>(i, 8) = base.at<float>(0, j)*attach.at<float>(2, j);

		A.at<float>(i + 1, 0) = 0.0;
		A.at<float>(i + 1, 1) = 0.0;
		A.at<float>(i + 1, 2) = 0.0;
		A.at<float>(i + 1, 3) = -base.at<float>(2, j)*attach.at<float>(0, j);
		A.at<float>(i + 1, 4) = -base.at<float>(2, j)*attach.at<float>(1, j);
		A.at<float>(i + 1, 5) = -base.at<float>(2, j)*attach.at<float>(2, j);
		A.at<float>(i + 1, 6) = base.at<float>(1, j)*attach.at<float>(0, j);
		A.at<float>(i + 1, 7) = base.at<float>(1, j)*attach.at<float>(1, j);
		A.at<float>(i + 1, 8) = base.at<float>(1, j)*attach.at<float>(2, j);

		i = i + 2;
	}

    return A;
}

// apply transformation to set of points
/*
H			matrix representing the transformation
geomObj		matrix with input objects (one per column)
type		the type of the geometric object (for now: only point and line)
return		transformed objects (one per column)
*/
cv::Mat applyH(cv::Mat &geomObj, cv::Mat &H, GeometryType type){
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

// get the conditioning matrix of given points
/*
p		the points as matrix
return	the condition matrix (already allocated)
*/
cv::Mat getCondition2D(cv::Mat &p){
	// TO DO !!!
	//the number of columns and rows
	int cols = p.cols;
	int rows = p.rows;

	//get the coordinate of centre point
	float t[2] = { 0.0 };
	for (int r = 0; r < rows-1; r++)
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
	
	//Create a transfer matrix T1
	cv::Mat T = cv::Mat::eye(3, 3, CV_32FC1);
	T.at<float>(0, 0) = 1.0/s[0];
	T.at<float>(0, 1) = 0.0;
	T.at<float>(0, 2) = -t[0]/s[0];
	T.at<float>(1, 0) = 0.0;
	T.at<float>(1, 1) = 1.0/s[1];
	T.at<float>(1, 2) = -t[1]/s[1];
	T.at<float>(2, 0) = 0.0;
	T.at<float>(2, 1) = 0.0;
	T.at<float>(2, 2) = 1.0;

	return T;
}

}

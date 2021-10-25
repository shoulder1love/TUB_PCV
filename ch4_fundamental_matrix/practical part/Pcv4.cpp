//============================================================================
// Name        : Pcv4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 2.0
// Copyright   : -
// Description : Estimation of Fundamental Matrix
//============================================================================

#include "Pcv4.h"

#include <random>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>   



using namespace cv;
using namespace std;


namespace pcv4 {

	/**
	* @brief Apply transformation to set of points
	* @param H Matrix representing the transformation
	* @param geomObj Matrix with input objects (one per column)
	* @param type The type of the geometric object (for now: only point and line)
	* @returns Transformed objects (one per column)
	*/
	cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type)
	{
		cv::Mat P;
		cv::Mat L;
		switch (type) {
		case GEOM_TYPE_POINT:
			P = H * geomObj;
			return P;
		case GEOM_TYPE_LINE:
			L = (H.inv().t())*geomObj;
			return L;
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
	* @brief Compute the fundamental matrix
	* @param p1 first set of points
	* @param p2 second set of points
	* @returns	the estimated fundamental matrix
	*/
	cv::Mat getFundamentalMatrix(cv::Mat& p1, cv::Mat& p2) {
		//TO DO !!!
		cv::Mat p1_H = getCondition2D(p1);
		cv::Mat p2_H = getCondition2D(p2);

		cv::Mat p1_ = applyH(p1, p1_H, GEOM_TYPE_POINT);
		cv::Mat p2_ = applyH(p2, p2_H, GEOM_TYPE_POINT);

		cv::Mat A_design = getDesignMatrix_fundamental(p1_, p2_);

		cv::Mat F = solve_dlt(A_design);

		forceSingularity(F);
		decondition(p1_H, p2_H, F);

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
		//TO DO !!!
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
	cv::Mat solve_dlt(cv::Mat& A)
	{
		// TO DO !!!
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
	void decondition(cv::Mat& T1, cv::Mat& T2, cv::Mat& F)
	{
		// TO DO !!!
		F = (T2.t())*F*T1;
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

		const unsigned subsetSize = 8;

		// TO DO !!!
		//Get Fundamental Matrix
		Mat F,F_;
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
	* @brief Draw points and corresponding epipolar lines into both images
	* @param img1 Structure containing first image
	* @param img2 Structure containing second image
	* @param p1 First point set (points in first image)
	* @param p2 First point set (points in second image)
	* @param F Fundamental matrix (mapping from point in img1 to lines in img2)
	*/
	void visualize(cv::Mat& img1, cv::Mat& img2, cv::Mat& p1, cv::Mat& p2, cv::Mat& F)
	{
		// make a copy to not draw into the original images and destroy them
		cv::Mat img1_copy = img1.clone();
		cv::Mat img2_copy = img2.clone();
		
		int N = p1.cols;
		
		Mat l1 = F.t()*p2;
		Mat l2 = F * p1;
		for (int i = 0; i < N; i++)
		{
			Point point1, point2;
			point1.x = p1.at<float>(0, i);
			point1.y = p1.at<float>(1, i);
			circle(img1_copy, point1, 4, Scalar(0, 255, 0), 2);
			point2.x = p2.at<float>(0, i);
			point2.y = p2.at<float>(1, i);
			circle(img2_copy, point2, 4, Scalar(0, 255, 0), 2);

			//Draw epilines 
			drawEpiLine(img1_copy, l1.at<float>(0, i), l1.at<float>(1, i), l1.at<float>(2, i));
			drawEpiLine(img2_copy, l2.at<float>(0, i), l2.at<float>(1, i), l2.at<float>(2, i));
		
		}
		
		namedWindow("Epiline1", CV_WINDOW_NORMAL);
		imshow("Epiline1", img1_copy);
		namedWindow("Epiline2", CV_WINDOW_NORMAL);
		imshow("Epiline2", img2_copy);
		waitKey(0);
		destroyAllWindows();
		
   	}



	/**
	* @brief Displays two images and catches the point pairs marked by left mouse clicks.
	* @details Points will be in homogeneous coordinates.
	* @param img1 The first image
	* @param img2 The second image
	* @param p1 Points within the first image (returned in the matrix by this method)
	* @param p2 Points within the second image (returned in the matrix by this method)
	*/
	int getPointsAutomatic(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &p1, cv::Mat &p2)
	{
		if (!img1.data || !img2.data)
		{
			cout << "ERROR reading images!!" << endl;
			return -1;
		}
		Mat img1_ = img1.clone();
		Mat img2_ = img2.clone();
		cvtColor(img1_, img1_, COLOR_RGB2GRAY);
		cvtColor(img2_, img2_, COLOR_RGB2GRAY);
		//Extract keypoints and descriptors in both images
		Ptr<ORB> orb = ORB::create(2000);
		//Create Keypoints vector
		vector<KeyPoint> Keypoints1, Keypoints2;
		//Create image descriptors<Mat>
		Mat descriptor1, descriptor2;
		orb->detect(img1, Keypoints1);
		orb->detect(img2, Keypoints2);
		//Get the coordinates of keypoints
		int k1 = Keypoints1.size();
		int k2 = Keypoints2.size();
		Mat p1_ = Mat::zeros(3, k1, CV_32FC1);
		Mat p2_ = Mat::zeros(3, k2, CV_32FC1);
		for (int i = 0; i < k1; i++)
		{
			p1_.at<float>(0, i) = Keypoints1[i].pt.x;
			p1_.at<float>(1, i) = Keypoints1[i].pt.y;
			p1_.at<float>(2, i) = 1.0;
		}
		for (int i = 0; i < k2; i++)
		{
			p2_.at<float>(0, i) = Keypoints2[i].pt.x;
			p2_.at<float>(1, i) = Keypoints2[i].pt.y;
			p2_.at<float>(2, i) = 1.0;
		}

		p1 = p1_.clone();
		p2 = p2_.clone();

		orb->detectAndCompute(img1, Mat(), Keypoints1, descriptor1);
		orb->detectAndCompute(img2, Mat(), Keypoints2, descriptor2);

		//visualize Keypoints
		Mat Show1, Show2;
		drawKeypoints(img1, Keypoints1, Show1);
		drawKeypoints(img2, Keypoints2, Show2);
		namedWindow("Keypoints1", CV_WINDOW_NORMAL);
		imshow("Keypoints1", Show1);
		namedWindow("Keypoints2", CV_WINDOW_NORMAL);
		imshow("Keypoints2", Show2);
		waitKey(0);


		//Match keypoints / descriptors
		vector<DMatch> matches(2000);
		BFMatcher matcher(NORM_HAMMING);			//Create a BFMMatcher matcher, use hamming distance
		matcher.match(descriptor1, descriptor2, matches);
		cout << "The number of matches: " << matches.size() << endl;

		//Visualization
		Mat ShowMatches;
 		drawMatches(img1, Keypoints1, img2, Keypoints2, matches, ShowMatches, Scalar(255, 0, 255));
		namedWindow("Matches", CV_WINDOW_NORMAL);
		imshow("Matches", ShowMatches);
		waitKey(0);

		return matches.size();
	}

}

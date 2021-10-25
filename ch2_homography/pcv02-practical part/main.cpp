//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv2.h"
#include "Helper.h"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include <vector>
#include<opencv2/features2d.hpp>


using namespace std;
using namespace cv;

// function loads input image, calls processing function, and saves result
/*
fname	path to input image
*/
void run(const std::string &fnameBase, const std::string &fnameLeft, const std::string &fnameRight) {

	// load image first two images, paths in argv[1] and argv[2]
	cv::Mat baseImage = cv::imread(fnameBase);
	cv::Mat attachImage = cv::imread(fnameLeft);
	if (!baseImage.data) {
		cerr << "ERROR: Cannot read image ( " << fnameBase << endl;
		cin.get();
		exit(-1);
	}
	if (!attachImage.data) {
		cerr << "ERROR: Cannot read image ( " << fnameLeft << endl;
		cin.get();
		exit(-1);
	}

	////------------------------------Addition code-------------------------------------
	////RGB to GRAY
	////cv::Mat baseGray, attachGray;
	////cv::cvtColor(baseImage, baseGray, CV_RGB2GRAY);
	////cv::cvtColor(attachImage, attachGray, CV_RGB2GRAY);

	////Initialization
	////Create two key point arrays(type:KeyPoint)
	//vector<KeyPoint> keyPoint_1, keyPoint_2;
	//Mat descriptors_1, descriptors_2;
	//Ptr<ORB> orb = ORB::create();

	////Firstly: detect corner positions
	//orb->detect(baseImage, keyPoint_1);
	//orb->detect(attachImage, keyPoint_2);

	////Secondly: Compute brief descriptors
	//orb->compute(baseImage, keyPoint_1, descriptors_1);
	//orb->compute(attachImage, keyPoint_2, descriptors_2);

	//Mat outImg1;
	//drawKeypoints(baseImage, keyPoint_1, outImg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB feature points", outImg1);

	////Thirdly: matching
	//vector<DMatch> matches;
	//BFMatcher matcher(NORM_HAMMING);

	//matcher.match(descriptors_1, descriptors_2, matches);

	//double min_dist = 0, max_dist = 0;
	//for (int i = 0; i < descriptors_1.rows; ++i)
	//{
	//	double dist = matches[i].distance;
	//	if (dist < min_dist) min_dist = dist;
	//	if (dist > max_dist) max_dist = dist;
	//}
	//cout << "Max dist: " << max_dist << endl;
	//cout << "Min dist: " << min_dist << endl;

	//std::vector<DMatch> good_matches;
	//for (int j = 0; j < descriptors_1.rows; ++j)
	//{
	//	if (matches[j].distance <= max(2 * min_dist, 30.0))
	//		good_matches.push_back(matches[j]);
	//}
	//Mat img_match;	//所有匹配图
	//drawMatches(baseImage, keyPoint_1, attachImage, keyPoint_2, matches, img_match);
	//imshow("所有匹配点对", img_match);


	//Mat img_goodmatch;//筛选后的匹配点图
	//drawMatches(baseImage, keyPoint_1, attachImage, keyPoint_2, good_matches, img_goodmatch);

	//imshow("筛选后的匹配点对", img_goodmatch);
	//waitKey(0);

	//------------------------ --------end---------------------------------------------

	// get corresponding points within the two image
	// start with one point within the attached image, then click on corresponding point in base image
 	cv::Mat p_basis, p_attach;
	int numberOfPointPairs = pcv2::getPoints(baseImage, attachImage, p_basis, p_attach);
	

	// just some putput
	cout << "Number of defined point pairs: " << numberOfPointPairs << endl;
	cout << endl << "Points in base image:" << endl;
	cout << p_basis << endl;
	cout << endl << "Points in second image:" << endl;
	cout << p_attach << endl;

	// calculate homography
	cv::Mat H = pcv2::homography2D(p_basis, p_attach);

	// create panorama
	cv::Mat panorama = pcv2::stitch(baseImage, attachImage, H);

	const char *windowName = "Panorama";

	// display panorama (resizeable)
	cv::namedWindow(windowName, 0);
	cv::imshow(windowName, panorama);
	cv::waitKey(0);
	cv::destroyWindow(windowName);

	// panorama is new base image, third image is the image to attach
	baseImage = panorama;
	// load third image
	attachImage = cv::imread(fnameRight);
	if (!attachImage.data) {
		cout << "ERROR: Cannot read image ( " << fnameRight << " )" << endl;
		cin.get();
		exit(-1);
	}

	// get corresponding points within the two image
	// start with one point within the attached image, then click on corresponding point in base image
	numberOfPointPairs = pcv2::getPoints(baseImage, attachImage, p_basis, p_attach);

	// just some putput
	cout << "Number of defined point pairs: " << numberOfPointPairs << endl;
	cout << endl << "Points in base image:" << endl;
	cout << p_basis << endl;
	cout << endl << "Points in second image:" << endl;
	cout << p_attach << endl;

	// calculate homography
	H = pcv2::homography2D(p_basis, p_attach);

	// create panorama
	panorama = pcv2::stitch(baseImage, attachImage, H);

	// display panorama (resizeable)
	cv::namedWindow(windowName, 0);
	cv::imshow(windowName, panorama);
	cv::waitKey(0);
	cv::destroyWindow(windowName);

	cv::imwrite("panorama.png", panorama);
}





// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

	// will contain path to the input image (taken from argv[1])
	string fnameBase, fnameLeft, fnameRight;

	// check if image paths are defined
	if (argc != 4) {
		cout << "Usage: pcv2 <path to base image> <path to 2nd image> <path to 3rd image>" << endl;
		cout << "Press enter to continue..." << endl;
		cin.get();
		return -1;
	}
	else {
		// if yes, assign it to variable fname
		fnameBase = argv[1];
		fnameLeft = argv[2];
		fnameRight = argv[3];
	}

	// start processing
	run(fnameBase, fnameLeft, fnameRight);

	cout << "Press enter to continue..." << endl;
	cin.get();

	return 0;

}

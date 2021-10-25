//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv4.h"


#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>


using namespace std;
using namespace pcv4;



// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

    // will contain path to input image (taken from argv[1])
    string img1Path, img2Path;

    // check if image paths were defined
    if (argc != 3){
        cerr << "Usage: pcv4test <path to 1st image> <path to 2nd image>" << endl;
        cerr << "Press enter to continue..." << endl;
        cin.get();
        return -1;
    }else{
        // if yes, assign it to variable fname
        img1Path = argv[1];
        img2Path = argv[2];
    }
    
    // load images
    cv::Mat fstImage = cv::imread(img1Path);
    cv::Mat sndImage = cv::imread(img2Path);
    
    if ( (!fstImage.data) or (!sndImage.data)){
        cerr << "ERROR: Could not load images" << endl;
        cerr << "Press enter to continue..." << endl;
        cin.get();
        exit(-2);
    }

    // get corresponding points within the two images
    // start with one point within the first image, then click on corresponding point in second image
    cv::Mat p_fst, p_snd;
    int numberOfPointPairs = getPointsAutomatic(fstImage, sndImage, p_fst, p_snd);
    
    // just some putput
    cout << "Number of defined point pairs: " << numberOfPointPairs << endl;
    cout << endl << "Points in first image:" << endl;
    cout << p_fst << endl;
    cout << endl << "Points in second image:" << endl;
    cout << p_snd << endl;
    
    std::cout << "=== Direct estimation: ===" << std::endl;
    
    {
        // calculate fundamental matrix
        cv::Mat F = getFundamentalMatrix(p_fst, p_snd);

        // calculate geometric error
        double err = getError(p_fst, p_snd, F);
        cout << "Geometric error: " << err << endl;
        unsigned numInlier = countInliers(p_fst, p_snd, F, 2.0f);
        cout << "Number of inliers: " << numInlier << " of " << p_fst.cols << endl;
        
        // visualize epipolar lines
        visualize(fstImage, sndImage, p_fst, p_snd, F);
    }
   
    std::cout << "=== Robust estimation: ===" << std::endl;
    
    {
        cv::Mat F = estimateFundamentalRANSAC(p_fst, p_snd, 1000);
    
        // calculate geometric error
        double err = getError(p_fst, p_snd, F);
        cout << "Geometric error: " << err << endl;
        unsigned numInlier = countInliers(p_fst, p_snd, F, 2.0f);
        cout << "Number of inliers: " << numInlier << " of " << p_fst.cols << endl;
        
        // visualize epipolar lines
        visualize(fstImage, sndImage, p_fst, p_snd, F);
    }

    cout << "Press enter to continue..." << endl;
    cin.get();

    return 0;

}

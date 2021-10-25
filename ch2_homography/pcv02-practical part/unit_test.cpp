//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv2.h"

#include <opencv2/opencv.hpp>

#include <iostream>


using namespace std;
using namespace pcv2;
using namespace cv;



void test_getCondition2D(void){

    Mat p = (Mat_<float>(3,4) << 93, 729, 703, 152, 617, 742, 1233, 1103, 1, 1, 1, 1);
    Mat Ttrue = (Mat_<float>(3,3) << 1./296.75, 0, -419.25/296.75, 0, 1./244.25, -923.75/244.25, 0, 0, 1);
    
    Mat Test = getCondition2D(p);
    if ( (Test.rows != 3) || (Test.cols != 3) || (Test.channels() != 1) ){
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cin.get();
        exit(-1);
    }
    Test.convertTo(Test, CV_32FC1);
    Test = Test / Test.at<float>(2,2);
    float eps = pow(10,-3);
    if (sum(abs(Test - Ttrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_getDesignMatrix_homography2D(void){
    
    Mat p1 = (Mat_<float>(3,4) << -1, 1, 1, -1, -1, -1, 1, 1,  1, 1, 1, 1);
    Mat p2 = (Mat_<float>(3,4) << -1.0994103, 1.0438079, 0.9561919, -0.90058976,  -1.2558856, -0.74411488, 1.2661204, 0.73387909, 1, 1, 1, 1);
    
    Mat Aest = getDesignMatrix_homography2D(p1, p2);
    if ( ( (Aest.rows != 8) && (Aest.rows != 9) ) || (Aest.cols != 9) || (Aest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with getDesignMatrix_homography2D(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cin.get();
        exit(-1);
    }
    Aest.convertTo(Aest, CV_32FC1);
    Mat Atrue;
    if (Aest.rows == 8)
        Atrue = (Mat_<float>(8,9) << 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1, -1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1, 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1, -0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1, 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1);
    else
        Atrue = (Mat_<float>(9,9) << 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1, -1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1, 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1, -0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1, 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    float eps = pow(10,-3);
    if (sum(abs(Aest - Atrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with getDesignMatrix_homography2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_solve_dlt(void){
    Mat A = (Mat_<float>(8,9) << 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 0, 0, 0, 1.0994103, 1.2558856, -1, 1.0994103, 1.2558856, -1, -1.0438079, 0.74411488, -1, 0, 0, 0, 1.0438079, -0.74411488, 1, 0, 0, 0, -1.0438079, 0.74411488, -1, -1.0438079, 0.74411488, -1, -0.9561919, -1.2661204, -1, 0, 0, 0, 0.9561919, 1.2661204, 1, 0, 0, 0, -0.9561919, -1.2661204, -1, 0.9561919, 1.2661204, 1, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, 0, 0, 0, 0.90058976, -0.73387909, -1, -0.90058976, 0.73387909, 1);
    Mat Hest = solve_dlt(A);
    if ( (Hest.rows != 3) || (Hest.cols != 3) || (Hest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with solve_dlt(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cin.get();
        exit(-1);
    }
    Hest.convertTo(Hest, CV_32FC1);
    Hest = Hest / Hest.at<float>(2,2);
    Mat Htrue = (Mat_<float>(3,3) << 0.57111752, -0.017852778, 0.013727478, -0.15091757, 0.57065326, -0.04098846, 0.024604173, -0.041672569, 0.56645769);
    Htrue = Htrue / Htrue.at<float>(2,2);
    float eps = pow(10,-3);
    if (sum(abs(Hest - Htrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with solve_dlt(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_decondition(void){
    
    Mat H = (Mat_<float>(3,3) << 0.57111752, -0.017852778, 0.013727478, -0.15091757, 0.57065326, -0.04098846, 0.024604173, -0.041672569, 0.56645769);
    Mat T1 = (Mat_<float>(3,3) << 1./319.5, 0, -1, 0, 1./319.5, -1, 0, 0, 1);
    Mat T2 = (Mat_<float>(3,3) << 1./296.75, 0, -419.25/296.75, 0, 1./244.25, -923.75/244.25, 0, 0, 1);
    decondition(T1, T2, H);
    if ( (H.rows != 3) || (H.cols != 3) || (H.channels() != 1) ){
        cout << "Warning: There seems to be a problem with decondition(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cin.get();
        exit(-1);
    }
    H.convertTo(H, CV_32FC1);
    H = H / H.at<float>(2,2);
    Mat Htrue = (Mat_<float>(3,3) << 0.9304952, -0.11296108, -16.839279, -0.19729686, 1.003845, -601.02362, 0.00012028422, -0.00024751772, 1);
    float eps = pow(10,-3);
    if (sum(abs(H - Htrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with decondition(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}

void test_homography2D(void){

    Mat p1 = (Mat_<float>(3,4) << 0, 639, 639, 0, 0, 0, 639, 639, 1, 1, 1, 1);	
    Mat p2 = (Mat_<float>(3,4) << 93, 729, 703, 152, 617, 742, 1233, 1103, 1, 1, 1, 1);
        
    Mat Hest = homography2D(p1, p2);
    if ( (Hest.rows != 3) || (Hest.cols != 3) || (Hest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with homography2D(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cin.get();
        exit(-1);
    }
    Hest.convertTo(Hest, CV_32FC1);
    Hest = Hest / Hest.at<float>(2,2);
    Mat Htrue = (Mat_<float>(3,3) << 0.9304952, -0.11296108, -16.839279, -0.19729686, 1.003845, -601.02362, 0.00012028422, -0.00024751772, 1);
    float eps = pow(10,-3);
    if (sum(abs(Hest - Htrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with homography2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cin.get();
        exit(-1);
    }
}



int main(int argc, char** argv) {

    test_homography2D();
    test_getCondition2D();
    test_getDesignMatrix_homography2D();
    test_solve_dlt();
    test_decondition();

    cout << "Finished basic testing: Everything seems to be fine." << endl;
	cin.get();
    return 0;

}

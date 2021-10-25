
#include "Pcv3.h"

#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <iostream>


// function calls processing functions
// output is tested on "correctness" 

using namespace pcv3;
using namespace std;


bool checkDifference(float estimate, float trueValue, float maxDiff, const char *name, const char *functionName)
{
    if (std::abs(estimate - trueValue) > maxDiff) {
        std::cout 
            << "Warning: There seems to be a problem in " << functionName << " with the parameter " << name << ":" << std::endl
            << "\t==> Expected:" << trueValue << std::endl
            << "\t==> But got:" << estimate << std::endl;
        cin.get();
        return false;
    }
    return true;
}


bool test_getCondition2D(void){
    
    bool correct = true;

    cv::Mat p = (cv::Mat_<float>(3,6) <<  18.5, 99.1, 13.8, 242.1,
                                151.1, 243.1, 46.8, 146.5,
                                221.8, 52.5, 147.1, 224.5,
                                1, 1, 1, 1, 1, 1);
    cv::Mat Ttrue = (cv::Mat_<float>(3,3) << 0.011883541, 0, -1.5204991, 0, 0.016626639, -2.3255126, 0, 0, 1);
        
    cv::Mat Test = getCondition2D(p);
    if ( (Test.rows != 3) || (Test.cols != 3) || (Test.channels() != 1) ){
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cout << "\t==> Expected 3x3, but got " << Test.rows << "x" << Test.cols << "." << endl;
        correct = false;
        cin.get();
    }
    Test.convertTo(Test, CV_32FC1);
    Test = Test / Test.at<float>(2,2);
    double eps = pow(10,-3);
    if (sum(abs(Test - Ttrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with getCondition2D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Ttrue << endl;
        cout << "\t==> But got:" << Test << endl;
        correct = false;
        cin.get();
    }
    return correct;
}

bool test_getCondition3D(void){
    
    bool correct = true;
    
    cv::Mat p = (cv::Mat_<float>(4,6) <<  44.7, -103.6, 47.4, -152.2,
                                -153.3, -149.4, -142.4, -146.6,
                                -150.1, 59.4, -96.9, 52.7, 258.7,
                                154.4, 59.8, 245.2, 151.3, 46.9,
                                1, 1, 1, 1, 1, 1);
    cv::Mat Ttrue = (cv::Mat_<float>(4,4) << 0.012117948, 0, 0, 0.9419685, 0, 0.011838989, 0, 0.83642465, 0, 0, 0.014988759, -2.2890332, 0, 0, 0, 1);
    
    cv::Mat Test = getCondition3D(p);
    if ( (Test.rows != 4) || (Test.cols != 4) || (Test.channels() != 1) ){
        cout << "Warning: There seems to be a problem with getCondition3D(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cout << "\t==> Expected 4x4, but got " << Test.rows << "x" << Test.cols << "." << endl;
        correct = false;
        cin.get();
    }
    Test.convertTo(Test, CV_32FC1);
    Test = Test / Test.at<float>(3,3);
    double eps = pow(10,-3);
    if (sum(abs(Test - Ttrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with getCondition3D(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Ttrue << endl;
        cout << "\t==> But got:" << Test << endl;
        correct = false;
        cin.get();
    }
    return correct;
}

bool test_getDesignMatrix_camera(void){
    
    bool correct = true;
    
    cv::Mat p1 = (cv::Mat_<float>(3,6) << -1.3006536, -0.34284019, -1.3565062, 1.3565062, 0.27510405, 1.3683897, -1.5473859,
                                0.11029005, 1.3622761, -1.4526141, 0.1202662, 1.4071679, 1, 1, 1, 1, 1, 1);
    cv::Mat p2 = (cv::Mat_<float>(4,6) << 1.4836408, -0.31345099, 1.5163593, -0.90238315, -0.91571301, -0.86845297, -0.84944731,
                                -0.89917129, -0.94060773, 1.5396607, -0.31077343, 1.4603393, 1.5885589, 0.025231123,
                                -1.3927054, 1.3862104, -0.021234035, -1.5860603, 1, 1, 1, 1, 1, 1);
    
    cv::Mat Aest = getDesignMatrix_camera(p1, p2);
    if ( (Aest.rows != 12) || (Aest.cols != 12) || (Aest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with getDesigncv::Matrix_camera(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cout << "\t==> Expected 12x12, but got " << Aest.rows << "x" << Aest.cols << "." << endl;
        correct = false;
        cin.get();
    }
    Aest.convertTo(Aest, CV_32FC1);
    cv::Mat Atrue = (cv::Mat_<float>(12,12) << -1.4836408, 0.84944731, -1.5885589, -1, 0, 0, 0, 0, -1.9297028, 1.1048367, -2.0661647,
                                    -1.3006536, 0, 0, 0, 0, -1.4836408, 0.84944731, -1.5885589, -1, -2.2957649, 1.3144228,
                                    -2.4581137, -1.5473859, 0.31345099, 0.89917129, -0.025231123, -1, 0, 0, 0, 0, 0.1074636,
                                    0.30827206, -0.0086502433, -0.34284019, 0, 0, 0, 0, 0.31345099, 0.89917129, -0.025231123,
                                    -1, -0.034570526, -0.099169649, 0.0027827418, 0.11029005, -1.5163593, 0.94060773,
                                    1.3927054, -1, 0, 0, 0, 0, -2.0569508, 1.2759403, 1.8892136, -1.3565062, 0, 0, 0, 0,
                                    -1.5163593, 0.94060773, 1.3927054, -1, 2.0657001, -1.2813674, -1.8972493, 1.3622761, 0.90238315,
                                    -1.5396607, -1.3862104, -1, 0, 0, 0, 0, -1.2240883, 2.0885594, 1.880403, 1.3565062, 0, 0, 0, 0, 0.90238315,
                                    -1.5396607, -1.3862104, -1, 1.3108145, -2.2365327, -2.0136287, -1.4526141, 0.91571301, 0.31077343, 0.021234035,
                                    -1, 0, 0, 0, 0, -0.25191635, -0.085495025, -0.005841569, 0.27510405, 0, 0, 0, 0, 0.91571301,
                                    0.31077343, 0.021234035, -1, -0.11012933, -0.03737554, -0.0025537368, 0.1202662, 0.86845297,
                                    -1.4603393, 1.5860603, -1, 0, 0, 0, 0, -1.1883821, 1.9983133, -2.1703486, 1.3683897, 0, 0, 0, 0,
                                    0.86845297, -1.4603393, 1.5860603, -1, -1.2220591, 2.0549426, -2.2318532, 1.4071679);
    
    double eps = pow(10,-3);
    if (sum(abs(Aest - Atrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with getDesigncv::Matrix_camera(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Atrue << endl;
        cout << "\t==> But got:" << Aest << endl;
        correct = false;
        cin.get();
    }
    return correct;	
}

bool test_solve_dlt(void){
    
    bool correct = true;
    
    cv::Mat A = (cv::Mat_<float>(12,12) << -1.4836408, 0.84944731, -1.5885589, -1, 0, 0, 0, 0, -1.9297028, 1.1048367, -2.0661647, -1.3006536,
                                0, 0, 0, 0, -1.4836408, 0.84944731, -1.5885589, -1, -2.2957649, 1.3144228, -2.4581137, -1.5473859,
                                0.31345099, 0.89917129, -0.025231123, -1, 0, 0, 0, 0, 0.1074636, 0.30827206, -0.0086502433, -0.34284019,
                                0, 0, 0, 0, 0.31345099, 0.89917129, -0.025231123, -1, -0.034570526, -0.099169649, 0.0027827418, 0.11029005,
                                -1.5163593, 0.94060773, 1.3927054, -1, 0, 0, 0, 0, -2.0569508, 1.2759403, 1.8892136, -1.3565062, 0, 0, 0, 0,
                                -1.5163593, 0.94060773, 1.3927054, -1, 2.0657001, -1.2813674, -1.8972493, 1.3622761, 0.90238315, -1.5396607,
                                -1.3862104, -1, 0, 0, 0, 0, -1.2240883, 2.0885594, 1.880403, 1.3565062, 0, 0, 0, 0, 0.90238315, -1.5396607,
                                -1.3862104, -1, 1.3108145, -2.2365327, -2.0136287, -1.4526141, 0.91571301, 0.31077343, 0.021234035,
                                -1, 0, 0, 0, 0, -0.25191635, -0.085495025, -0.005841569, 0.27510405, 0, 0, 0, 0, 0.91571301, 0.31077343,
                                0.021234035, -1, -0.11012933, -0.03737554, -0.0025537368, 0.1202662, 0.86845297, -1.4603393, 1.5860603,
                                -1, 0, 0, 0, 0, -1.1883821, 1.9983133, -2.1703486, 1.3683897, 0, 0, 0, 0, 0.86845297, -1.4603393, 1.5860603,
                                -1, -1.2220591, 2.0549426, -2.2318532, 1.4071679);
    cv::Mat Pest = solve_dlt(A);
    if ( (Pest.rows != 3) || (Pest.cols != 4) || (Pest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with solve_dlt(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cout << "\t==> Expected 3x4, but got " << Pest.rows << "x" << Pest.cols << "." << endl;
        correct = false;
        cin.get();
    }
    Pest.convertTo(Pest, CV_32FC1);
    Pest = Pest / Pest.at<float>(2,3);
    cv::Mat Ptrue = (cv::Mat_<float>(3,4) << -0.32120419, 0.3686696, -0.0095243761, 0.0035335352, -0.052101973, -0.083373591, -0.59302819, -0.0046088211, -0.030212782, -0.026015088, 0.0050823218, 0.63073134);
    Ptrue = Ptrue / Ptrue.at<float>(2,3);

    double eps = pow(10,-3);
    if (sum(abs(Pest - Ptrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with solve_dlt(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Ptrue << endl;
        cout << "\t==> But got:" << Pest << endl;
        correct = false;
        cin.get();
    }
    return correct;
}

bool test_decondition(void){
    
    bool correct = true;
    
    cv::Mat P = (cv::Mat_<float>(3,4) << -0.32120419, 0.3686696, -0.0095243761, 0.0035335352, -0.052101973, -0.083373591, -0.59302819, -0.0046088211, -0.030212782, -0.026015088, 0.0050823218, 0.63073134);
    cv::Mat T1 = (cv::Mat_<float>(3,3) << 0.011883541, 0, -1.5204991, 0, 0.016626639, -2.3255126, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<float>(4,4) << 0.012117948, 0, 0, 0.9419685, 0, 0.011838989, 0, 0.83642465, 0, 0, 0.014988759, -2.2890332, 0, 0, 0, 1);
    decondition(T1, T2, P);
    if ( (P.rows != 3) || (P.cols != 4) || (P.channels() != 1) ){
        cout << "Warning: There seems to be a problem with decondition(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cout << "\t==> Expected 3x4, but got " << P.rows << "x" << P.cols << "." << endl;
        correct = false;
        cin.get();
    }
    P.convertTo(P, CV_32FC1);
    P = P / P.at<float>(2,3);
    cv::Mat Ptrue = (cv::Mat_<float>(3,4) << -0.65811008, 0.57636172, -0.0039836233, 132.55562, -0.15676615, -0.18008056, -0.9210307, 270.33484, -0.00064357655, -0.00054140261, 0.00013390853, 1);
    double eps = pow(10,-3);
    if (sum(abs(P - Ptrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with decondition(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Ptrue << endl;
        cout << "\t==> But got:" << P << endl;
        correct = false;
        cin.get();
    }
    return correct;
}

bool test_calibrate(void){

    bool correct = true;
    
    cv::Mat p1 = (cv::Mat_<float>(3,6) << 18.5, 99.1, 13.8, 242.1, 151.1, 243.1, 46.8, 146.5, 221.8, 52.5, 147.1, 224.5, 1, 1, 1, 1, 1, 1);
    cv::Mat p2 = (cv::Mat_<float>(4,6) << 44.7, -103.6, 47.4, -152.2, -153.3, -149.4, -142.4, -146.6, -150.1, 59.4, -96.9, 52.7, 258.7, 154.4, 59.8, 245.2, 151.3, 46.9, 1, 1, 1, 1, 1, 1);
        
    cv::Mat Pest = calibrate(p1, p2);
    if ( (Pest.rows != 3) || (Pest.cols != 4) || (Pest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with calibrate(..)!" << endl;
        cout << "\t==> Wrong dimensions!" << endl;
        cout << "\t==> Expected 3x4, but got " << Pest.rows << "x" << Pest.cols << "." << endl;
        correct = false;
        cin.get();
    }
    Pest.convertTo(Pest, CV_32FC1);
    Pest = Pest / Pest.at<float>(2,3);
    cv::Mat Ptrue = (cv::Mat_<float>(3,4) << -0.65811008, 0.57636172, -0.0039836233, 132.55562, -0.15676615, -0.18008056, -0.9210307, 270.33484, -0.00064357655, -0.00054140261, 0.00013390853, 1);
    double eps = pow(10,-3);
    if (sum(abs(Pest - Ptrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with calibrate(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations!" << endl;
        cout << "\t==> Expected:" << Ptrue << endl;
        cout << "\t==> But got:" << Pest << endl;
        correct = false;
        cin.get();
    }
    return correct;
}

bool test_interprete(void){

    bool correct = true;
    
    cv::Mat P = (cv::Mat_<float>(3,4) << -0.65811008, 0.57636172, -0.0039836233, 132.55562, -0.15676615, -0.18008056, -0.9210307, 270.33484, -0.00064357655, -0.00054140261, 0.00013390853, 1);
    
    cv::Mat Kest, Rest;
    pcv3::ProjectionMatrixInterpretation info;
    pcv3::interprete(P, Kest, Rest, info);

    if ( (Kest.rows != 3) || (Kest.cols != 3) || (Kest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with interprete(..)!" << endl;
        cout << "\t==> Wrong dimensions of K!" << endl;
        cout << "\t==> Expected 3x3, but got " << Kest.rows << "x" << Kest.cols << "." << endl;
        correct = false;
        cin.get();
    }
    if ( (Rest.rows != 3) || (Rest.cols != 3) || (Rest.channels() != 1) ){
        cout << "Warning: There seems to be a problem with interprete(..)!" << endl;
        cout << "\t==> Wrong dimensions of R!" << endl;
        cout << "\t==> Expected 3x3, but got " << Rest.rows << "x" << Rest.cols << "." << endl;
        correct = false;
        cin.get();
    }
    Kest.convertTo(Kest, CV_32FC1);
    Kest = Kest / Kest.at<float>(2,2);
    cv::Mat Ktrue = (cv::Mat_<float>(3,3) << 1015.7469, -10.457143, 153.00758, 0, 1112.4618, 103.48759, 0, 0, 1);
    double eps = 1e-3;
    if (sum(abs(Kest - Ktrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with interprete(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations for K!" << endl;
        cout << "\t==> Expected:" << Ktrue << endl;
        cout << "\t==> But got:" << Kest << endl;
        correct = false;
        cin.get();
    }
    Rest.convertTo(Rest, CV_32FC1);
    cv::Mat Rtrue = (cv::Mat_<float>(3,3) << -0.64794523, 0.76071578, -0.038450673, -0.095171571, -0.13094184, -0.98681134, -0.75571775, -0.63574028, 0.15724166);
    if (sum(abs(Rest - Rtrue)).val[0] > eps){
        cout << "Warning: There seems to be a problem with interprete(..)!" << endl;
        cout << "\t==> Wrong or inaccurate calculations for R!" << endl;
        cout << "\t==> Expected:" << Rtrue << endl;
        cout << "\t==> But got:" << Rest << endl;
        correct = false;
        cin.get();
    }
    
    
    correct &= checkDifference(info.principalDistance, 1015.7469, eps, "principalDistance", "interprete(..)");
    correct &= checkDifference(info.skew,              89.410156, eps, "skew", "interprete(..)");
    correct &= checkDifference(info.aspectRatio,       1.0952156, eps, "aspectRatio", "interprete(..)");
    correct &= checkDifference(info.principalPoint[0], 153.00758, eps, "principalPoint[0]", "interprete(..)");
    correct &= checkDifference(info.principalPoint[1], 103.48759, eps, "principalPoint[1]", "interprete(..)");
    correct &= checkDifference(info.omega,             76.107491, eps, "omega", "interprete(..)");
    correct &= checkDifference(info.phi,              -49.088123, eps, "phi", "interprete(..)");
    correct &= checkDifference(info.kappa,             171.64403, eps, "kappa", "interprete(..)");
    correct &= checkDifference(info.cameraLocation[0],  890.0155, eps, "cameraLocation[0]", "interprete(..)");
    correct &= checkDifference(info.cameraLocation[1], 786.18341, eps, "cameraLocation[1]", "interprete(..)");
    correct &= checkDifference(info.cameraLocation[2],-11.688845, eps, "cameraLocation[2]", "interprete(..)");

    return correct;
}


int main() {

    bool correct = true;
    cout << endl << "********************" << endl;
    cout << "Testing: Start" << endl;
    correct &= test_getCondition2D();
    correct &= test_getCondition3D();
    correct &= test_getDesignMatrix_camera();
    correct &= test_solve_dlt();
    correct &= test_decondition();
    correct &= test_calibrate();
    correct &= test_interprete();
    cout << "Testing: Done" << endl;
    if (correct)
        cout << "Everything seems (!) to be correct." << endl;
    else
        cout << "There seem to be problems." << endl;
    cout << endl << "********************" << endl << endl;

    
    cout << "Press enter to continue..." << endl;
    cin.get();

    return 0;
}

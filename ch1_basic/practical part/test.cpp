//============================================================================
// Name        : main.cpp
// Author      : Irina Nurutdinova
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Pcv1.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;
using namespace std;
using namespace pcv1;


void test_getConnectingLine(void){

    Mat v1 = (Mat_<float>(3,1) << 0, 0, 1);
    Mat v2 = (Mat_<float>(3,1) << 1, 1, 1);
    Mat lt = (Mat_<float>(3,1) << -1, 1, 0);
    Mat lc = getConnectingLine(v1, v2);
    
    if (sum(lc != lt).val[0] != 0){
        cout << "There seems to be a problem with getConnectingLine(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getScaleMatrix(void){

    Mat St = (Mat_<float>(3,3) << 3, 0, 0, 0, 3, 0, 0, 0, 1);
    Mat Sc = getScaleMatrix(3);
    
    if (sum(Sc != St).val[0] != 0){
        cout << "There seems to be a problem with getScaleMatrix(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getRotMatrix(void){

    Mat Rt = (Mat_<float>(3,3) << 1./sqrt(2), -1./sqrt(2), 0, 1./sqrt(2), 1./sqrt(2), 0, 0, 0, 1);
    Mat Rc = getRotMatrix(45);
    
    if (sum(Rc != Rt).val[0] != 0){
        cout << "There seems to be a problem with getRotMatrix(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getTranslMatrix(void){

    Mat Tt = (Mat_<float>(3,3) << 1, 0, -1, 0, 1, -1, 0, 0, 1);
    Mat Tc = getTranslMatrix(-1,-1);
    
    if (sum(Tc != Tt).val[0] != 0){
        cout << "There seems to be a problem with getTranslMatrix(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }	
}

void test_getH(void){

    Mat St = (Mat_<float>(3,3) << 3, 0, 0, 0, 3, 0, 0, 0, 1);
    Mat Rt = (Mat_<float>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    Mat Tt = (Mat_<float>(3,3) << 1, 0, -1, 0, 1, -1, 0, 0, 1);
    Mat Ht = (Mat_<float>(3,3) << 0, -3, 3, 3, 0, -3, 0, 0, 1);
    Mat Hc = getH(Tt, Rt, St);

    if (sum(Hc != Ht).val[0] != 0){
        cout << "There seems to be a problem with getH(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
}
void test_applyH(void){

    Mat H = (Mat_<float>(3,3) << 0, -3, 3, 3, 0, -3, 0, 0, 1);
    Mat v = (Mat_<float>(3,1) << 1, 1, 1);
    Mat vnt = (Mat_<float>(3,1) << 0, 0, 1);
    Mat l = (Mat_<float>(3,1) << -1, 1, 0);
    Mat lnt = (Mat_<float>(3,1) << -1, -1, 0)/3.;
    
    Mat vnc = applyH(v, H, GEOM_TYPE_POINT);
    Mat lnc = applyH(l, H, GEOM_TYPE_LINE);

    if (sum(vnc != vnt).val[0] != 0){
        cout << "There seems to be a problem with applyH(..) for points!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
    if (sum(lnc != lnt).val[0] != 0){
        cout << "There seems to be a problem with applyH(..) for lines!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
}

void test_isPointOnLine(void){
    
    Mat v = (Mat_<float>(3,1) << 1, 1, 1);
    Mat l = (Mat_<float>(3,1) << -1, 1, 0);

    if (!isPointOnLine(v, l)){
        cout << "There seems to be a problem with isPointOnLine(..)!" << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
        exit(-1);
    }
}



// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {
    
    test_getConnectingLine();
    test_getScaleMatrix();
    test_getRotMatrix();
    test_getTranslMatrix();
    test_getH();
    test_applyH();
    test_isPointOnLine();
    
    cout << "Finished basic testing: Everything seems to be fine." << endl;

    return 0;

}



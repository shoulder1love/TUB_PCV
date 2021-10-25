//============================================================================
// Name        : Pcv1.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Pcv1.h"

#include <stdexcept>

const float PI = 3.1415926;

using namespace std;
using namespace cv;

namespace pcv1 {

// calculates the joining line between two points
/*
p1, p2 : the two points in homogeneous coordinates
return : the joining line in homogeneous coordinates
*/
Mat getConnectingLine(Mat& p1, Mat& p2){

	Mat m;
	m = p1.cross(p2);
	return m;
}

// generates translation matrix T defined by translation (dx, dy)^T
/*
dx, dy : the translation in x- and y-direction, respectively
return : the resulting translation matrix
*/
Mat getTranslMatrix(float dx, float dy){
	Mat H = (Mat_<float>(3, 3) << 1, 0, dx, 0, 1, dy, 0, 0, 1);

	return H;
}

// generates rotation matrix R defined by angle phi
/*
phi		: the rotation angle in degree (!)
return	: the resulting rotation matrix
*/
Mat getRotMatrix(float phi){
	float a = phi / 180 * PI;
	Mat R = (Mat_<float>(3, 3) << cos(a), -sin(a), 0, sin(a), cos(a), 0, 0, 0, 1);
	return R;
}

// generates scaling matrix S defined by scaling factor lambda
/*
lambda	: the scaling parameter
return	: the resulting scaling matrix
*/
Mat getScaleMatrix(float lambda){
	Mat S = (Mat_<float>(3, 3) << lambda, 0, 0, 0, lambda, 0, 0, 0, 1);
	return S;
}

// combines translation-, rotation-, and scaling-matrices to a single transformation matrix H
/*
T			: translation matrix
R			: rotation matrix
S			: scaling matrix
return	: resulting homography
*/
Mat getH(Mat& T, Mat& R, Mat& S){
	Mat H = S * R * T;
	return H;
}

// transforms a geometric object by a given homography
/*
geomObj	: the geometric object to be transformed
H			: the homography defining the transformation
type		: the type of the geometric object (for now: only point and line)
return	: the transformed object
*/
Mat applyH(Mat& geomObj, Mat& H, GeometryType type){

    switch (type) {
        case GEOM_TYPE_POINT:
			return H * geomObj;
        case GEOM_TYPE_LINE:
			return H.inv().t()*geomObj;
        default:
            throw std::runtime_error("Unhandled case!");
    }
}

// checks if a point is on a line
/*
point		: the given point
line		: the given line
eps		: the used accuracy (set to 10^-5 by default (see header))
return	: true if point is on line
*/
bool isPointOnLine(Mat& point, Mat& line, float eps){
	Mat point_t = point.t();
	Mat n = point_t * line;
	float m = n.at<float>(0, 0);
	if (m <eps)
	{
		return true;
	}
	else
		return false;
}


// function loads input image, calls processing function and saves result
/*
fname	path to input image
*/
void run(const string &fname){

    //window names
    string win1 = string ("Image");

    // load image as gray-scale, path in argv[1]
    cout << "Load image: start" << endl;
    Mat inputImage;
    // TO DO !!!
	//load image
	//0: gray image
	inputImage = imread(fname,0);
 
    if (!inputImage.data){
        cout << "ERROR: image could not be loaded from " << fname << endl;
        cout << "Press enter to continue..." << endl;
        cin.get();
    }else
        cout << "Load image: done ( " << inputImage.rows << " x " << inputImage.cols << " )" << endl;
    
    // show input image
    namedWindow( win1.c_str(),CV_WINDOW_AUTOSIZE );
    imshow( win1.c_str(), inputImage );
    waitKey(50);

    // the two given points as OpenCV matrices
    Mat x(2, 1, CV_32FC1);
    x.at<float>(0, 0) = 2;
    x.at<float>(1, 0) = 3;
    Mat y(2, 1, CV_32FC1);
    y.at<float>(0, 0) = -4;
    y.at<float>(1, 0) = 5;
    
    // same points in homogeneous coordinates
    Mat v1(3, 1, CV_32FC1);
    v1.at<float>(0, 0) = x.at<float>(0, 0);
    v1.at<float>(1, 0) = x.at<float>(1, 0);
    v1.at<float>(2, 0) = 1;
    Mat v2(3, 1, CV_32FC1);
    // TO DO !!!
	v2.at<float>(0, 0) = y.at<float>(0, 0);
	v2.at<float>(1, 0) = y.at<float>(1, 0);
	v2.at<float>(2, 0) = 1;
    // define v2 as homogeneous version of y
    
    // print points
    cout << "point 1: " << v1.t() << "^T" << endl;
    cout << "point 2: " << v2.t() << "^T" << endl;
    cout << endl;
    
    // connecting line between those points in homogeneous coordinates
    Mat line = getConnectingLine(v1, v2);
    
    // print line
    cout << "joining line: " << line << "^T" << endl;
    cout << endl;    
    
    // the parameters of the transformation
    int dx = 6;				// translation in x
    int dy = -7;			// translation in y
    float phi = 15;		// rotation angle in degree
    float lambda = 8;		// scaling factor

    // matrices for transformation
    // calculate translation matrix
    Mat T = getTranslMatrix(dx, dy);
    // calculate rotation matrix
    Mat R = getRotMatrix(phi);
    // calculate scale matrix
    Mat S = getScaleMatrix(lambda);
    // combine individual transformations to a homography
    Mat H = getH(T, R, S);
    
    // print calculated matrices
    cout << "Translation matrix: " << endl;
    cout << T << endl;
    cout << endl;
    cout << "Rotation matrix: " << endl;
    cout << R << endl;
    cout << endl;
    cout << "Scaling matrix: " << endl;
    cout << S << endl;
    cout << endl;
    cout << "Homography: " << endl;
    cout << H << endl;
    cout << endl;

    // transform first point x (and print it)
    Mat v1_new = applyH(v1, H, GEOM_TYPE_POINT);
    cout << "new point 1: " << v1_new << "^T" << endl;
    // transform second point y (and print it)
    Mat v2_new = applyH(v2, H, GEOM_TYPE_POINT);
    cout << "new point 2: " << v2_new << "^T" << endl;
    cout << endl;
    // transform joining line (and print it)
    Mat line_new = applyH(line, H, GEOM_TYPE_LINE);
    cout << "new line: " << line_new << "^T" << endl;
    cout << endl;

    // check if transformed points are still on transformed line
    bool xOnLine = isPointOnLine(v1_new, line_new);
    bool yOnLine = isPointOnLine(v2_new, line_new);
    if (xOnLine)
        cout << "first point lies still on the line *yay*" << endl;
    else
        cout << "first point does not lie on the line *oh oh*" << endl;

    if (yOnLine)
        cout << "second point lies still on the line *yay*" << endl;
    else
        cout << "second point does not lie on the line *oh oh*" << endl;

	system("PAUSE");
}


}

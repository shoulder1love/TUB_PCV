#include "Helper.h"

#include <vector>
#include <string>

#include <fstream>

namespace pcv5 {

struct WinInfo { 
    cv::Mat img; 
    std::string name; 
    std::vector<cv::Point2f> pointList; 
};



// mouse call back to get points and draw circles
/*
event	specifies encountered mouse event
x,y	position of mouse pointer
flags	not used here
param	a struct containing used IplImage and window title
*/
void getPointsCB(int event, int x, int y, int flags, void* param){

    // cast to a structure
    WinInfo* win = (WinInfo*) param;

    switch(event){
        // if left mouse button was pressed
        case CV_EVENT_LBUTTONDOWN:{
            // create point representing mouse position
            cv::Point2f p(x,y);
            // draw green point
            cv::circle(win->img, p, 2, cv::Scalar(0, 255, 0), 2);
            // draw green circle
            cv::circle(win->img, p, 15, cv::Scalar(0, 255, 0), 2);
            // update image
            cv::imshow(win->name.c_str(), win->img);
            // add point to point list
            win->pointList.push_back(p);
        }break;
    }
}


int getPointsManual(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &p1, cv::Mat &p2)
{
    std::cout << std::endl;
    std::cout << "Please select at least four points by clicking at the corresponding image positions:" << std::endl;
    std::cout << "Firstly click at the point that shall be transformed (within the image to be attached), followed by a click on the corresponding point within the base image" << std::endl;
    std::cout << "Continue until you have collected as many point pairs as you wish" << std::endl;
    std::cout << "Stop the point selection by pressing any key" << std::endl << std::endl;

    
    WinInfo windowInfoBase = {
        img1.clone(),
        "Image 1",
        {}
    };
    
    WinInfo windowInfoAttach = {
        img2.clone(),
        "Image 2",
        {}
    };
    
    
    // show input images and install mouse callback
    cv::namedWindow( windowInfoBase.name.c_str(), 0 );
    cv::imshow( windowInfoBase.name.c_str(), windowInfoBase.img );
    cv::setMouseCallback(windowInfoBase.name.c_str(), getPointsCB, (void*) &windowInfoBase);
    

    cv::namedWindow( windowInfoAttach.name.c_str(), 0 );
    cv::imshow( windowInfoAttach.name.c_str(), windowInfoAttach.img );
    cv::setMouseCallback(windowInfoAttach.name.c_str(), getPointsCB, (void*) &windowInfoAttach);

    // wait until any key was pressed
    cv::waitKey(0);
    
    cv::destroyWindow( windowInfoBase.name.c_str() );
    cv::destroyWindow( windowInfoAttach.name.c_str() );

    // allocate memory for point-lists (represented as matrix)
    int numOfPoints = windowInfoBase.pointList.size();
    p1 = cv::Mat(3, numOfPoints, CV_32FC1);
    p2 = cv::Mat(3, numOfPoints, CV_32FC1);
    // read points from global variable, transform them into homogeneous coordinates
    for(int p = 0; p<numOfPoints; p++){
        p2.at<float>(0, p) = windowInfoAttach.pointList.at(p).x;
        p2.at<float>(1, p) = windowInfoAttach.pointList.at(p).y;
        p2.at<float>(2, p) = 1;
        p1.at<float>(0, p) = windowInfoBase.pointList.at(p).x;
        p1.at<float>(1, p) = windowInfoBase.pointList.at(p).y;
        p1.at<float>(2, p) = 1;
    }
    return numOfPoints;
}



/** 
 * @brief Draws line given in homogeneous representation into image
 * @param img the image to draw into
 * @param a The line parameters
 * @param b The line parameters
 * @param c The line parameters
 */
void drawEpiLine(cv::Mat& img, double a, double b, double c)
{

    // calculate intersection with image borders
    cv::Point p1 = cv::Point(-c/a, 0);						// schnittpunkt mit unterer bildkante (x-achse)
    cv::Point p2 = cv::Point(0, -c/b);						// schnittpunkt mit linker bildkante (y-achse)
    cv::Point p3 = cv::Point((-b*(img.rows-1)-c)/a, img.rows-1);		// schnittpunkt mit oberer bildkante
    cv::Point p4 = cv::Point(img.cols-1, (-a*(img.cols-1)-c)/b);		// schnittpunkt mit rechter bildkante

    // check start and end points
    cv::Point startPoint, endPoint, cur_p;
    startPoint.x = startPoint.y = endPoint.x = endPoint.y = 0;
    bool set_start = false;
    for(int p=0; p<4; p++){
        switch(p){
            case 0: cur_p = p1; break;
            case 1: cur_p = p2; break;
            case 2: cur_p = p3; break;
            case 3: cur_p = p4; break;
        }
        if ( (cur_p.x >= 0) and (cur_p.x < img.cols) and (cur_p.y >= 0) and (cur_p.y < img.rows) ){
            if (!set_start){
                startPoint = cur_p;
                set_start = true;
            }else{
                endPoint = cur_p;
            }
        }
    }

    // draw line
    cv::line(img, startPoint, endPoint, cv::Scalar(0,0,255), 1);
}



LevenbergMarquardt::LevenbergMarquardt(OptimizationProblem &optimizationProblem, std::unique_ptr<OptimizationProblem::State> initialState) : 
                            m_optimizationProblem(optimizationProblem), m_state(std::move(initialState))
{
    m_newState.reset(m_state->clone());
    m_jacobiMatrix.reset(m_optimizationProblem.createJacobiMatrix());
    
    m_residuals.create(m_optimizationProblem.getNumResiduals(), 1, CV_32F);
    m_newResiduals.create(m_optimizationProblem.getNumResiduals(), 1, CV_32F);
    m_update.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
    m_JtR.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
    m_diagonal.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
    

    m_state->computeResiduals(m_residuals.ptr<float>());
    m_lastError = m_residuals.dot(m_residuals);
//std::cout << "Initial error: " << m_lastError << std::endl;
}

void LevenbergMarquardt::iterate()
{
    m_state->computeJacobiMatrix(m_jacobiMatrix.get());
    m_jacobiMatrix->transposedMultiply(m_JtR.ptr<float>(), m_residuals.ptr<float>());
    
    m_jacobiMatrix->computeDiagJtJ(m_diagonal.ptr<float>());
    m_diagonal *= m_damping;
    
    /*
    auto preconditioner = [](cv::Mat &vec) {
        // todo: Jacobi?
    };
    */
    
    cv::Mat matMulTmp;
    matMulTmp.create(m_optimizationProblem.getNumResiduals(), 1, CV_32F);
    auto matMul = [&](cv::Mat &dst, const cv::Mat &src) {
        dst.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
        m_jacobiMatrix->multiply(matMulTmp.ptr<float>(), src.ptr<float>());
        m_jacobiMatrix->transposedMultiply(dst.ptr<float>(), matMulTmp.ptr<float>());
        dst += m_diagonal.mul(src);
    };

    // solve (JtJ + diagonal) . update = JtR for update
    m_update.setTo(0.0f);
    {
        cv::Mat r = m_JtR.clone();
//std::cout << "update " << m_update.t() << std::endl;
//std::cout << "m_JtR " << m_JtR.t() << std::endl;
        cv::Mat p = r.clone();
        float residual = r.dot(r);
        if (residual > 1e-8f) {
            for (unsigned cgIter = 0; cgIter < 100; cgIter++) {
                cv::Mat Ap;
                matMul(Ap, p);
                float pAp = p.dot(Ap);
                float alpha = residual / pAp;
                m_update += alpha * p;
                r -= alpha * Ap;
//std::cout << "pAp " << pAp << std::endl;
//std::cout << "alpha " << alpha << std::endl;
//std::cout << "update " << m_update.t() << std::endl;
                
                float newResidual = r.dot(r);
//std::cout << "CGD res: " << newResidual << std::endl;
                if (newResidual < 1e-8f) 
                    break;
                p *= newResidual / residual;
                p += r;
                residual = newResidual;
            }
        }
#if 0
        {
            cv::Mat y;
            matMul(y, m_update);
            r = y - m_JtR;
            std::cout << "cgd error: " << r.dot(r) << std::endl;
        }
#endif
    }

    m_state->update(m_update.ptr<float>(), m_newState.get());
    
    m_newState->computeResiduals(m_newResiduals.ptr<float>());
    float newError = m_newResiduals.dot(m_newResiduals);

//std::cout << "LM res: " << newError << " damping " << m_damping << std::endl;

    if (newError < m_lastError) {
        m_lastError = newError;
        std::swap(m_newState, m_state);
        std::swap(m_newResiduals, m_residuals);
        m_damping *= 0.9f;
    } else {
        m_damping *= 2.0f;
    }
}



}

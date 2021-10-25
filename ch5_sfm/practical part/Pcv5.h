//============================================================================
// Name        : Pcv5.h
// Author      : Ronny Haensch, Andreas Ley
// Version     : 2.0
// Copyright   : -
// Description : header file for the fourth PCV assignment
//============================================================================

#include "Helper.h"

#include <opencv2/opencv.hpp>

#include <string>

namespace pcv5 {

enum GeometryType {
    GEOM_TYPE_POINT,
    GEOM_TYPE_LINE,
};


// functions to be implemented
// --> please edit ONLY these functions!


/**
 * @brief Compute the fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns	the estimated fundamental matrix
 */
cv::Mat getFundamentalMatrix(cv::Mat& p1, cv::Mat& p2);


/**
 * @brief Estimate projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The projection matrix to be computed
 */
cv::Mat calibrate(cv::Mat& points2D, cv::Mat& points3D);


/**
 * @brief Extract and prints information about interior and exterior orientation from camera
 * @param P The 3x4 projection matrix
 * @param K Matrix for returning the computed internal calibration
 * @param R Matrix for returning the computed rotation
 * @param info Structure for returning the interpretation such as principal distance
 */
void interprete(cv::Mat &P, cv::Mat &K, cv::Mat &R, ProjectionMatrixInterpretation &info);

/**
 * @brief Define the design matrix as needed to compute fundamental matrix
 * @param p1 first set of points
 * @param p2 second set of points
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_fundamental(cv::Mat& p1, cv::Mat& p2); 


/**
 * @brief Define the design matrix as needed to compute projection matrix
 * @param points2D Set of 2D points within the image
 * @param points3D Set of 3D points at the object
 * @returns The design matrix to be computed
 */
cv::Mat getDesignMatrix_camera(cv::Mat& points2D, cv::Mat& points3D);

/**
 * @brief Enforce rank of 2 on fundamental matrix
 * @param F The matrix to be changed
 */
void forceSingularity(cv::Mat& F);

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated fundamental matrix
 */
cv::Mat solve_dlt_F(cv::Mat& A);

/**
 * @brief Solve homogeneous equation system by usage of SVD
 * @param A The design matrix
 * @returns The estimated projection matrix
 */
cv::Mat solve_dlt_P(cv::Mat& A);

/**
 * @brief Decondition a fundamental matrix that was estimated from conditioned points
 * @param T1 Conditioning matrix of set of 2D image points
 * @param T2 Conditioning matrix of set of 2D image points
 * @param F Conditioned fundamental matrix that has to be un-conditioned (in-place)
 */
void decondition_F(cv::Mat& T1, cv::Mat& T2, cv::Mat& F);


/**
 * @brief Decondition a projection matrix that was estimated from conditioned point clouds
 * @param T_2D Conditioning matrix of set of 2D image points
 * @param T_3D Conditioning matrix of set of 3D object points
 * @param P Conditioned projection matrix that has to be un-conditioned (in-place)
 */
void decondition_P(cv::Mat& T_2D, cv::Mat& T_3D, cv::Mat& P);

/**
 * @brief Calculate geometric error of estimated fundamental matrix
 * @details Implement the mean "Sampson distance"
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @returns		geometric error
 */
double getError(cv::Mat p1, cv::Mat p2, cv::Mat& F);

/**
 * @brief Count the number of inliers of an estimated fundamental matrix
 * @param p1		first set of points
 * @param p2		second set of points
 * @param F		fundamental matrix
 * @param threshold Maximal "Sampson distance" to sti9ll be countes as an inlier
 * @returns		Number of inliers
 */
unsigned countInliers(cv::Mat& p1, cv::Mat& p2, cv::Mat& F, float threshold);



cv::Mat estimateFundamentalRANSAC(cv::Mat& p1, cv::Mat& p2, unsigned numIterations);


/**
 * @brief Apply transformation to set of points
 * @param H Matrix representing the transformation
 * @param geomObj Matrix with input objects (one per column)
 * @param type The type of the geometric object (for now: only point and line)
 * @returns Transformed objects (one per column)
 */
cv::Mat applyH(cv::Mat& geomObj, cv::Mat& H, GeometryType type);

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix
 */
cv::Mat getCondition2D(cv::Mat& p);

/**
 * @brief Get the conditioning matrix of given points
 * @param p The points as matrix
 * @returns The condition matrix 
 */
cv::Mat getCondition3D(cv::Mat& p);


cv::Mat linearTriangulation(const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& x1, const cv::Mat& x2);


cv::Mat computeCameraPose(const cv::Mat &K, const cv::Mat &p1, const cv::Mat &p2);


struct NumUpdateParams {
    enum {
        TRACK = 4,
        CAMERA = 6,
        INTERNAL_CALIB = 3,
    };
};

struct KeyPoint
{
    Vector<2> location;
    unsigned trackIdx;
    float weight;
};

struct Camera {
    unsigned internalCalibIdx;
    std::vector<KeyPoint> keypoints;
};

struct Scene {
    std::vector<Camera> cameras;
    unsigned numTracks;
    unsigned numInternalCalibs;
};


class BundleAdjustment : public OptimizationProblem {
    public:
        BundleAdjustment(Scene &scene);
        
        class BAState;

        class BAJacobiMatrix : public JacobiMatrix {
            public:
                BAJacobiMatrix(const Scene &scene);
                
                virtual ~BAJacobiMatrix() = default;

                virtual void multiply(float * __restrict dst, const float * __restrict src) const override;
                virtual void transposedMultiply(float * __restrict dst, const float * __restrict src) const override;
                virtual void computeDiagJtJ(float * __restrict dst) const override;
            protected:
                struct RowBlock {
                    unsigned internalCalibIdx;
                    unsigned cameraIdx;
                    unsigned keypointIdx;
                    unsigned trackIdx;
                    
                    Matrix<2, NumUpdateParams::INTERNAL_CALIB> J_internalCalib;
                    Matrix<2, NumUpdateParams::CAMERA> J_camera;
                    Matrix<2, NumUpdateParams::TRACK> J_track;
                };
                unsigned m_internalCalibOffset;
                unsigned m_cameraOffset;
                unsigned m_trackOffset;
                unsigned m_totalUpdateParams;
                std::vector<RowBlock> m_rows;
                
                friend class BAState;
        };

        class BAState : public State {
            public:
                struct TrackState {
                    Vector<4> location;
                };

                struct CameraState {
                    Matrix<4, 4> H;
                };

                struct InternalCalibrationState {
                    Matrix<3, 3> K;
                };

                std::vector<TrackState> m_tracks;
                std::vector<CameraState> m_cameras;
                std::vector<InternalCalibrationState> m_internalCalibs;
                const Scene &m_scene;

                BAState(const Scene &scene);
                virtual ~BAState() = default;

                virtual State* clone() const override;
                virtual void computeResiduals(float *residuals) const override;
                virtual void computeJacobiMatrix(JacobiMatrix *dst) const override;
                virtual void update(const float *update, State *dst) const override;
                
                void weighDownOutliers();
        };

        virtual JacobiMatrix* createJacobiMatrix() const override;
        
        void downweightOutlierKeypoints(BAState &state);
    protected:
        Scene &m_scene;
};


Scene buildScene(const std::vector<std::string> &imagesFilenames);
void produceInitialState(const Scene &scene, const Matrix<3, 3> &initialInternalCalib, BundleAdjustment::BAState &state);

}


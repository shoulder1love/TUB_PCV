#ifndef HELPER_H
#define HELPER_H


#include <opencv2/opencv.hpp>
#include <memory>


namespace pcv5 {


/// Interpretation of the internal and external parts of a projection matrix.
struct ProjectionMatrixInterpretation
{
    /// Principal distance or focal length
    float principalDistance;
    /// Skew as an angle and in degrees
    float skew;
    /// Aspect ratio of the pixels
    float aspectRatio;
    /// Location of principal point in image (pixel) coordinates
    float principalPoint[2];
    /// Camera rotation angle 1/3
    float omega;
    /// Camera rotation angle 2/3
    float phi;
    /// Camera rotation angle 3/3
    float kappa;
    /// 3D camera location in world coordinates
    float cameraLocation[3];
};

/// Unreasonable replacement for s.th. reasonable like the Eigen library. But I didn't want to introduce more dependencies.
template<unsigned Rows, unsigned Cols, typename Type = float>
class Matrix
{
    public:
        enum {
            ROWS = Rows,
            COLS = Cols
        };
        
        void setZero() { for (unsigned i = 0; i < ROWS*COLS; i++) m_elems[i] = Type(0); }
        void setIdentity() { 
            for (unsigned i = 0; i < ROWS; i++)
                for (unsigned j = 0; j < COLS; j++) 
                    (*this)(i, j) = i == j?Type(1):Type(0);
        }
        
        inline Type &operator()(unsigned r, unsigned c = 0) { return m_elems[r*COLS+c]; }
        inline const Type &operator()(unsigned r, unsigned c = 0) const { return m_elems[r*COLS+c]; }
        
        inline Type &operator[](unsigned idx) { return m_elems[idx]; }
        inline const Type &operator[](unsigned idx) const { return m_elems[idx]; }
        
        template<unsigned otherCols>
        Matrix<ROWS, otherCols, Type> operator*(const Matrix<COLS, otherCols, Type> &rhs) const {
            Matrix<ROWS, otherCols, Type> result;
            
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < otherCols; c++) {
                    Type sum = Type(0);
                    for (unsigned i = 0; i < COLS; i++)
                        sum += (*this)(r, i) * rhs(i, c);
                    result(r, c) = sum;
                }
            
            return result;
        }

        Matrix<ROWS, COLS, Type> operator+(const Matrix<COLS, COLS, Type> &rhs) const {
            Matrix<ROWS, COLS, Type> result;
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < COLS; c++)
                    result(r, c) = (*this)(r, c) + rhs(r, c);
            return result;
        }

        Matrix<ROWS, COLS, Type> operator-(const Matrix<COLS, COLS, Type> &rhs) const {
            Matrix<ROWS, COLS, Type> result;
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < COLS; c++)
                    result(r, c) = (*this)(r, c) + rhs(r, c);
            return result;
        }

        const Matrix<ROWS, COLS, Type> &operator+=(const Matrix<COLS, COLS, Type> &rhs) {
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < COLS; c++)
                    (*this)(r, c) += rhs(r, c);
            return *this;
        }

        const Matrix<ROWS, COLS, Type> &operator-=(const Matrix<COLS, COLS, Type> &rhs) {
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < COLS; c++)
                    (*this)(r, c) -= rhs(r, c);
            return *this;
        }
        
        
        Matrix<ROWS, COLS, Type> operator+(const Type &scalar) const {
            return elemWiseOp([&scalar](float f)->float{
                return f + scalar;
            });
        }

        Matrix<ROWS, COLS, Type> operator-(const Type &scalar) const {
            return elemWiseOp([&scalar](float f)->float{
                return f - scalar;
            });
        }

        Matrix<ROWS, COLS, Type> operator*(const Type &scalar) const {
            return elemWiseOp([&scalar](float f)->float{
                return f * scalar;
            });
        }

        Matrix<ROWS, COLS, Type> operator/(const Type &scalar) const {
            return elemWiseOp([&scalar](float f)->float{
                return f / scalar;
            });
        }

        
        const Matrix<ROWS, COLS, Type> &operator+=(const Type &scalar) {
            return elemWiseOpInplace([&scalar](float f)->float{
                return f + scalar;
            });
        }

        const Matrix<ROWS, COLS, Type> &operator-=(const Type &scalar) {
            return elemWiseOpInplace([&scalar](float f)->float{
                return f - scalar;
            });
        }

        const Matrix<ROWS, COLS, Type> &operator*=(const Type &scalar) {
            return elemWiseOpInplace([&scalar](float f)->float{
                return f * scalar;
            });
        }

        const Matrix<ROWS, COLS, Type> &operator/=(const Type &scalar) {
            return elemWiseOpInplace([&scalar](float f)->float{
                return f / scalar;
            });
        }

        
        const Matrix<ROWS, COLS, Type> &operator=(const Type &scalar) {
            return elemWiseOpInplace([&scalar](float f)->float{
                return scalar;
            });
        }
        
        
        Matrix<ROWS-1, COLS, Type> dropLastRow() const {
            Matrix<ROWS-1, COLS, Type> result;
            for (unsigned i = 0; i < (ROWS-1)*COLS; i++)
                result[i] = (*this)[i];
            return result;
        }
/*
        operator Type() const { 
            static_assert(ROWS == 1, "Matrix with Rows > 1 can't be cast into a scalar");
            static_assert(COLS == 1, "Matrix with Cols > 1 can't be cast into a scalar");
            return m_elems[0]; 
        }
*/
    protected:
        Type m_elems[ROWS*COLS];

        template<class Op>
        Matrix<ROWS, COLS, Type> elemWiseOp(Op op) const {
            Matrix<ROWS, COLS, Type> result;
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < COLS; c++)
                    result(r, c) = op((*this)(r, c));
            return result;
        }
        
        template<class Op>
        const Matrix<ROWS, COLS, Type> &elemWiseOpInplace(Op op) {
            for (unsigned r = 0; r < ROWS; r++) 
                for (unsigned c = 0; c < COLS; c++)
                    (*this)(r, c) = op((*this)(r, c));
            return *this;
        }
        
};

template<unsigned Rows, unsigned Cols, typename Type>
Matrix<Rows-1, Cols, Type> dropLastRow(const Matrix<Rows, Cols, Type> &m) {
    return m.dropLastRow();
}



template<unsigned rows, typename Type = float>
using Vector = Matrix<rows, 1, Type>;

template<unsigned rows, typename Type>
Type innerProd(const Vector<rows, Type> &lhs, const Vector<rows, Type> &rhs) {
    Type result = Type(0);
    for (unsigned i = 0; i < rows; i++)
        result += lhs(i) * rhs(i);
    return result;
}

template<unsigned rows, typename Type>
Matrix<rows, rows, Type> outerProd(const Vector<rows, Type> &lhs, const Vector<rows, Type> &rhs) {
    Matrix<rows, rows, Type> result;
    for (unsigned i = 0; i < rows; i++)
        for (unsigned j = 0; j < rows; j++)
            result(i, j) = lhs(i) * rhs(j);
    return result;
}


template<unsigned rows, typename Type>
Vector<rows-1, Type> hom2eucl(const Vector<rows, Type> &vec) {
    Vector<rows-1, Type> result;
    for (unsigned i = 0; i < rows-1; i++)
        result(i) = vec(i) / vec(rows-1);
    return result;
}


template<typename Type>
Matrix<4, 4, Type> translationMatrix(Type x, Type y, Type z)
{
    Matrix<4, 4, Type> result;
    result.setIdentity();
    result(0, 3) = x;
    result(1, 3) = y;
    result(2, 3) = z;
    return result;
}

template<typename Type>
Matrix<4, 4, Type> rotationMatrixX(Type radAngle)
{
    Matrix<4, 4, Type> result;
    result.setIdentity();
    result(1, 1) = std::cos(radAngle);
    result(1, 2) = -std::sin(radAngle);
    result(2, 1) = std::sin(radAngle);
    result(2, 2) = std::cos(radAngle);
    return result;
}

template<typename Type>
Matrix<4, 4, Type> rotationMatrixY(Type radAngle)
{
    Matrix<4, 4, Type> result;
    result.setIdentity();
    result(0, 0) = std::cos(radAngle);
    result(0, 2) = std::sin(radAngle);
    result(2, 0) = -std::sin(radAngle);
    result(2, 2) = std::cos(radAngle);
    return result;
}

template<typename Type>
Matrix<4, 4, Type> rotationMatrixZ(Type radAngle)
{
    Matrix<4, 4, Type> result;
    result.setIdentity();
    result(0, 0) = std::cos(radAngle);
    result(0, 1) = -std::sin(radAngle);
    result(1, 0) = std::sin(radAngle);
    result(1, 1) = std::cos(radAngle);
    return result;
}


/**
 * @brief Displays two images and catches the point pairs marked by left mouse clicks.
 * @details Points will be in homogeneous coordinates.
 * @param img1 The first image
 * @param img2 The second image
 * @param p1 Points within the first image (returned in the matrix by this method)
 * @param p2 Points within the second image (returned in the matrix by this method)
 */
int getPointsManual(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &p1, cv::Mat &p2);

/** 
 * @brief Draws line given in homogeneous representation into image
 * @param img the image to draw into
 * @param a The line parameters
 * @param b The line parameters
 * @param c The line parameters
 */
void drawEpiLine(cv::Mat& img, double a, double b, double c);


class OptimizationProblem
{
    public:
        class JacobiMatrix {
            public:
                virtual ~JacobiMatrix() = default;
                
                virtual void multiply(float * __restrict dst, const float * __restrict src) const = 0;
                virtual void transposedMultiply(float * __restrict dst, const float * __restrict src) const = 0;
                virtual void computeDiagJtJ(float * __restrict dst) const = 0;
        };

        class State {
            public:
                virtual ~State() = default;

                virtual State* clone() const = 0;
                virtual void computeResiduals(float *residuals) const = 0;
                virtual void computeJacobiMatrix(JacobiMatrix *dst) const = 0;
                virtual void update(const float *update, State *dst) const = 0;
        };
        
        inline unsigned getNumUpdateParameters() const { return m_numUpdateParameters; }
        inline unsigned getNumResiduals() const { return m_numResiduals; }
        
        virtual JacobiMatrix* createJacobiMatrix() const = 0;
    protected:
        unsigned m_numUpdateParameters;
        unsigned m_numResiduals;
        std::unique_ptr<State> m_state;
};

class LevenbergMarquardt
{
    public:
        LevenbergMarquardt(OptimizationProblem &optimizationProblem, std::unique_ptr<OptimizationProblem::State> initialState);
        
        void iterate();

        inline float getLastError() const { return m_lastError; }
        inline float getDamping() const { return m_damping; }
        
        const OptimizationProblem::State *getState() const { return m_state.get(); }
    protected:
        OptimizationProblem &m_optimizationProblem;
        
        std::unique_ptr<OptimizationProblem::State> m_state;
        std::unique_ptr<OptimizationProblem::State> m_newState;
        std::unique_ptr<OptimizationProblem::JacobiMatrix> m_jacobiMatrix;
        cv::Mat m_diagonal;
               
        cv::Mat m_residuals;
        cv::Mat m_newResiduals;
        cv::Mat m_JtR;
        cv::Mat m_update;
        
        float m_lastError = 0.0f;
        float m_damping = 1.0f;
        
        cv::Mat m_conditioner;
};


}


#endif // HELPER_H

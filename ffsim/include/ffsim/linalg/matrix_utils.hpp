#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace ffsim
{
namespace linalg
{
using namespace Eigen;
using Complex = std::complex<double>;
inline bool array_all_close(const MatrixXcd& mat1, const MatrixXcd& mat2, double rtol = 1e-5,
                            double atol = 1e-8)
{
    if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols())
    {
        return false;
    }

    for (int i = 0; i < mat1.rows(); ++i)
    {
        for (int j = 0; j < mat1.cols(); ++j)
        {
            const Complex& a = mat1(i, j);
            const Complex& b = mat2(i, j);
            if (std::abs(a - b) > atol + rtol * std::abs(b))
            {
                return false;
            }
        }
    }
    return true;
}

inline bool is_real_symmetric(const MatrixXcd& mat, double rtol = 1e-5, double atol = 1e-8)
{
    if (mat.rows() != mat.cols())
    {
        return false;
    }

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            if (std::abs(mat(i, j).imag()) > atol)
            {
                return false;
            }
        }
    }
    return array_all_close(mat, mat.transpose(), rtol, atol);
}

inline bool is_unitary(const MatrixXcd& mat, double rtol = 1e-5, double atol = 1e-8)
{
    if (mat.rows() != mat.cols())
    {
        return false;
    }

    MatrixXcd I = MatrixXcd::Identity(mat.rows(), mat.cols());
    return array_all_close(mat * mat.adjoint(), I, rtol, atol);
}

} // namespace linalg
} // namespace ffsim
#endif // MATRIX_UTILS_HPP

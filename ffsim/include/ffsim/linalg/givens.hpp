/*
# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
*/

#ifndef GIVENS_HPP
#define GIVENS_HPP

#include <Eigen/Dense>
#include <cassert>
#include <complex>
#include <vector>

namespace ffsim
{
namespace linalg
{

using namespace Eigen;
using Complex = std::complex<double>;

extern "C"
{
    void zrotg_(Complex* a, Complex* b, double* c, Complex* s);
}

std::tuple<double, Complex, Complex> zrotg(Complex ca, Complex cb)
{
    if (std::abs(ca) < 1e-12)
    {
        return {0.0, {1.0, 0.0}, {0.0, 0.0}};
    }
    if (std::abs(cb) < 1e-12)
    {
        return {1.0, {0.0, 0.0}, {0.0, 0.0}};
    }
    double c;
    Complex s;
    Complex a = ca;
    Complex b = cb;

    zrotg_(&a, &b, &c, &s);
    return {c, s, a};
};

/**
 * @brief Computes parameters for a complex Givens rotation.
 *
 * This is analogous to the BLAS/LAPACK `zrotg` function. It computes
 * the Givens rotation such that it zeroes out `cb` in the vector `[ca, cb]^T`.
 *
 * @param ca First complex value
 * @param cb Second complex value to be eliminated
 * @return Tuple of (c, s, r) where:
 *         - `c` is the cosine,
 *         - `s` is the sine (complex),
 *         - `r` is the rotated result at index 0
 */
std::tuple<double, Complex, Complex> zrotg(Complex ca, Complex cb);

/**
 * @brief Applies a complex Givens rotation to two complex vectors `x` and `y`.
 *
 * Modifies both `x` and `y` in-place. Each element is rotated by the same `(c,
 * s)` parameters.
 *
 * @param x First vector
 * @param y Second vector
 * @param c Cosine component of the rotation
 * @param s Sine component of the rotation (complex)
 */
inline void zrot(VectorXcd& x, VectorXcd& y, double c, Complex s)
{
    int n = static_cast<int>(std::min(x.size(), y.size()));
    for (int i = 0; i < n; ++i)
    {
        Complex temp = c * x[i] + s * y[i];
        y[i] = c * y[i] - std::conj(s) * x[i];
        x[i] = temp;
    }
};

/**
 * @brief Represents a single Givens rotation.
 *
 * A Givens rotation is defined by a cosine `c`, sine `s`, and the two indices
 * `(i, j)` that it acts on. This structure stores the parameters needed to
 * construct the rotation matrix or apply it to vectors or matrices.
 */
struct GivensRotation
{
    double c;    ///< Cosine part of the rotation
    Complex s;   ///< Sine part of the rotation (complex)
    size_t i, j; ///< Indices of the rows (or columns) involved in the rotation

    /**
     * @brief Constructs a Givens rotation object.
     * @param c Cosine coefficient
     * @param s Sine coefficient
     * @param first_index First index involved
     * @param second_index Second index involved
     */
    GivensRotation(double c, Complex s, size_t first_index, size_t second_index) : c(c), s(s), i(first_index), j(second_index) // NOLINT(bugprone-easily-swappable-parameters)
    {}
};

/**
 * @brief Decomposes a complex square matrix into Givens rotations.
 *
 * Performs a QR-like decomposition of a complex matrix using a sequence of
 * alternating left and right Givens rotations. Returns the final diagonal
 * elements and the sequence of applied right-side rotations (including
 * converted left-rotations).
 *
 * This decomposition is used to diagonalize a Hermitian or general complex
 * matrix via successive elimination of off-diagonal elements.
 *
 * @param mat A square complex matrix (Hermitian or general)
 * @return A pair of:
 *   - A list of GivensRotation representing the transformation steps
 *   - A vector of complex values representing the resulting diagonal
 */

std::pair<std::vector<GivensRotation>, VectorXcd> givens_decomposition(const MatrixXcd& mat)
{
    int n = static_cast<int>(mat.rows());
    MatrixXcd current_matrix = mat;

    std::vector<GivensRotation> left_rotations;
    std::vector<GivensRotation> right_rotations;

    for (int i = 0; i < n - 1; ++i)
    {
        if (i % 2 == 0)
        {
            for (int j = 0; j < i + 1; ++j)
            {
                int target = i - j;
                int row = n - j - 1;
                if (std::abs(current_matrix(row, target)) > 1e-12)
                {
                    auto [c, s, _] =
                        zrotg(current_matrix(row, target + 1), current_matrix(row, target));
                    right_rotations.emplace_back(c, s, target + 1, target);
                    VectorXcd col1 = current_matrix.col(target + 1);
                    VectorXcd col2 = current_matrix.col(target);
                    zrot(col1, col2, c, s);

                    current_matrix.col(target + 1) = col1;
                    current_matrix.col(target) = col2;
                }
            }
        }
        else
        {
            for (int j = 0; j < i + 1; ++j)
            {
                int target = n - i + j - 1;
                int col = j;
                if (std::abs(current_matrix(target, col)) > 1e-9)
                {
                    auto [c, s, _] =
                        zrotg(current_matrix(target - 1, col), current_matrix(target, col));
                    left_rotations.emplace_back(c, s, target - 1, target);
                    VectorXcd row1 = current_matrix.row(target - 1);
                    VectorXcd row2 = current_matrix.row(target);
                    zrot(row1, row2, c, s);
                    current_matrix.row(target - 1) = row1;
                    current_matrix.row(target) = row2;
                }
            }
        }
    }

    std::reverse(left_rotations.begin(), left_rotations.end());

    for (const auto& rot : left_rotations)
    {
        double c = rot.c;
        Complex s = std::conj(rot.s) * current_matrix(static_cast<Index>(rot.i), static_cast<Index>(rot.i));
        auto [new_c, new_s, _] = zrotg(c * current_matrix(static_cast<Index>(rot.j), static_cast<Index>(rot.j)), s);
        right_rotations.emplace_back(new_c, -std::conj(new_s), rot.i, rot.j);

        Matrix2cd givens_mat;
        givens_mat << new_c, -new_s, std::conj(new_s), new_c;
        givens_mat(0, 0) *= current_matrix(static_cast<Index>(rot.i), static_cast<Index>(rot.i));
        givens_mat(1, 0) *= current_matrix(static_cast<Index>(rot.i), static_cast<Index>(rot.i));
        givens_mat(0, 1) *= current_matrix(static_cast<Index>(rot.j), static_cast<Index>(rot.j));
        givens_mat(1, 1) *= current_matrix(static_cast<Index>(rot.j), static_cast<Index>(rot.j));

        auto [c2, s2, _2] = zrotg(givens_mat(1, 1), givens_mat(1, 0));
        Matrix2cd givens_mat2;
        givens_mat2 << c2, s2, -std::conj(s2), c2;
        Matrix2cd final_mat = givens_mat * givens_mat2;
        current_matrix(static_cast<Index>(rot.i), static_cast<Index>(rot.i)) = final_mat(0, 0);
        current_matrix(static_cast<Index>(rot.j), static_cast<Index>(rot.j)) = final_mat(1, 1);
    }
    return {right_rotations, current_matrix.diagonal()};
}

} // namespace linalg
} // namespace ffsim

#endif // GIVENS_HPP
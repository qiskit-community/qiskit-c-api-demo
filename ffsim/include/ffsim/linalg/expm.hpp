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

#ifndef EXPM_HPP
#define EXPM_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace ffsim
{
namespace linalg
{
using namespace Eigen;
/**
 * @brief Computes the matrix exponential of a given matrix A.
 * @param A The input matrix for which the exponential is to be computed.
 * @return The matrix exponential of the input matrix A.
 */
MatrixXcd expm(const MatrixXcd &A)
{
    return A.exp();
}
} // namespace linalg
} // namespace ffsim
#endif // EXPM_HPP
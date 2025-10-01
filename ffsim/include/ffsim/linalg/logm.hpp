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

#ifndef LOGM_HPP
#define LOGM_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace ffsim
{
namespace linalg
{
using namespace Eigen;
/**
 * @brief Computes the matrix logarithm of a given matrix A.
 * @param A The input matrix for which the logarithm is to be computed.
 * @return The matrix logarithm of the input matrix A.
 */
MatrixXcd logm(const MatrixXcd &A)
{
    return A.log();
}
} // namespace linalg
} // namespace ffsim
#endif // LOGM_HPP
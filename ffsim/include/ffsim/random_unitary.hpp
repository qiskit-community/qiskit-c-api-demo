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

#ifndef RANDOM_UNITARY_HPP
#define RANDOM_UNITARY_HPP

#include <Eigen/Dense>
#include <complex>
#include <random>

namespace ffsim
{
using namespace Eigen;
using Complex = std::complex<double>;

/**
 * @brief Generates a random unitary matrix of size N x N.
 * @details This function uses the QR decomposition method to generate a random
 * unitary matrix. It creates a random matrix, performs QR decomposition, and
 * returns the unitary matrix Q. The resulting matrix is guaranteed to be
 * unitary.
 * @param N The size of the unitary matrix.
 * @return A random unitary matrix of size N x N.
 */
MatrixXcd random_unitary(int N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    MatrixXcd A = MatrixXcd::Zero(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i, j) = Complex(dist(gen), dist(gen));
        }
    }

    HouseholderQR<MatrixXcd> qr(A);
    MatrixXcd Q = qr.householderQ();
    return Q;
}
} // namespace ffsim
#endif

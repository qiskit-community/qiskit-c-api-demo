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

#ifndef PHASE_SHIFT_HPP
#define PHASE_SHIFT_HPP

#include <Eigen/Dense>
#include <complex>

namespace ffsim
{
namespace gates
{
using namespace Eigen;
using Complex = std::complex<double>;

/**
 * @brief Applies a phase shift to a matrix.
 * @param mat The input matrix to be modified.
 * @param phase The phase shift to be applied.
 * @param indices The indices of the rows/columns to which the phase shift is
 * applied.
 */
void apply_phase_shift_in_place(
    MatrixXcd &mat, const Complex &phase, const std::vector<size_t> &indices
)
{
    for (size_t row : indices) {
        mat.row(static_cast<Index>(row)) *= phase;
    }
}

} // namespace gates
} // namespace ffsim
#endif // PHASE_SHIFT_HPP
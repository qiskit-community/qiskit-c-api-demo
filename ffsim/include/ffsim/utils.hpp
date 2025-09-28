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

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "gates/orbital_rotation.hpp"
#include "linalg/expm.hpp"
#include "linalg/logm.hpp"
#include "linalg/matrix_utils.hpp"
#include <Eigen/Dense>

namespace ffsim
{
using namespace gates;
using namespace Eigen;

/**
 * @brief Converts an orbital rotation matrix to a vector of parameters.
 * @details This function takes an orbital rotation matrix and converts it to a
 * vector of parameters.
 * @param orbital_rotation The input orbital rotation matrix.
 * @param real A boolean indicating whether the matrix is real (true) or complex
 * (false).
 * @return A vector of parameters representing the orbital rotation matrix.
 */
VectorXd orbital_rotation_to_parameters(const MatrixXcd& orbital_rotation, bool real)
{
    if (real)
    {
        for (int i = 0; i < orbital_rotation.rows(); ++i)
        {
            for (int j = 0; j < orbital_rotation.cols(); ++j)
            {
                if (std::abs(orbital_rotation(i, j).imag()) > 1e-12)
                {
                    throw std::runtime_error("real was set to True, but the orbital "
                                             "rotation has a complex data type. "
                                             "Try passing an orbital rotation with a "
                                             "real-valued data type, or else "
                                             "set real=False.");
                }
            }
        }
    }

    int norb = static_cast<int>(orbital_rotation.rows());
    std::vector<std::pair<uint64_t, uint64_t>> triu_indices_no_diag;

    for (int i = 0; i < norb; ++i)
    {
        for (int j = i + 1; j < norb; ++j)
        {
            triu_indices_no_diag.emplace_back(i, j);
        }
    }

    MatrixXcd mat = linalg::logm(orbital_rotation);
    int param_len = real ? norb * (norb - 1) / 2 : norb * norb;

    VectorXd params = VectorXd::Zero(param_len);
    for (size_t idx = 0; idx < triu_indices_no_diag.size(); ++idx)
    {
        auto [i, j] = triu_indices_no_diag[idx];
        params(static_cast<Index>(idx)) = mat(static_cast<Index>(i), static_cast<Index>(j)).real();
    }

    if (!real)
    {
        std::vector<std::pair<uint64_t, uint64_t>> triu_indices;
        for (int i = 0; i < norb; ++i)
        {
            for (int j = i; j < norb; ++j)
            {
                triu_indices.emplace_back(i, j);
            }
        }
        for (size_t idx = 0; idx < triu_indices.size(); ++idx)
        {
            auto [i, j] = triu_indices[idx];
            params(static_cast<Index>(idx + triu_indices_no_diag.size())) = mat(static_cast<Index>(i), static_cast<Index>(j)).imag();
        }
    }

    return params;
};

/**
 * @brief Converts a vector of parameters to an orbital rotation matrix.
 * @details This function takes a vector of parameters and converts it to an
 * orbital rotation matrix.
 * @param params The input vector of parameters.
 * @param norb The number of orbitals.
 * @param real A boolean indicating whether the matrix is real (true) or complex
 * (false).
 * @return The resulting orbital rotation matrix.
 */
MatrixXcd orbital_rotation_from_parameters(const VectorXcd& params, int norb, bool real)
{
    std::vector<std::pair<uint64_t, uint64_t>> triu_indices_no_diag;
    for (int i = 0; i < norb; ++i)
    {
        for (int j = i + 1; j < norb; ++j)
        {
            triu_indices_no_diag.emplace_back(i, j);
        }
    }

    MatrixXcd generator = MatrixXcd::Zero(norb, norb);
    if (!real)
    {
        std::vector<std::pair<uint64_t, uint64_t>> triu_indices;
        for (int i = 0; i < norb; ++i)
        {
            for (int j = i; j < norb; ++j)
            {
                triu_indices.emplace_back(i, j);
            }
        }
        for (size_t idx = 0; idx < triu_indices.size(); ++idx)
        {
            auto [i, j] = triu_indices[idx];
            Complex imag_param_val = params[static_cast<Index>(idx + triu_indices_no_diag.size())];
            generator(static_cast<Index>(i), static_cast<Index>(j)) = Complex(0.0, imag_param_val.real());
            generator(static_cast<Index>(j), static_cast<Index>(i)) = Complex(0.0, imag_param_val.real());
        }
    }

    for (size_t idx = 0; idx < triu_indices_no_diag.size(); ++idx)
    {
        auto [i, j] = triu_indices_no_diag[idx];
        Complex param_val = params[static_cast<Index>(idx)];
        generator(static_cast<Index>(i), static_cast<Index>(j)) += Complex(param_val.real(), 0.0);
        generator(static_cast<Index>(j), static_cast<Index>(i)) -= Complex(param_val.real(), 0.0);
    }
    return linalg::expm(generator);
};

/**
 * @brief Rounds a value for use in the acos function.
 * @details This function ensures that the value passed to acos is within the
 * valid range [-1, 1]. It clamps the value to this range to avoid NaN results.
 * @param value The input value to be rounded.
 * @return The rounded value, clamped to the range [-1, 1].
 */
double round_for_acos(double value)
{
    constexpr double EPSILON = 1e-12;
    if (value > 1.0 && value < 1.0 + EPSILON)
    {
        return 1.0;
    }
    else if (value < -1.0 && value > -1.0 - EPSILON)
    {
        return -1.0;
    }
    else
    {
        return value;
    }
}

/**
 * @brief Validates the orbital rotation matrix.
 * @details This function checks if the given orbital rotation matrix is valid
 * by comparing it to an identity matrix.
 * @param mat The input orbital rotation matrix.
 * @param rtol The relative tolerance for comparison.
 * @param atol The absolute tolerance for comparison.
 */
void validate_orbital_rotation(const OrbitalRotation& mat, double rtol, double atol)
{
    if (mat.type == OrbitalRotationType::Spinless)
    {
        if (!linalg::is_unitary(mat.spinless, rtol, atol))
        {
            throw std::runtime_error("The input orbital rotation matrix was not unitary.");
        }
    }
    else
    {
        if (const auto& matrix0_opt = mat.spinfull[0]; matrix0_opt.has_value())
        {
            if (!linalg::is_unitary(matrix0_opt.value(), rtol, atol))
            {
                throw std::runtime_error(
                    "The input orbital rotation matrix for spin alpha was not unitary.");
            }
        }
        if (const auto& matrix1_opt = mat.spinfull[1]; matrix1_opt.has_value())
        {
            if (!linalg::is_unitary(matrix1_opt.value(), rtol, atol))
            {
                throw std::runtime_error(
                    "The input orbital rotation matrix for spin beta was not unitary.");
            }
        }
    }
}

} // namespace ffsim
#endif // PARAMETERS_HPP
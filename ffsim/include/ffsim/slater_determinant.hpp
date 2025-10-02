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

#ifndef SLATER_DETERMINANT_HPP
#define SLATER_DETERMINANT_HPP

#include "circuit_instruction.hpp"
#include "gates/orbital_rotation.hpp"
#include "linalg/givens.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <complex>

namespace ffsim
{

using namespace Eigen;
using namespace linalg;
using namespace gates;
using CircuitInstruction = ffsim::CircuitInstruction;

/**
 * @brief Performs Givens decomposition for a Slater determinant.
 *
 * Returns a sequence of Givens rotations that diagonalize the orbital
 * coefficient matrix.
 *
 * @param orbital_coeffs Matrix representing occupied orbitals
 * @return List of Givens rotations that diagonalize the matrix
 */
std::vector<GivensRotation> givens_decomposition_slater(const MatrixXcd &orbital_coeffs)
{
    size_t m = orbital_coeffs.rows();
    size_t n = orbital_coeffs.cols();
    MatrixXcd mat = orbital_coeffs;

    std::vector<GivensRotation> rotations;
    for (int j = static_cast<int>(n - 1); j >= static_cast<int>(n - m + 1); --j) {
        for (int i = 0; i < static_cast<int>(m - n + j); ++i) {
            if (std::norm(mat(i, j)) > 0.0) {
                auto [c, s, _] = zrotg(mat(i + 1, j), mat(i, j));

                VectorXcd row1 = mat.row(i + 1).transpose();
                VectorXcd row2 = mat.row(i).transpose();
                zrot(row1, row2, c, s);
                mat.row(i + 1) = row1.transpose();
                mat.row(i) = row2.transpose();
            }
        }
    }

    for (size_t i = 0; i < m; ++i) {
        for (int j = static_cast<int>(n - m + i); j > static_cast<int>(i); --j) {
            if (std::norm(mat(static_cast<Index>(i), static_cast<Index>(j))) > 0.0) {
                auto [c, s, _] = zrotg(
                    mat(static_cast<Index>(i), static_cast<Index>(j - 1)),
                    mat(static_cast<Index>(i), static_cast<Index>(j))
                );
                rotations.emplace_back(c, s, j, j - 1);
                VectorXcd col1 = mat.col(j - 1).transpose();
                VectorXcd col2 = mat.col(j).transpose();
                zrot(col1, col2, c, s);
                mat.col(j - 1) = col1.transpose();
                mat.col(j) = col2.transpose();
            }
        }
    }

    std::reverse(rotations.begin(), rotations.end());
    return rotations;
}

/**
 * @brief Generates circuit instructions for a general Slater determinant in JW
 * basis.
 *
 * @param qubits Qubit indices to use (should match matrix dimension)
 * @param orbital_coeffs Coefficient matrix describing the occupied orbitals
 * @return List of Jordan-Wigner circuit instructions implementing the state
 * preparation
 */
std::vector<CircuitInstruction> prepare_slater_determinant_jw(
    const std::vector<uint32_t> &qubits, const MatrixXcd &orbital_coeffs
)
{
    std::vector<CircuitInstruction> instructions;
    size_t m = orbital_coeffs.rows();
    size_t n = orbital_coeffs.cols();

    instructions.reserve(m);

    for (size_t i = 0; i < m; ++i) {
        instructions.push_back({"x", {static_cast<unsigned int>(qubits[i])}, {}, {}});
    }

    if (m == n)
        return instructions;

    auto givens_rotation = givens_decomposition_slater(orbital_coeffs);

    for (const auto &rotation : givens_rotation) {
        double c = round_for_acos(rotation.c);
        double theta = 2.0 * std::acos(c);
        double beta = std::arg(rotation.s) - 0.5 * M_PI;
        instructions.push_back(
            {"xx_plus_yy",
             {static_cast<unsigned int>(qubits[rotation.i]),
              static_cast<unsigned int>(qubits[rotation.j])},
             {},
             {theta, beta}}
        );
    }
    return instructions;
}

/**
 * @brief Prepares a Slater determinant in the Jordan-Wigner representation.
 *
 */
class PrepareSlaterDeterminantJW
{
  public:
    /**
     * @brief Constructor.
     * @param norb Number of spatial orbitals
     * @param occupied_orbitals Pair of lists of occupied orbitals for (alpha,
     * beta) electrons
     * @param orbital_rotation Optional orbital rotation
     * @param qubits List of 2*norb qubit indices (alpha followed by beta spin
     * orbitals)
     * @param validate Whether to validate the unitarity of the input orbital
     * rotation
     * @param rtol Relative tolerance for validation
     * @param atol Absolute tolerance for validation
     */
    PrepareSlaterDeterminantJW(
        uint64_t norb,
        const std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
            &occupied_orbitals,
        const std::optional<OrbitalRotation> &orbital_rotation,
        const std::vector<uint32_t> &qubits, bool validate, double rtol, double atol
    )
      : norb(norb), occupied_orbitals(occupied_orbitals), qubits(qubits)
    {
        if (qubits.size() != 2 * norb) {
            throw std::runtime_error("Qubits sizew must be equal to 2 * norb");
        }

        if (orbital_rotation.has_value()) {
            if (validate) {
                validate_orbital_rotation(orbital_rotation.value(), rtol, atol);
            }
            if (orbital_rotation->type == OrbitalRotationType::Spinless) {
                orbital_rotation_a_ = orbital_rotation.value().spinless;
                orbital_rotation_b_ = orbital_rotation.value().spinless;
            } else {
                orbital_rotation_a_ = orbital_rotation.value().spinfull[0].value_or(
                    MatrixXcd::Identity(
                        static_cast<Index>(norb), static_cast<Index>(norb)
                    )
                );
                orbital_rotation_b_ = orbital_rotation.value().spinfull[1].value_or(
                    MatrixXcd::Identity(
                        static_cast<Index>(norb), static_cast<Index>(norb)
                    )
                );
            }
        } else {
            orbital_rotation_a_ =
                MatrixXcd::Identity(static_cast<Index>(norb), static_cast<Index>(norb));
            orbital_rotation_b_ =
                MatrixXcd::Identity(static_cast<Index>(norb), static_cast<Index>(norb));
        }
    }

    /**
     * @brief Returns the circuit instructions that prepare the Slater
     * determinant.
     * @return List of Jordan-Wigner-based circuit instructions
     */
    std::vector<CircuitInstruction> instructions() const
    {
        uint64_t norb = qubits.size() / 2;
        std::vector<uint32_t> alpha_qubits(
            qubits.begin(), qubits.begin() + static_cast<std::ptrdiff_t>(norb)
        );
        std::vector<uint32_t> beta_qubits(
            qubits.begin() + static_cast<std::ptrdiff_t>(norb), qubits.end()
        );
        std::vector<CircuitInstruction> instructions;
        const auto &occ_a = occupied_orbitals.first;
        const auto &occ_b = occupied_orbitals.second;
        if (orbital_rotation_a_.isApprox(
                MatrixXcd::Identity(static_cast<Index>(norb), static_cast<Index>(norb)),
                1e-12
            )) {
            for (auto a : occ_a) {
                instructions.push_back(
                    {"x", {static_cast<unsigned int>(alpha_qubits[a])}, {}, {}}
                );
            }
        } else {
            MatrixXcd orb_a_t = orbital_rotation_a_(all, occ_a).transpose();
            auto slater_instr_a = prepare_slater_determinant_jw(alpha_qubits, orb_a_t);
            instructions.insert(
                instructions.end(), slater_instr_a.begin(), slater_instr_a.end()
            );
        }
        if (orbital_rotation_b_.isApprox(
                MatrixXcd::Identity(static_cast<Index>(norb), static_cast<Index>(norb)),
                1e-12
            )) {
            for (auto b : occ_b) {
                instructions.push_back(
                    {"x", {static_cast<unsigned int>(beta_qubits[b])}, {}, {}}
                );
            }
        } else {
            MatrixXcd orb_b_t = orbital_rotation_b_(all, occ_b).transpose();
            auto slater_instr_b = prepare_slater_determinant_jw(beta_qubits, orb_b_t);
            instructions.insert(
                instructions.end(), slater_instr_b.begin(), slater_instr_b.end()
            );
        }

        return instructions;
    }

    /**
     * @brief Returns a mutable reference to the orbital rotation matrix for alpha
     * spin.
     * @return Mutable reference to the orbital rotation matrix for alpha spin
     */
    MatrixXcd &orbital_rotation_a()
    {
        return orbital_rotation_a_;
    }

    /**
     * @brief Returns a mutable reference to the orbital rotation matrix for beta
     * spin.
     * @return Mutable reference to the orbital rotation matrix for beta spin
     */
    MatrixXcd &orbital_rotation_b()
    {
        return orbital_rotation_b_;
    }

  private:
    uint64_t norb; ///< Number of spatial orbitals
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
        occupied_orbitals;         ///< Occupied orbitals for (alpha, beta) electrons
    MatrixXcd orbital_rotation_a_; ///< Orbital rotation matrix for alpha spin
    MatrixXcd orbital_rotation_b_; ///< Orbital rotation matrix for beta spin
    std::vector<uint32_t> qubits;  ///< List of 2*norb qubit indices (alpha
                                   ///< followed by beta spin orbitals)
};

/**
 * @brief Prepares a Hartree-Fock state in the Jordan-Wigner representation.
 *
 * This class generates quantum circuit instructions to prepare the default
 * Hartree-Fock determinant with specified numbers of alpha and beta electrons.
 */
class PrepareHatreeFockJW
{
  public:
    /**
     * @brief Constructor.
     * @param norb Number of spatial orbitals
     * @param nelec Number of electrons as a pair (alpha, beta)
     */
    PrepareHatreeFockJW(uint64_t norb, std::pair<uint64_t, uint64_t> nelec)
      : norb(norb), nelec(nelec)
    {
    }

    /**
     * @brief Returns the quantum circuit instructions to prepare the Hartree-Fock
     * state.
     * @param qubits List of 2*norb qubit indices (alpha followed by beta spin
     * orbitals)
     * @return List of Jordan-Wigner-based circuit instructions
     */
    std::vector<CircuitInstruction>
    instructions(const std::vector<uint32_t> &qubits) const
    {
        std::vector<uint64_t> n_alpha(nelec.first);
        std::iota(n_alpha.begin(), n_alpha.end(), 0);
        std::vector<uint64_t> n_beta(nelec.second);
        std::iota(n_beta.begin(), n_beta.end(), 0);
        return PrepareSlaterDeterminantJW(
                   norb, {n_alpha, n_beta}, std::nullopt, qubits, true, 1e-5, 1e-8
        )
            .instructions();
    }

  private:
    uint64_t norb;
    std::pair<uint64_t, uint64_t> nelec;
};

} // namespace ffsim

#endif // SLATER_DETERMINANT_HPP
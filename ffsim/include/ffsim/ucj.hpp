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

#ifndef UCJOP_HPP
#define UCJOP_HPP

#include "circuit_instruction.hpp"
#include "diag_coulomb_jw.hpp"
#include "gates/diag_coulomb.hpp"
#include "linalg/expm.hpp"
#include "orbital_rotation_jw.hpp"
#include "slater_determinant.hpp"
#include "ucjop_spinbalanced.hpp"

namespace ffsim
{

/** * @brief Generate the circuit instructions for preparing a Slater determinant
 * in Jordan-Wigner encoding.
 *
 * This function prepares a Slater determinant state on the specified qubits
 * using the provided orbital coefficients.
 *
 * @param qubits The list of qubit indices (should be of length 2 * norb).
 * @param norb The number of orbitals.
 * @param nelec The number of electrons in each spin sector (pair of spin-up and
 * spin-down counts).
 * @param params The parameters for the Slater determinant, typically containing
 * the orbital coefficients.
 * @return A vector of CircuitInstruction objects implementing the Slater
 * determinant state preparation.
 *
 * @throws std::runtime_error if the length of `qubits` is not equal to 2 * norb.
 */
std::vector<CircuitInstruction> slater_determinant_instruction(
    const std::vector<uint32_t> &qubits, const size_t norb,
    const std::pair<uint64_t, uint64_t> &nelec, const VectorXcd &params
)
{
    if (qubits.size() != 2 * norb) {
        throw std::runtime_error("The length of `qubits` must be (norb * 2)");
    }

    std::vector<uint64_t> n_alpha(nelec.first);
    std::iota(n_alpha.begin(), n_alpha.end(), 0);
    std::vector<uint64_t> n_beta(nelec.second);
    std::iota(n_beta.begin(), n_beta.end(), 0);

    size_t nela = nelec.first;

    MatrixXcd Kmat =
        MatrixXcd::Zero(static_cast<Index>(norb), static_cast<Index>(norb));
    int idx = 0;
    for (int i = 0; i < nela; ++i) {
        for (int j = 0; j < (norb - nela); ++j) {
            double real_part = params(idx++).real();
            double imag_part = params(idx++).real(); // assume params are all real
            std::complex<double> val(real_part, imag_part);
            Kmat(static_cast<Index>(i), static_cast<Index>(j + nela)) = val;
        }
    }

    Kmat = Kmat - Kmat.adjoint().eval();
    MatrixXcd orbital_rotation = Kmat.exp();

    PrepareSlaterDeterminantJW slater_determinant(
        norb, {n_alpha, n_beta},
        OrbitalRotation{
            OrbitalRotationType::Spinless, orbital_rotation,
            std::array<std::optional<MatrixXcd>, 2>{std::nullopt, std::nullopt}
        },
        qubits, true, 1e-5, 1e-8
    );

    std::vector<CircuitInstruction> slater_inst = slater_determinant.instructions();

    return slater_inst;
}

/**
 * @brief Generate the circuit instructions for a UCJOpSpinBalanced operator in
 * Jordan–Wigner encoding.
 *
 * This function is intended for use outside.
 *
 * @param qubits The list of qubit indices.
 * @param ucj_op The UCJOpSpinBalanced operator.
 * @return A vector of CircuitInstruction objects implementing the UCJ operator.
 */
std::vector<CircuitInstruction> ucj_op_spin_balanced_jw(
    const std::vector<uint32_t> &qubits, const UCJOpSpinBalanced &ucj_op
)
{
    std::vector<CircuitInstruction> instructions;
    uint64_t norb = ucj_op.norb();
    size_t n_reps = ucj_op.n_reps();

    for (int rep = 0; rep < n_reps; ++rep) {
        MatrixXcd diag_coulomb_mat_aa(norb, norb);
        MatrixXcd diag_coulomb_mat_ab(norb, norb);
        for (int i = 0; i < norb; ++i) {
            for (int j = 0; j < norb; ++j) {
                diag_coulomb_mat_aa(i, j) = ucj_op.diag_coulomb_mats(rep, 0, i, j);
                diag_coulomb_mat_ab(i, j) = ucj_op.diag_coulomb_mats(rep, 1, i, j);
            }
        }

        MatrixXcd orbital_rotation(norb, norb);
        for (int i = 0; i < norb; ++i) {
            for (int j = 0; j < norb; ++j) {
                orbital_rotation(i, j) = std::conj(ucj_op.orbital_rotations(rep, j, i));
            }
        }

        auto rot1 = OrbitalRotationJW(
            norb,
            OrbitalRotation{
                OrbitalRotationType::Spinless, orbital_rotation,
                std::array<std::optional<MatrixXcd>, 2>{std::nullopt, std::nullopt}
            },
            true, 1e-5, 1e-8
        );
        auto inst1 = rot1.instructions(qubits);
        instructions.insert(instructions.end(), inst1.begin(), inst1.end());

        auto diag = DiagCoulombEvolutionJW(
            norb,
            Mat{MatType::Triple,
                MatrixXcd(),
                {diag_coulomb_mat_aa, diag_coulomb_mat_ab, diag_coulomb_mat_aa}},
            -1.0, false
        );
        auto inst2 = diag.instructions(qubits);
        instructions.insert(instructions.end(), inst2.begin(), inst2.end());

        MatrixXcd orbital_rotation2(norb, norb);
        for (int i = 0; i < norb; ++i) {
            for (int j = 0; j < norb; ++j) {
                orbital_rotation2(i, j) = ucj_op.orbital_rotations(rep, i, j);
            }
        }
        auto rot2 = OrbitalRotationJW(
            norb,
            OrbitalRotation{
                OrbitalRotationType::Spinless, orbital_rotation2,
                std::array<std::optional<MatrixXcd>, 2>{std::nullopt, std::nullopt}
            },
            true, 1e-5, 1e-8
        );
        auto inst3 = rot2.instructions(qubits);
        instructions.insert(instructions.end(), inst3.begin(), inst3.end());
    }

    if (ucj_op.final_orbital_rotation.has_value()) {
        auto final_orbital_rotation = OrbitalRotationJW(
            norb,
            OrbitalRotation{
                OrbitalRotationType::Spinless, ucj_op.final_orbital_rotation.value(),
                std::array<std::optional<MatrixXcd>, 2>{std::nullopt, std::nullopt}
            },
            true, 1e-5, 1e-8
        );
        auto inst4 = final_orbital_rotation.instructions(qubits);
        instructions.insert(instructions.end(), inst4.begin(), inst4.end());
    }

    return instructions;
}

/**
 * @brief A class to convert a UCJOpSpinBalanced operator into Jordan-Wigner
 * basis quantum circuit instructions.
 *
 * This class provides a convenient interface for generating circuit
 * instructions that implement the spin-balanced unitary cluster Jastrow (UCJ)
 * operator on a given set of qubits using the Jordan-Wigner encoding.
 */
class UCJOpSpinBalancedJW
{
  public:
    /**
     * @brief Construct a UCJOpSpinBalancedJW object.
     *
     * @param ucj_op_ The UCJOpSpinBalanced operator to be applied.
     */
    explicit UCJOpSpinBalancedJW(const UCJOpSpinBalanced &ucj_op_) : ucj_op(ucj_op_)
    {
    }

    /**
     * @brief Generate the quantum circuit instructions that implement the UCJ
     * operator.
     *
     * @param qubits The list of qubit indices on which to apply the operator.
     * @return A vector of CircuitInstruction objects representing the quantum
     * gates.
     *
     * @throws std::runtime_error if the number of qubits is incorrect.
     */
    std::vector<CircuitInstruction>
    instructions(const std::vector<uint32_t> &qubits) const
    {
        size_t required_length = ucj_op.norb() * 2;
        if (qubits.size() != required_length) {
            throw std::runtime_error("The length of `qubits` must be (norb * 2)");
        }
        return ucj_op_spin_balanced_jw(qubits, ucj_op);
    }

  private:
    UCJOpSpinBalanced ucj_op;
};

/**
 * @brief Generate the circuit instructions for preparing a Hartree-Fock state
 * and applying a UCJOpSpinBalanced operator in Jordan–Wigner encoding.
 *
 * This function combines Hartree-Fock state preparation and the application
 * of the spin-balanced UCJ operator on the specified qubits.
 *
 * @param qubits The list of qubit indices.
 * @param nelec The number of electrons in each spin sector (pair of spin-up and
 * spin-down counts).
 * @param ucj_op The UCJOpSpinBalanced operator.
 * @return A vector of CircuitInstruction objects implementing the Hartree-Fock
 * state preparation and the UCJ operator.
 */
std::vector<CircuitInstruction> hf_and_ucj_op_spin_balanced_jw(
    const std::vector<uint32_t> &qubits, const std::pair<uint64_t, uint64_t> &nelec,
    const UCJOpSpinBalanced &ucj_op
)
{
    std::vector<CircuitInstruction> instructions;
    uint64_t norb = ucj_op.norb();

    if (qubits.size() != 2 * norb) {
        throw std::runtime_error("The length of `qubits` must be (norb * 2)");
    }

    std::vector<uint64_t> n_alpha(nelec.first);
    std::iota(n_alpha.begin(), n_alpha.end(), 0);
    std::vector<uint64_t> n_beta(nelec.second);
    std::iota(n_beta.begin(), n_beta.end(), 0);

    const size_t n_reps = ucj_op.n_reps();

    PrepareSlaterDeterminantJW slater_determinant(
        norb, {n_alpha, n_beta}, std::nullopt, qubits, true, 1e-5, 1e-8
    );

    MatrixXcd orbital_rotation = MatrixXcd(norb, norb);
    for (uint64_t i = 0; i < norb; ++i) {
        for (uint64_t j = 0; j < norb; ++j) {
            orbital_rotation(static_cast<Index>(i), static_cast<Index>(j)) = std::conj(
                ucj_op.orbital_rotations(0, static_cast<long>(j), static_cast<long>(i))
            );
        }
    }

    MatrixXcd orbital_rotation_a = orbital_rotation;
    MatrixXcd orbital_rotation_b = orbital_rotation;

    MatrixXcd combined_mat_a =
        orbital_rotation_a * slater_determinant.orbital_rotation_a();
    MatrixXcd combined_mat_b =
        orbital_rotation_b * slater_determinant.orbital_rotation_b();

    slater_determinant.orbital_rotation_a() = combined_mat_a;
    slater_determinant.orbital_rotation_b() = combined_mat_b;

    std::vector<CircuitInstruction> slater_inst = slater_determinant.instructions();
    instructions.insert(instructions.end(), slater_inst.begin(), slater_inst.end());

    for (int rep = 0; rep < n_reps; ++rep) {
        MatrixXcd diag_coulomb_mat_aa(norb, norb);
        MatrixXcd diag_coulomb_mat_ab(norb, norb);
        for (int i = 0; i < norb; ++i) {
            for (int j = 0; j < norb; ++j) {
                diag_coulomb_mat_aa(i, j) = ucj_op.diag_coulomb_mats(rep, 0, i, j);
                diag_coulomb_mat_ab(i, j) = ucj_op.diag_coulomb_mats(rep, 1, i, j);
            }
        }

        auto diag = DiagCoulombEvolutionJW(
            norb,
            Mat{MatType::Triple,
                MatrixXcd(),
                {diag_coulomb_mat_aa, diag_coulomb_mat_ab, diag_coulomb_mat_aa}},
            -1.0, false
        );
        auto inst2 = diag.instructions(qubits);
        instructions.insert(instructions.end(), inst2.begin(), inst2.end());

        if (rep == n_reps - 1) {
            MatrixXcd orbital_rotation2(norb, norb);
            for (int i = 0; i < norb; ++i) {
                for (int j = 0; j < norb; ++j) {
                    orbital_rotation2(i, j) = ucj_op.orbital_rotations(rep, i, j);
                }
            }
            if (ucj_op.final_orbital_rotation.has_value()) {
                auto rot2 = OrbitalRotationJW(
                    norb,
                    OrbitalRotation{
                        OrbitalRotationType::Spinless,
                        ucj_op.final_orbital_rotation.value() * orbital_rotation2,
                        std::array<std::optional<MatrixXcd>, 2>{
                            std::nullopt, std::nullopt
                        }
                    },
                    true, 1e-5, 1e-8
                );
                auto inst3 = rot2.instructions(qubits);
                instructions.insert(instructions.end(), inst3.begin(), inst3.end());
            } else {
                auto rot2 = OrbitalRotationJW(
                    norb,
                    OrbitalRotation{
                        OrbitalRotationType::Spinless, orbital_rotation2,
                        std::array<std::optional<MatrixXcd>, 2>{
                            std::nullopt, std::nullopt
                        }
                    },
                    true, 1e-5, 1e-8
                );
                auto inst3 = rot2.instructions(qubits);
                instructions.insert(instructions.end(), inst3.begin(), inst3.end());
            }
        }
    }

    return instructions;
}

} // namespace ffsim

#endif // UCJOP_HPP
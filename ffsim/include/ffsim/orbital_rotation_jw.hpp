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

#ifndef ORBITAL_ROTATION_JW_HPP
#define ORBITAL_ROTATION_JW_HPP

#include "circuit_instruction.hpp"
#include "ffsim/linalg/givens.hpp"
#include "ffsim/utils.hpp"
#include "gates/orbital_rotation.hpp"
#include <Eigen/Dense>

namespace ffsim
{

using namespace Eigen;
using namespace gates;

/**
 * @brief Applies an orbital rotation to a list of qubits.
 *
 * This is a helper function for performing an orbital rotation.
 *
 * @param qubits Qubit indices representing the orbital register
 * @param orbital_rotation Orbital rotation matrix
 * @return Vector of `CircuitInstruction` objects implementing the rotation
 */
std::vector<CircuitInstruction> orbital_rotation_jw(
    const std::vector<uint32_t> &qubits, const MatrixXcd &orbital_rotation
)
{
    auto [givens_rotations, phase_shifts] =
        linalg::givens_decomposition(orbital_rotation);
    std::vector<CircuitInstruction> instructions;

    for (const auto &rotation : givens_rotations) {
        double c = round_for_acos(rotation.c);
        double theta = 2.0 * std::acos(c);
        double beta = std::arg(rotation.s) - 0.5 * M_PI;
        instructions.push_back(
            {"xx_plus_yy",
             {static_cast<unsigned int>(qubits[rotation.i]),
              static_cast<unsigned int>(qubits[rotation.j])},
             {},
             std::vector<double>{theta, beta}}
        );
    }

    for (size_t i = 0; i < phase_shifts.size(); ++i) {
        double theta = std::arg(phase_shifts(static_cast<Index>(i)));
        instructions.push_back(
            {"rz",
             {static_cast<unsigned int>(qubits[i])},
             {},
             std::vector<double>{theta}}
        );
    }

    return instructions;
}

/**
 * @brief Orbital rotation operations in the Jordan-Wigner representation.
 *
 * The rotation is applied separately to alpha and beta spins if the input
 * `OrbitalRotation` is of type `Spinfull`, or to both spins if `Spinless`.
 */
class OrbitalRotationJW
{
  public:
    /**
     * @brief Constructs an orbital rotation operator for Jordan-Wigner basis.
     *
     * @param norb Number of spatial orbitals
     * @param orbital_rotation Orbital rotation matrix (spinless or spinfull)
     * @param validate Whether to validate unitarity of the rotation matrices
     * @param rtol Relative tolerance for validation
     * @param atol Absolute tolerance for validation
     *
     * @throws std::runtime_error if validation fails
     */
    OrbitalRotationJW(
        uint64_t norb, const OrbitalRotation &orbital_rotation, bool validate = true,
        double rtol = 1e-5, double atol = 1e-8
    )
      : norb(norb)
    {
        if (validate) {
            validate_orbital_rotation(orbital_rotation, rtol, atol);
        }
        if (orbital_rotation.type == OrbitalRotationType::Spinless) {
            orbital_rotation_a = orbital_rotation.spinless;
            orbital_rotation_b = orbital_rotation.spinless;
        } else {
            orbital_rotation_a = orbital_rotation.spinfull[0].value_or(
                MatrixXcd::Identity(static_cast<Index>(norb), static_cast<Index>(norb))
            );
            orbital_rotation_b = orbital_rotation.spinfull[1].value_or(
                MatrixXcd::Identity(static_cast<Index>(norb), static_cast<Index>(norb))
            );
        }
    }

    /**
     * @brief Generates quantum circuit instructions for the orbital rotation.
     *
     * @param qubits List of 2*norb qubit indices (alpha followed by beta spin
     * orbitals)
     * @return Vector of `CircuitInstruction` objects encoding the gate sequence
     */
    std::vector<CircuitInstruction>
    instructions(const std::vector<uint32_t> &qubits) const
    {
        auto norb_tmp = static_cast<std::ptrdiff_t>(qubits.size() / 2);
        std::vector<uint32_t> alpha_qubits(qubits.begin(), qubits.begin() + norb_tmp);
        std::vector<uint32_t> beta_qubits(qubits.begin() + norb_tmp, qubits.end());

        auto instructions = orbital_rotation_jw(alpha_qubits, orbital_rotation_a);
        auto beta_instructions = orbital_rotation_jw(beta_qubits, orbital_rotation_b);
        instructions.insert(
            instructions.end(), beta_instructions.begin(), beta_instructions.end()
        );
        return instructions;
    }

  private:
    uint64_t norb;                ///< Number of orbitals
    MatrixXcd orbital_rotation_a; ///< Orbital rotation matrix for alpha spin
    MatrixXcd orbital_rotation_b; ///< Orbital rotation matrix for beta spin
};

} // namespace ffsim

#endif // ORBITAL_ROTATION_JW_HPP
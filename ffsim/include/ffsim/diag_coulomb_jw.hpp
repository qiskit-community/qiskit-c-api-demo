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

#ifndef DIAG_COULOMB_JW_HPP
#define DIAG_COULOMB_JW_HPP

#include "circuit_instruction.hpp"
#include "gates/diag_coulomb.hpp"
#include <optional>
#include <vector>

namespace ffsim
{

using namespace gates;

/**
 * @brief Jordan-Wigner basis representation of diagonal Coulomb evolution
 * operator.
 *
 * This class builds a quantum circuit that implements the evolution
 * under a diagonal Coulomb Hamiltonian term using the Jordan-Wigner (JW)
 * mapping.
 */
class DiagCoulombEvolutionJW
{
  public:
    uint64_t norb;         ///< Number of orbitals
    Mat mat;               ///< Representation of the Coulomb matrix (single or triple form)
    double time;           ///< Time evolution parameter
    bool z_representation; ///< Whether to use the Z basis (Pauli-Z representation)

    /**
     * @brief Constructs the JW evolution operator for a diagonal Coulomb term.
     *
     * @param norb Number of spatial orbitals
     * @param mat Coulomb operator representation (single or spin-resolved triple)
     * @param time Time evolution parameter
     * @param z_representation Whether to use Z-basis representation
     */
    DiagCoulombEvolutionJW(uint64_t norb, const Mat& mat, double time, bool z_representation)
        : norb(norb), mat(mat), time(time), z_representation(z_representation)
    {
    }

    /**
     * @brief Returns quantum circuit instructions that implements the evolution
     * operator.
     *
     * The method generates a vector of `CircuitInstruction`s, which encode
     * the quantum gates to implement the evolution under the diagonal Coulomb
     * Hamiltonian in the Jordan-Wigner basis.
     *
     * @param qubits A vector of 2*norb qubit indices (alpha followed by beta spin
     * orbitals)
     * @return Vector of circuit instructions
     */
    std::vector<CircuitInstruction> instructions(const std::vector<uint32_t>& qubits) const
    {
        if (z_representation)
        {
            return diag_coulomb_evolution_z_rep_jw(qubits);
        }
        else
        {
            return diag_coulomb_evolution_num_rep_jw(qubits);
        }
    }

  private:
    /**
     * @brief Implements the number-representation based evolution (number
     * operators).
     *
     * @param qubits Input qubit indices
     * @return Quantum circuit as vector of circuit instructions
     */
    std::vector<CircuitInstruction>
    diag_coulomb_evolution_num_rep_jw(const std::vector<uint32_t>& qubits) const
    {
        std::vector<CircuitInstruction> instructions;

        auto [mat_aa, mat_ab, mat_bb] = [&]()
        {
            if (mat.type == MatType::Single)
            {
                return std::make_tuple(mat.single, mat.single, mat.single);
            }
            else
            {
                return std::make_tuple(mat.triple[0].value_or(MatrixXcd::Zero(norb, norb)),
                                       mat.triple[1].value_or(MatrixXcd::Zero(norb, norb)),
                                       mat.triple[2].value_or(MatrixXcd::Zero(norb, norb)));
            }
        }();

        for (int sigma = 0; sigma < 2; ++sigma)
        {
            const auto& this_mat = (sigma == 0) ? mat_aa : mat_bb;
            for (size_t i = 0; i < norb; ++i)
            {
                if (std::abs(this_mat(i, i).real()) > 1e-12 ||
                    std::abs(this_mat(i, i).imag()) > 1e-12)
                {
                    instructions.push_back(
                        {"rz",
                         {static_cast<unsigned int>(qubits[i + sigma * norb])},
                         {},
                         std::vector<double>{-0.5 * this_mat(i, i).real() * time}});
                }
                for (size_t j = i + 1; j < norb; ++j)
                {
                    if (std::abs(this_mat(i, j).real()) > 1e-12 ||
                        std::abs(this_mat(i, j).imag()) > 1e-12)
                    {
                        instructions.push_back(
                            {"cp",
                             {static_cast<unsigned int>(qubits[i + sigma * norb]),
                              static_cast<unsigned int>(qubits[j + sigma * norb])},
                             {},
                             std::vector<double>{-1.0 * this_mat(i, j).real() * time}});
                    }
                }
            }
        }

        for (size_t i = 0; i < norb; ++i)
        {
            if (std::abs(mat_ab(i, i).real()) > 1e-12 || std::abs(mat_ab(i, i).imag()) > 1e-12)
            {
                instructions.push_back({"cp",
                                        {static_cast<unsigned int>(qubits[i]),
                                         static_cast<unsigned int>(qubits[i + norb])},
                                        {},
                                        std::vector<double>{-1.0 * mat_ab(i, i).real() * time}});
            }
            for (size_t j = i + 1; j < norb; ++j)
            {
                if (std::abs(mat_ab(i, j).real()) > 1e-12 || std::abs(mat_ab(i, j).imag()) > 1e-12)
                {
                    instructions.push_back(
                        {"cp",
                         {static_cast<unsigned int>(qubits[i]),
                          static_cast<unsigned int>(qubits[j + norb])},
                         {},
                         std::vector<double>{-1.0 * mat_ab(i, j).real() * time}});
                }
                if (std::abs(mat_ab(j, i).real()) > 1e-12 || std::abs(mat_ab(j, i).imag()) > 1e-12)
                {
                    instructions.push_back(
                        {"cp",
                         {static_cast<unsigned int>(qubits[j]),
                          static_cast<unsigned int>(qubits[i + norb])},
                         {},
                         std::vector<double>{-1.0 * mat_ab(j, i).real() * time}});
                }
            }
        }
        return instructions;
    }
    /**
     * @brief Implements the Z-representation based evolution (Pauli-Z strings).
     *
     * @param qubits Input qubit indices
     * @return Quantum circuit as vector of circuit instructions
     */
    std::vector<CircuitInstruction>
    diag_coulomb_evolution_z_rep_jw(const std::vector<uint32_t>& qubits) const
    {
        auto [mat_aa, mat_ab, mat_bb] = [&]()
        {
            if (mat.type == MatType::Single)
            {
                return std::make_tuple(mat.single, mat.single, mat.single);
            }
            else
            {
                return std::make_tuple(mat.triple[0].value_or(MatrixXcd::Zero(norb, norb)),
                                       mat.triple[1].value_or(MatrixXcd::Zero(norb, norb)),
                                       mat.triple[2].value_or(MatrixXcd::Zero(norb, norb)));
            }
        }();

        std::vector<CircuitInstruction> instructions;

        for (size_t i = 0; i < 2 * norb; ++i)
        {
            for (size_t j = i + 1; j < 2 * norb; ++j)
            {
                const MatrixXcd this_mat = (i < norb && j < norb)     ? mat_aa
                                           : (i >= norb && j >= norb) ? mat_bb
                                                                      : mat_ab;
                const auto val = this_mat(i % norb, j % norb);
                if (std::abs(val.real()) > 1e-12 || std::abs(val.imag()) > 1e-12)
                {
                    instructions.push_back({"rzz",
                                            {static_cast<unsigned int>(qubits[i]),
                                             static_cast<unsigned int>(qubits[j])},
                                            {},
                                            std::vector<double>{-1.0 * val.real() * time}});
                }
            }
        }
        return instructions;
    }
};

} // namespace ffsim

#endif // DIAG_COULOMB_JW_HPP
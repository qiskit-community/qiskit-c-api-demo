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

#ifndef CIRCUIT_INSTRUCTION_HPP
#define CIRCUIT_INSTRUCTION_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ffsim
{

/**
 * @brief Represents a quantum gate operation in a QuantumCircuit in Qiskit.
 * @details This structure encapsulates the gate type, the qubits it acts on,
 *          optional classical bits for measurement, and any parameters
 * associated with the gate (e.g., angles for rotation gates).
 * @param gate The quantum gate being represented.
 * @param qubits The qubits that the gate operates on.
 * @param clbits Optional classical bits for measurement.
 * @param params Parameters associated with the gate.
 * @return A CircuitInstruction object representing the gate operation.
 */
struct CircuitInstruction {
    std::string gate;
    std::vector<uint64_t> qubits;
    std::optional<std::vector<uint64_t>> clbits;
    std::vector<double> params;
};

} // namespace ffsim

#endif // CIRCUIT_INSTRUCTION_HPP

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

#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <unordered_map>

#include "boost/dynamic_bitset.hpp"
#include "ffsim/ucj.hpp"
#include "ffsim/ucjop_spinbalanced.hpp"
#include "load_parameters.hpp"
#include "qiskit/addon/sqd/configuration_recovery.hpp"
#include "qiskit/addon/sqd/postselection.hpp"
#include "qiskit/addon/sqd/subsampling.hpp"
#include "sbd_helper.hpp"
#include "sqd_helper.hpp"

#include "circuit/quantumcircuit.hpp"
#include "compiler/transpiler.hpp"
#include "primitives/backend_sampler_v2.hpp"
#include "service/qiskit_runtime_service.hpp"

using namespace Qiskit;
using namespace Qiskit::circuit;
using namespace Qiskit::providers;
using namespace Qiskit::primitives;
using namespace Qiskit::service;
using namespace Qiskit::compiler;

using Sampler = BackendSamplerV2;

// Test stub: generate num_samples random bitstrings of length num_bits
// with Bernoulli(p=0.5) and aggregate into counts (bitstring -> occurrences).
// Use this when a real backend/simulator is unavailable (debugging).
std::unordered_map<std::string, uint64_t> generate_counts_uniform(
    int num_samples, // NOLINT(bugprone-easily-swappable-parameters)
    int num_bits,    // NOLINT(bugprone-easily-swappable-parameters)
    std::optional<unsigned int> seed = std::nullopt
)
{
    std::mt19937 rng(seed.value_or(std::random_device{}()));
    std::bernoulli_distribution dist(0.5);

    std::unordered_map<std::string, uint64_t> counts;

    for (int i = 0; i < num_samples; ++i) {
        std::string bitstring;
        bitstring.reserve(num_bits);
        for (int j = 0; j < num_bits; ++j) {
            bitstring += dist(rng) ? '1' : '0';
        }
        counts[bitstring]++;
    }
    return counts;
}

// Convert an array of boost::dynamic_bitset<> to string-based BitString objects.
static auto bitsets_to_bitstrings(const std::vector<boost::dynamic_bitset<>> &bitsets)
    -> std::vector<BitString>
{
    std::vector<BitString> bitstrings;
    bitstrings.reserve(bitsets.size());
    std::string str;
    for (const auto &bitset : bitsets) {
        boost::to_string(bitset, str);
        bitstrings.emplace_back(str);
    }
    return bitstrings;
}
// Convert string-based BitString objects back to boost::dynamic_bitset<>.
// Internal representation for efficient bitwise operations in recovery/post-selection.
static auto bitsets_from_bitstrings(const std::vector<BitString> &bitstrings)
    -> std::vector<boost::dynamic_bitset<>>
{
    std::vector<boost::dynamic_bitset<>> bitsets;
    bitsets.reserve(bitsets.size());
    for (const auto &bitstring : bitstrings) {
        bitsets.emplace_back(bitstring.to_string());
    }
    return bitsets;
}

// Load initial alpha/beta occupancies from a JSON file.
// Format: { "init_occupancies": [ alpha..., beta... ] }  (even length)
// After splitting, reverse each to match the internal right-to-left convention.
std::array<std::vector<double>, 2> load_initial_occupancies(const std::string &filename)
{
    std::ifstream i(filename);
    nlohmann::json input;
    i >> input;

    // Validate input JSON: throw on missing key to fail fast on user error.
    if (!input.contains("init_occupancies"))
        throw std::invalid_argument(
            "no init_params in initial parameter json: file=" + filename
        );
    std::vector<double> init_occupancy = input["init_occupancies"];

    if ((init_occupancy.size() & 1) != 0) {
        throw std::runtime_error(
            "Initial occupancies list must have even number of elements"
        );
    }
    const auto half_size = init_occupancy.size() / 2;

    std::vector<double> alpha_occupancy(
        init_occupancy.begin(),
        init_occupancy.begin() + static_cast<std::ptrdiff_t>(half_size)
    );
    std::vector<double> beta_occupancy(
        init_occupancy.begin() + static_cast<std::ptrdiff_t>(half_size),
        init_occupancy.end()
    );

    std::reverse(alpha_occupancy.begin(), alpha_occupancy.end());
    std::reverse(beta_occupancy.begin(), beta_occupancy.end());

    return {alpha_occupancy, beta_occupancy};
}

// Utility: normalize counts (occurrences) into probabilities in [0,1].
// Empty input returns an empty map (no exception).
std::unordered_map<std::string, double>
normalize_counts_dict(const std::unordered_map<std::string, uint64_t> &counts)
{
    // Check if the input map is empty
    if (counts.empty()) {
        return {}; // Return an empty map
    }

    // Calculate the total counts
    uint64_t total_counts = std::accumulate(
        counts.begin(), counts.end(), 0, [](uint64_t sum, const auto &pair) {
            return sum + pair.second;
        }
    );

    // Create a new map with normalized values
    std::unordered_map<std::string, double> probabilities;
    for (const auto &[key, value] : counts) {
        probabilities[key] =
            static_cast<double>(value) / static_cast<double>(total_counts);
    }

    return probabilities;
}

// Transform counts (bitstring -> count) into parallel arrays (bitstrings,
// probabilities).
std::pair<std::vector<BitString>, std::vector<double>>
counts_to_arrays(const std::unordered_map<std::string, uint64_t> &counts)
{
    std::vector<BitString> bs_mat;
    std::vector<double> freq_arr;

    if (counts.empty())
        return {bs_mat, freq_arr};

    // Normalize the counts to probabilities
    auto prob_dict = normalize_counts_dict(counts);

    // Convert bitstrings to a 2D boolean matrix
    for (const auto &[bitstring, _] : prob_dict) {
        bs_mat.push_back(BitString(bitstring));
    }

    // Convert probabilities to a 1D array
    for (const auto &[_, probability] : prob_dict) {
        freq_arr.push_back(probability);
    }

    return {bs_mat, freq_arr};
}

using namespace Eigen;
using namespace ffsim;

int main(int argc, char *argv[])
{
    try {
        // ===== MPI initialization =====
        // This workflow assumes MPI. Request FUNNELED (only main thread calls MPI).
        int provided;
        int mpi_init_error =
            MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (mpi_init_error != MPI_SUCCESS) {
            char err_msg[1024];
            int err_msg_len;
            MPI_Error_string(mpi_init_error, err_msg, &err_msg_len);
            // On failure, print a readable error and exit immediately (cluster-friendly
            // diagnostics).
            std::cerr << "MPI_Init failed: " << err_msg << std::endl;
            return mpi_init_error;
        }

        // ===== SBD (diagonalization sub-workflow) configuration =====
        // Build SBD parameters from CLI args. Used by sbd_main for energy evaluation,
        // etc.
        SBD diag_data = generate_sbd_data(argc, argv);

        // ===== SQD (sampling/recovery sub-workflow) configuration =====
        // Holds run_id, backend, shot count, and other metadata for sampling and
        // recovery.
        SQD sqd_data = generate_sqd_data(argc, argv);
        sqd_data.comm = MPI_COMM_WORLD;
        MPI_Comm_rank(sqd_data.comm, &sqd_data.mpi_rank);
        MPI_Comm_size(sqd_data.comm, &sqd_data.mpi_size);
        int message_size = static_cast<int>(sqd_data.run_id.size());
        MPI_Bcast(&message_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (sqd_data.mpi_rank != 0) {
            sqd_data.run_id.resize(message_size);
        }
        // Broadcast the textual run_id to all ranks to keep logs/artifacts consistent.
        // Rank 0 sends the size, others resize buffer, then receive content.
        MPI_Bcast(sqd_data.run_id.data(), message_size, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Random generators:
        //  - rng    : used for sampling/subsampling (fixed seed for reproducibility).
        //  - rc_rng : used for configuration recovery randomness (derived from rng).
        std::mt19937 rng(1234);
        std::mt19937 rc_rng(rng());
        // Batch sizing for SBD input (alpha-determinant groups).
        uint64_t samples_per_batch = sqd_data.samples_per_batch;

        // Read initial parameters (norb, nelec, params for lucj) from JSON.
        const std::string input_file_path = "../data/parameters_fe4s4.json";
        double tol = 1e-8;
        uint64_t norb;
        size_t n_reps = 1;
        std::pair<uint64_t, uint64_t> nelec;
        std::vector<std::pair<uint64_t, uint64_t>> interaction_aa;
        std::vector<std::pair<uint64_t, uint64_t>> interaction_ab;
        std::vector<double> init_params;

        // Centralize I/O on rank 0. Abort the whole job on input failure.
        if (sqd_data.mpi_rank == 0) {
            try {
                load_initial_parameters(
                    input_file_path, norb, nelec, interaction_aa, interaction_ab,
                    init_params
                );
            } catch (const std::exception &e) {
                std::cerr << "Error loading initial parameters: " << e.what()
                          << std::endl;
                MPI_Abort(sqd_data.comm, 1);
                return 1;
            }
            log(sqd_data, {"initial parameters are loaded. param_length=",
                           std::to_string(init_params.size())});
        }

        // Measurement results: (bitstring -> counts). Produced on rank 0, then
        // array-ified later.
        std::unordered_map<std::string, uint64_t> counts;

        auto num_elec_a = nelec.first;
        auto num_elec_b = nelec.second;
        if (sqd_data.mpi_rank == 0) {
// ===== Sampling mode switch =====
// a) Mock: generate_counts_uniform (debugging)
// b) Real: build LUCJ circuit -> transpile -> run on backend with Sampler -> get counts
#if USE_RANDOM_SHOTS != 0
            counts = generate_counts_uniform(sqd_data.num_shots, 2 * norb, 1234);
#else
            //////////////// LUCJ Circuit Generation ////////////////
            size_t params_size = init_params.size();

            Eigen::VectorXcd params(params_size);
            for (size_t i = 0; i < params_size; ++i) {
                params(static_cast<Eigen::Index>(i)) = init_params[i];
            }
            std::optional<MatrixXcd> t1 = std::nullopt;
            // 'interaction_pairs' allows passing (alpha-alpha, alpha-beta/beta-beta)
            // coupling patterns.
            std::array<std::optional<std::vector<std::pair<uint64_t, uint64_t>>>, 2>
                interaction_pairs = {
                    std::make_optional<std::vector<std::pair<uint64_t, uint64_t>>>(
                        interaction_aa
                    ),
                    std::make_optional<std::vector<std::pair<uint64_t, uint64_t>>>(
                        interaction_ab
                    )
                };

            // Construct the spin-balanced UCJ operator from parameter vector.

            UCJOpSpinBalanced ucj_op = UCJOpSpinBalanced::from_parameters(
                params, norb, n_reps, interaction_pairs, true
            );
            std::vector<uint32_t> qubits(2 * norb);
            std::iota(qubits.begin(), qubits.end(), 0);
            auto instructions = hf_and_ucj_op_spin_balanced_jw(qubits, nelec, ucj_op);

            // Quantum circuit with Qiskit C++
            // quantum registers
            auto qr = QuantumRegister(2 * norb);
            // classical registers
            auto cr = ClassicalRegister(2 * norb, "meas");
            // classical registers for test
            auto cr_test = ClassicalRegister(2 * norb, "test");
            std::vector<ClassicalRegister> cregs({cr});
            if (sqd_data.use_reset_mitigation) {
                cregs.push_back(cr_test);
            }
            // create a quantum circuits with registers
            auto circ = QuantumCircuit(std::vector<QuantumRegister>({qr}), cregs);

            if (sqd_data.use_reset_mitigation) {
                // add measure to all qubits and store to test bits
                circ.measure(qr, cr_test);
            }

            // add gates from instruction list from hf_and_ucj_op_spin_balanced_jw
            //   for demo: calling Qiskit C++ circuit functions to make quantum circuit
            for (const auto &instr : instructions) {
                if (std::string("x") == instr.gate) {
                    // X gate
                    circ.x(instr.qubits[0]);
                } else if (std::string("rz") == instr.gate) {
                    // RZ gate
                    circ.rz(instr.params[0], instr.qubits[0]);
                } else if (std::string("cp") == instr.gate) {
                    // controlled phase gate
                    circ.cp(instr.params[0], instr.qubits[0], instr.qubits[1]);
                } else if (std::string("xx_plus_yy") == instr.gate) {
                    // XX_plus_YY gate
                    circ.xx_plus_yy(
                        instr.params[0], instr.params[1], instr.qubits[0],
                        instr.qubits[1]
                    );
                }
            }
            // this is smarter way using standard gate mapping to convert gate name to
            // op auto map = get_standard_gate_name_mapping(); for (const auto &instr :
            // instructions) {
            //    auto op = map[instr.gate];
            //    if (instr.params.size() > 0)
            //         op.set_params(instr.params);
            //    circ.append(op, instr.qubits);
            // }

            // sampling all the qubits
            for (size_t i = 0; i < circ.num_qubits(); ++i) {
                circ.measure(i, i);
            }

            // get backend from Quantum Runtime Service
            // set 2 environment variables before executing
            // QISKIT_IBM_TOKEN = "your API key"
            // QISKIT_IBM_INSTANCE = "your CRN"
            std::string backend_name = sqd_data.backend_name;
            auto service = QiskitRuntimeService();
            auto backend = service.backend(backend_name);

            // Transpile a quantum circuit for the target backend.
            auto transpiled = transpile(circ, backend);

            uint64_t num_shots = sqd_data.num_shots;

            // Configure the Sampler execution (num_shots from SQD configuration).
            auto sampler = Sampler(backend, num_shots);

            auto job = sampler.run({SamplerPub(transpiled)});
            if (job == nullptr)
                return -1;
            auto result = job->result();
            auto pub_result = result[0];

            // Extract classical counts from the execution result.
            // These form the classical distribution for downstream recovery/selection.
            if (sqd_data.use_reset_mitigation) {
                auto meas_bits = pub_result.data("meas");
                auto test_bits = pub_result.data("test");

                // get index whose bitcout is zero
                auto bitcounts = test_bits.bitcount();
                reg_t zero_index;
                zero_index.reserve(bitcounts.size());
                for (uint_t i = 0; i < bitcounts.size(); i++) {
                    if (bitcounts[i] == 0) {
                        zero_index.push_back(i);
                    }
                }

                // return bits only whose bitcount are zero
                counts = meas_bits.get_counts(zero_index);
            } else {
                counts = pub_result.data().get_counts();
            }

#endif // USE_RANDOM_SHOTS
        }

        ////// Configuration Recovery, Post Selection, Diagonalization //////

        // Expand counts (map) into (bitstrings[], probs[]).
        auto [bitstring_matrix_full_, probs_arr_full] = counts_to_arrays(counts);
        auto bitstring_matrix_full = bitsets_from_bitstrings(bitstring_matrix_full_);

        std::vector<boost::dynamic_bitset<>> bs_mat_tmp;
        std::vector<double> probs_arr_tmp;
        std::array<std::vector<double>, 2> latest_occupancies, initial_occupancies;
        int n_recovery = static_cast<int>(sqd_data.n_recovery);

        try {
            // Load prior alpha/beta occupancies used as the initial distribution for
            // recovery.
            initial_occupancies =
                load_initial_occupancies("../data/initial_occupancies_fe4s4.json");
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error loading initial occupancies: " << e.what() << std::endl;
            return 1;
        }
        // ===== Configuration recovery loop (n_recovery iterations) =====
        // Each iter: recover_configurations → postselect → subsample → SBD
        // (diagonalize) → update occupancies.
        for (uint64_t i_recovery = 0; i_recovery < n_recovery; ++i_recovery) {
            log(sqd_data, {"start recovery: iteration=", std::to_string(i_recovery)});

            // Iteration 0: feed full bitstring/probability sets and seed recovery from
            // initial occupancies.
            if (i_recovery == 0) {
                bs_mat_tmp = bitstring_matrix_full;
                probs_arr_tmp = probs_arr_full;
                latest_occupancies = initial_occupancies;
            }
            if (sqd_data.mpi_rank == 0) {
                // Recover physically consistent configurations from observed
                // probabilities
                // + prior occupancies.
                auto recovered = Qiskit::addon::sqd::recover_configurations(
                    bitstring_matrix_full, probs_arr_full, latest_occupancies,
                    {num_elec_a, num_elec_b}, rc_rng
                );
                bs_mat_tmp = std::move(recovered.first);
                probs_arr_tmp = std::move(recovered.second);

                std::vector<boost::dynamic_bitset<>> batch;
                // Post-selection: accept bitstrings whose left/right (alpha/beta)
                // Hamming weights match target electron counts.
                auto [postselected_bitstrings, postselected_probs] =
                    Qiskit::addon::sqd::postselect_bitstrings(
                        bs_mat_tmp, probs_arr_tmp,
                        Qiskit::addon::sqd::MatchesRightLeftHamming(
                            num_elec_a, num_elec_b
                        )
                    );
                log(sqd_data, {"Number of postselected bitstrings: ",
                               std::to_string(postselected_bitstrings.size())});

                // Subsample to a single batch of fixed size for SBD, to cap IO/compute
                // per iteration.
                Qiskit::addon::sqd::subsample(
                    batch, postselected_bitstrings, postselected_probs,
                    std::min(samples_per_batch, postselected_bitstrings.size()), rng
                );
                // Write alpha-determinants file for SBD input (includes run id /
                // iteration for traceability).
                diag_data.adetfile = write_alphadets_file(
                    sqd_data, norb, num_elec_a, bitsets_to_bitstrings(batch),
                    sqd_data.samples_per_batch * 2, i_recovery
                );
            }
            // Run SBD to get energy and batch occupancies (interleaved alpha/beta...).
            // Energy goes to logs; occupancies seed the next iteration.
            auto [energy_sci, occs_batch] = sbd_main(sqd_data.comm, diag_data);
            log(sqd_data, {"energy: ", std::to_string(energy_sci)});

            // Convert interleaved [alpha0, beta0, alpha1, beta1, ...] to { alpha[],
            // beta[]
            // }. NOTE: assert ensures occs_batch size matches 2 * alpha.size().
            assert(2 * latest_occupancies[0].size() == occs_batch.size());
            for (std::size_t j = 0; j < latest_occupancies[0].size(); ++j) {
                latest_occupancies[0][j] = occs_batch[2 * j];     // alpha orbital
                latest_occupancies[1][j] = occs_batch[2 * j + 1]; // beta orbital
            }
        }

        // Synchronize and tear down MPI. No MPI calls are allowed beyond this point.
        MPI_Finalize();

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Unhandled exception in main: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception in main" << std::endl;
        MPI_Finalize();
        return 1;
    }
}

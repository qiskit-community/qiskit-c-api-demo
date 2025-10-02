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

#ifndef UCJOP_SPINBALANCED_HPP
#define UCJOP_SPINBALANCED_HPP

#include "gates/orbital_rotation.hpp"
#include "linalg/double_factorized_decomp.hpp"
#include "linalg/expm.hpp"
#include "linalg/matrix_utils.hpp"
#include "ucjop_spinbalanced.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <unordered_set>
#include <unsupported/Eigen/CXX11/Tensor>

namespace std
{
template <>
struct hash<::std::pair<uint64_t, uint64_t>> {
    size_t operator()(const std::pair<uint64_t, uint64_t> &p) const
    {
        return hash<size_t>()(p.first) ^ (hash<size_t>()(p.second) << 1);
    }
};
} // namespace std

namespace ffsim
{

using namespace Eigen;
using Complex = std::complex<double>;

namespace
{

inline void validate_interaction_pairs(
    const std::optional<std::vector<std::pair<uint64_t, uint64_t>>> &interaction_pairs,
    bool ordered
)
{
    if (!interaction_pairs)
        return;
    std::unordered_set<std::pair<uint64_t, uint64_t>> unique_pairs;
    for (const auto &p : *interaction_pairs) {
        if (!unique_pairs.insert(p).second) {
            throw std::runtime_error("Duplicate interaction pairs encountered.");
        }
        if (!ordered && p.first > p.second) {
            throw std::runtime_error(
                "When specifying alpha-alpha or beta-beta interaction pairs, "
                "you must provide only upper triangular pairs. "
                "Got (" +
                std::to_string(p.first) + ", " + std::to_string(p.second) +
                "), which is a lower triangular pair."
            );
        }
    }
}

} // namespace

/**
 * @brief Represents a spin-balanced unitary cluster Jastrow (UCJ) operator.
 *
 * This struct encapsulates the data required to build and manipulate the
 * UCJ operator, including diagonal Coulomb terms, orbital rotations, and
 * optional final orbital rotation.
 */
struct UCJOpSpinBalanced {
    /// Diagonal Coulomb matrices (shape: n_reps × 2 × norb × norb)
    Tensor<Complex, 4> diag_coulomb_mats;

    /// Orbital rotations (shape: n_reps × norb × norb)
    Tensor<Complex, 3> orbital_rotations;

    /// Optional final orbital rotation matrix
    std::optional<MatrixXcd> final_orbital_rotation;

    /**
     * @brief Constructs a UCJOpSpinBalanced instance with optional validation.
     *
     * @param diag_coulomb_mats Coulomb matrices (n_reps × 2 × norb × norb)
     * @param orbital_rotations Orbital rotations (n_reps × norb × norb)
     * @param final_orbital_rotation Optional final rotation matrix
     * @param validate Whether to validate the matrices
     * @param rtol Relative tolerance for validation
     * @param atol Absolute tolerance for validation
     */
    UCJOpSpinBalanced(
        const Tensor<Complex, 4> &diag_coulomb_mats,
        const Tensor<Complex, 3> &orbital_rotations,
        const std::optional<MatrixXcd> &final_orbital_rotation, bool validate,
        double rtol, double atol
    )
      : diag_coulomb_mats(diag_coulomb_mats), orbital_rotations(orbital_rotations),
        final_orbital_rotation(final_orbital_rotation)
    {
        if (validate) {
            if (diag_coulomb_mats.dimension(1) != 2) {
                throw std::runtime_error(
                    "diag_coulomb_mats should have shape (n_reps, 2, norb, norb)"
                );
            }
            if (diag_coulomb_mats.dimension(0) != orbital_rotations.dimension(0)) {
                throw std::runtime_error(
                    "diag_coulomb_mats and orbital_rotations must match in reps"
                );
            }
            uint64_t norb = diag_coulomb_mats.dimension(3);
            MatrixXcd mat(norb, norb);
            for (int i = 0; i < diag_coulomb_mats.dimension(0); ++i) {
                for (int j = 0; j < 2; ++j) {
                    for (int a = 0; a < norb; ++a) {
                        for (int b = 0; b < norb; ++b) {
                            mat(a, b) = diag_coulomb_mats(i, j, a, b);
                        }
                    }
                    // if (!is_real_symmetric(mat, rtol, atol)) {
                    //   throw std::runtime_error("diag_coulomb_mats must be real
                    //   symmetric");
                    // }
                }
            }
            MatrixXcd orb_rot_tmp(norb, norb);
            for (int i = 0; i < orbital_rotations.dimension(0); ++i) {
                for (int a = 0; a < norb; ++a) {
                    for (int b = 0; b < norb; ++b) {
                        orb_rot_tmp(a, b) = orbital_rotations(i, a, b);
                    }
                }
                if (!linalg::is_unitary(orb_rot_tmp, rtol, atol)) {
                    throw std::runtime_error("orbital_rotations must be unitary");
                }
            }

            if (final_orbital_rotation.has_value() &&
                !linalg::is_unitary(final_orbital_rotation.value(), rtol, atol)) {
                throw std::runtime_error("final_orbital_rotation must be unitary");
            }
        }
    }

    /**
     * @brief Computes the number of parameters required for the UCJ operator.
     *
     * @param norb Number of orbitals
     * @param n_reps Number of repetitions
     * @param interaction_pairs Interaction mask for each spin sector
     * @param with_final_orbital_rotation Whether a final orbital rotation is
     * included
     * @return Total number of real parameters
     */
    static size_t n_params(
        uint64_t norb, size_t n_reps,
        const std::array<std::optional<std::vector<std::pair<uint64_t, uint64_t>>>, 2>
            &interaction_pairs,
        bool with_final_orbital_rotation
    )
    {
        validate_interaction_pairs(interaction_pairs[0], false);
        validate_interaction_pairs(interaction_pairs[1], false);

        size_t n_triu = norb * (norb + 1) / 2;
        size_t n_params_aa = n_triu;
        if (const auto &pairs_aa_opt = interaction_pairs[0]; pairs_aa_opt.has_value()) {
            n_params_aa = pairs_aa_opt.value().size();
        }
        size_t n_params_ab = n_triu;
        if (const auto &pairs_ab_opt = interaction_pairs[1]; pairs_ab_opt.has_value()) {
            n_params_ab = pairs_ab_opt.value().size();
        }

        return n_reps * (n_params_aa + n_params_ab + norb * norb) +
               (with_final_orbital_rotation ? norb * norb : 0);
    }

    /**
     * @brief Constructs a UCJ operator from a flat parameter vector.
     *
     * @param params Real-valued parameter vector
     * @param norb Number of orbitals
     * @param n_reps Number of repetitions
     * @param interaction_pairs Optional interaction masks
     * @param with_final_orbital_rotation Whether to include the final orbital
     * rotation
     * @return UCJOpSpinBalanced instance
     */
    static UCJOpSpinBalanced from_parameters(
        const VectorXcd &params, uint64_t norb, size_t n_reps,
        const std::array<std::optional<std::vector<std::pair<uint64_t, uint64_t>>>, 2>
            &interaction_pairs,
        bool with_final_orbital_rotation
    )
    {
        auto expected_params =
            n_params(norb, n_reps, interaction_pairs, with_final_orbital_rotation);
        if (params.size() != expected_params) {
            throw std::runtime_error(
                "Expected " + std::to_string(expected_params) +
                " parameters, but got " + std::to_string(params.size())
            );
        }

        std::vector<std::pair<uint64_t, uint64_t>> triu;
        for (size_t i = 0; i < norb; ++i) {
            for (size_t j = i; j < norb; ++j) {
                triu.emplace_back(i, j);
            }
        }
        const auto &pairs_aa = interaction_pairs[0].value_or(triu);
        const auto &pairs_ab = interaction_pairs[1].value_or(triu);

        size_t index = 0;

        Tensor<Complex, 4> diag_coulomb_mats(
            static_cast<long>(n_reps), static_cast<long>(2), static_cast<long>(norb),
            static_cast<long>(norb)
        );
        diag_coulomb_mats.setZero();
        Tensor<Complex, 3> orbital_rotations(
            static_cast<long>(n_reps), static_cast<long>(norb), static_cast<long>(norb)
        );
        orbital_rotations.setZero();
        for (size_t rep = 0; rep < n_reps; ++rep) {
            MatrixXcd orbital_rotation = orbital_rotation_from_parameters(
                params.segment(
                    static_cast<Index>(index), static_cast<Index>(norb * norb)
                ),
                static_cast<int>(norb), false
            );
            index += norb * norb;
            for (int i = 0; i < norb; ++i) {
                for (int j = 0; j < norb; ++j) {
                    orbital_rotations(static_cast<long>(rep), i, j) =
                        orbital_rotation(i, j);
                }
            }
            for (int t = 0; t < 2; ++t) {
                const auto &pairs = (t == 0) ? pairs_aa : pairs_ab;
                for (const auto &[i, j] : pairs) {
                    Complex val = params(static_cast<Index>(index++));
                    diag_coulomb_mats(
                        static_cast<long>(rep), t, static_cast<long>(i),
                        static_cast<long>(j)
                    ) = val;
                    diag_coulomb_mats(
                        static_cast<long>(rep), t, static_cast<long>(j),
                        static_cast<long>(i)
                    ) = val;
                }
            }
        }

        std::optional<MatrixXcd> final_orbital_rotation;

        if (with_final_orbital_rotation) {
            final_orbital_rotation = orbital_rotation_from_parameters(
                params.segment(
                    static_cast<Index>(index), static_cast<Index>(norb * norb)
                ),
                static_cast<int>(norb), false
            );
        }

        return UCJOpSpinBalanced(
            diag_coulomb_mats, orbital_rotations, final_orbital_rotation, true, 1e-5,
            1e-8
        );
    }

    /**
     * @brief Converts the UCJ operator to a flat parameter vector.
     *
     * @param interaction_pairs Optional interaction masks
     * @return Real-valued parameter vector
     */
    VectorXcd to_parameters(
        const std::array<std::optional<std::vector<std::pair<uint64_t, uint64_t>>>, 2>
            &interaction_pairs
    ) const
    {
        size_t n_reps = diag_coulomb_mats.dimension(0);
        uint64_t norb = diag_coulomb_mats.dimension(3);
        auto total = n_params(
            norb, n_reps, interaction_pairs, final_orbital_rotation.has_value()
        );

        VectorXcd params(total);

        std::vector<std::pair<uint64_t, uint64_t>> triu;
        for (size_t i = 0; i < norb; ++i) {
            for (size_t j = i; j < norb; ++j) {
                triu.emplace_back(i, j);
            }
        }
        const auto &pairs_aa = interaction_pairs[0].value_or(triu);
        const auto &pairs_ab = interaction_pairs[1].value_or(triu);
        size_t index = 0;
        for (int rep = 0; rep < n_reps; ++rep) {
            MatrixXcd orb_rot(norb, norb);
            for (int i = 0; i < norb; ++i) {
                for (int j = 0; j < norb; ++j) {
                    orb_rot(i, j) = orbital_rotations(rep, i, j);
                }
            }
            params.segment(static_cast<Index>(index), static_cast<Index>(norb * norb)) =
                orbital_rotation_to_parameters(orb_rot, false);
            index += norb * norb;

            for (int t = 0; t < 2; ++t) {
                const auto &pairs = (t == 0) ? pairs_aa : pairs_ab;
                for (const auto &[i, j] : pairs) {
                    params(static_cast<Index>(index++)) = diag_coulomb_mats(
                        rep, t, static_cast<long>(i), static_cast<long>(j)
                    );
                }
            }
        }

        if (final_orbital_rotation) {
            params.segment(static_cast<Index>(index), static_cast<Index>(norb * norb)) =
                orbital_rotation_to_parameters(final_orbital_rotation.value(), false);
        }

        return params;
    }

    /**
     * @brief Constructs a UCJ operator from t2 amplitudes.
     *
     * @param t2 t2 amplitude tensor (nocc × nocc × nvrt × nvrt)
     * @param t1 Optional t1 amplitudes (for final orbital rotation)
     * @param n_reps Optional override for number of repetitions
     * @param interaction_pairs Optional interaction masks
     * @param tol Tolerance for truncating eigenvalue decomposition
     * @return UCJOpSpinBalanced instance
     */
    static UCJOpSpinBalanced from_t_amplitudes(
        const Tensor<Complex, 4> &t2, const std::optional<MatrixXcd> &t1,
        std::optional<size_t> n_reps,
        const std::array<std::optional<std::vector<std::pair<uint64_t, uint64_t>>>, 2>
            &interaction_pairs,
        double tol
    )
    {
        validate_interaction_pairs(interaction_pairs[0], false);
        validate_interaction_pairs(interaction_pairs[1], false);
        size_t nocc = t2.dimension(0);
        size_t nvrt = t2.dimension(2);
        uint64_t norb = nocc + nvrt;

        auto [diag_coulomb_mats, orbital_rotations] =
            linalg::double_factorized_t2(t2, tol, std::nullopt);
        size_t n_vecs = diag_coulomb_mats.size() / (norb * norb);
        size_t orb_vecs = orbital_rotations.size() / (norb * norb);
        Eigen::DSizes<ptrdiff_t, 3> diag_shape_3d(
            static_cast<ptrdiff_t>(n_vecs), static_cast<ptrdiff_t>(norb),
            static_cast<ptrdiff_t>(norb)
        );
        Eigen::DSizes<ptrdiff_t, 3> orb_shape_3d(
            static_cast<ptrdiff_t>(orb_vecs), static_cast<ptrdiff_t>(norb),
            static_cast<ptrdiff_t>(norb)
        );

        Tensor<Complex, 3> diag_coulomb_mats_reshaped =
            diag_coulomb_mats.reshape(diag_shape_3d);
        Tensor<Complex, 3> orbital_rotations_reshaped =
            orbital_rotations.reshape(orb_shape_3d);
        if (n_reps.has_value()) {
            size_t n_terms = std::min(n_reps.value(), n_vecs);
            Eigen::DSizes<Index, 3> slice_shape{
                static_cast<Index>(n_terms), static_cast<long>(norb),
                static_cast<long>(norb)
            };
            auto diag_slice = diag_coulomb_mats_reshaped.slice(
                array<Index, 3>{0, 0, 0},
                array<Index, 3>{
                    static_cast<long>(n_terms), static_cast<long>(norb),
                    static_cast<long>(norb)
                }
            );
            Tensor<Complex, 3> diag_tmp(slice_shape);
            diag_tmp.setZero();
            diag_tmp = diag_slice;
            diag_coulomb_mats_reshaped = diag_tmp;

            auto orb_slice = orbital_rotations_reshaped.slice(
                array<Index, 3>{0, 0, 0},
                array<Index, 3>{
                    static_cast<long>(n_terms), static_cast<long>(norb),
                    static_cast<long>(norb)
                }
            );
            Tensor<Complex, 3> orb_tmp(slice_shape);
            orb_tmp.setZero();
            orb_tmp = orb_slice;
            orbital_rotations_reshaped = orb_tmp;

            n_vecs = n_terms;
        }
        Tensor<Complex, 4> diag_coulomb_mats_stacked(
            static_cast<long>(n_vecs), 2, static_cast<long>(norb),
            static_cast<long>(norb)
        );
        diag_coulomb_mats_stacked.setZero();
        for (int i = 0; i < n_vecs; ++i) {
            for (int j = 0; j < norb; ++j) {
                for (int k = 0; k < norb; ++k) {
                    Complex val = diag_coulomb_mats_reshaped(i, j, k);
                    diag_coulomb_mats_stacked(i, 0, j, k) = val;
                    diag_coulomb_mats_stacked(i, 1, j, k) = val;
                }
            }
        }

        if (n_reps.has_value() && n_vecs < n_reps.value()) {
            size_t pad = n_reps.value() - n_vecs;
            Tensor<Complex, 4> diag_coulomb_mats_pad(
                static_cast<long>(pad), 2, static_cast<long>(norb),
                static_cast<long>(norb)
            );
            diag_coulomb_mats_pad.setZero();
            diag_coulomb_mats_stacked =
                diag_coulomb_mats_stacked.concatenate(diag_coulomb_mats_pad, 0);

            Tensor<Complex, 3> orbital_rotations_pad(
                static_cast<long>(pad), static_cast<long>(norb), static_cast<long>(norb)
            );
            orbital_rotations_pad.setZero();
            for (int p = 0; p < pad; ++p) {
                for (int i = 0; i < norb; ++i) {
                    orbital_rotations_pad(p, i, i) = Complex(1.0, 0.0);
                }
            }
            orbital_rotations_reshaped =
                orbital_rotations_reshaped.concatenate(orbital_rotations_pad, 0);
        }

        std::optional<MatrixXcd> final_orbital_rotation = std::nullopt;
        if (t1.has_value()) {
            const MatrixXcd &t1_mat = t1.value();
            MatrixXcd final_orbital_rotation_generator =
                MatrixXcd::Zero(static_cast<Index>(norb), static_cast<Index>(norb));
            final_orbital_rotation_generator.block(
                0, static_cast<Index>(nocc), static_cast<Index>(nocc),
                static_cast<Index>(nvrt)
            ) = t1_mat;
            final_orbital_rotation_generator.block(
                static_cast<Index>(nocc), 0, static_cast<Index>(nvrt),
                static_cast<Index>(nocc)
            ) = -t1_mat.adjoint();
            final_orbital_rotation = linalg::expm(final_orbital_rotation_generator);
        }

        auto mask_and_zero =
            [&](
                Tensor<Complex, 4> &tensor, size_t index,
                const std::optional<std::vector<std::pair<uint64_t, uint64_t>>> &pairs
            ) {
                if (!pairs.has_value())
                    return;
                size_t n_vecs = tensor.dimension(0);

                std::vector<std::vector<bool>> mask(
                    norb, std::vector<bool>(norb, false)
                );
                for (const auto &[i, j] : pairs.value()) {
                    mask[i][j] = true;
                    mask[j][i] = true;
                }

                for (int v = 0; v < n_vecs; ++v) {
                    for (int i = 0; i < norb; ++i) {
                        for (int j = 0; j < norb; ++j) {
                            if (!mask[i][j]) {
                                tensor(v, static_cast<long>(index), i, j) =
                                    Complex(0.0, 0.0);
                            }
                        }
                    }
                }
            };

        mask_and_zero(diag_coulomb_mats_stacked, 0, interaction_pairs[0]);
        mask_and_zero(diag_coulomb_mats_stacked, 1, interaction_pairs[1]);

        return UCJOpSpinBalanced(
            diag_coulomb_mats_stacked, orbital_rotations_reshaped,
            final_orbital_rotation, true, 1e-5, 1e-8
        );
    }

    /**
     * @brief Returns the number of orbitals (norb).
     */
    size_t norb() const
    {
        return diag_coulomb_mats.dimension(3);
    }

    /**
     * @brief Returns the number of repetitions (n_reps).
     */
    size_t n_reps() const
    {
        return diag_coulomb_mats.dimension(0);
    }
};

} // namespace ffsim

#endif // UCJOP_SPINBALANCED_HPP
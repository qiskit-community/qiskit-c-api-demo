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

#ifndef DIAG_COULOMB_HPP
#define DIAG_COULOMB_HPP

#include "ffsim/linalg/givens.hpp"
#include "orbital_rotation.hpp"
#include <Eigen/Dense>
#include <complex>
#include <optional>
#include <vector>

namespace ffsim
{
namespace gates
{
using namespace Eigen;

/**
 * @brief Matrix type for diagonal Coulomb evolution.
 * @details This enum class defines the types of matrices used in the diagonal
 * Coulomb evolution process.
 *
 * - Single: Represents a single matrix.
 * - Triple: Represents a set of three matrices.
 */
enum class MatType
{
    Single, ///< Represents a single matrix for all orbitals.
    Triple, ///< Represents a set of three matrices for (aa, ab, bb) orbitals.
};

/**
 * @brief Matrix structure for diagonal Coulomb evolution.
 * @details This structure holds the type of matrix and the actual matrix data.
 *
 * - type: The type of matrix (Single or Triple).
 * - single: The single matrix (used when type is Single).
 * - triple: The three matrices (used when type is Triple).
 */
struct Mat
{
    MatType type;                                   ///< Type of matrix (Single or Triple).
    MatrixXcd single;                               ///< Single matrix (used when type is Single).
    std::array<std::optional<MatrixXcd>, 3> triple; ///< Three matrices (used when type is Triple).
};

std::tuple<MatrixXcd, MatrixXcd, MatrixXcd>
get_mat_exp(const Mat& mat, double time, uint64_t norb, bool z_representation,
            MatrixXcd& mat_exp_aa, MatrixXcd& mat_exp_ab, MatrixXcd& mat_exp_bb)
{
    const Complex I(0.0, 1.0);
    if (mat.type == MatType::Single)
    {
        MatrixXcd mat_aa = mat.single;
        MatrixXcd mat_ab = mat.single;
        for (size_t i = 0; i < norb; ++i)
        {
            mat_aa(i, i) *= 0.5;
        }
        if (z_representation)
        {
            mat_aa *= 0.25;
            mat_ab *= 0.25;
        }

        mat_exp_aa = (-I * time * mat_aa.array()).exp().matrix();
        mat_exp_ab = (-I * time * mat_ab.array()).exp().matrix();
        mat_exp_bb = mat_exp_aa;

        return {mat_exp_aa, mat_exp_ab, mat_exp_bb};
    }
    else
    {
        if (mat.triple[0].has_value())
        {
            MatrixXcd mat_aa = mat.triple[0].value();
            for (size_t i = 0; i < norb; ++i)
            {
                mat_aa(i, i) *= 0.5;
            }
            if (z_representation)
            {
                mat_aa *= 0.25;
            }
            mat_exp_aa = (-I * time * mat_aa.array()).exp().matrix();
        }
        else
        {
            mat_exp_aa = MatrixXcd::Identity(norb, norb);
        }

        if (mat.triple[1].has_value())
        {
            MatrixXcd mat_ab = mat.triple[1].value();
            if (z_representation)
            {
                mat_ab *= 0.25;
            }
            mat_exp_ab = (-I * time * mat_ab.array()).exp().matrix();
        }
        else
        {
            mat_exp_ab = MatrixXcd::Identity(norb, norb);
        }
        if (mat.triple[2].has_value())
        {
            MatrixXcd mat_bb = mat.triple[2].value();
            for (size_t i = 0; i < norb; ++i)
            {
                mat_bb(i, i) *= 0.5;
            }
            if (z_representation)
            {
                mat_bb *= 0.25;
            }
            mat_exp_bb = (-I * time * mat_bb.array()).exp().matrix();
        }
        else
        {
            mat_exp_bb = MatrixXcd::Identity(norb, norb);
        }
        return {mat_exp_aa, mat_exp_ab, mat_exp_bb};
    }
}

OrbitalRotation conjugate_orbital_rotation(const OrbitalRotation& orb_rot)
{
    if (orb_rot.type == OrbitalRotationType::Spinless)
    {
        return OrbitalRotation{OrbitalRotationType::Spinless,
                               orb_rot.spinless.adjoint(),
                               {std::nullopt, std::nullopt}};
    }
    else
    {
        std::array<std::optional<MatrixXcd>, 2> spinfull_conj;
        for (int i = 0; i < 2; ++i)
        {
            if (orb_rot.spinfull[i].has_value())
            {
                spinfull_conj[i] = orb_rot.spinfull[i].value().adjoint();
            }
            else
            {
                spinfull_conj[i] = std::nullopt;
            }
        }
        return OrbitalRotation{OrbitalRotationType::Spinfull, MatrixXcd(), spinfull_conj};
    }
}

void apply_diag_coulomb_evolution_in_place_num_rep(
    MatrixXcd& vec, const MatrixXcd& mat_exp_aa, const MatrixXcd& mat_exp_ab,
    const MatrixXcd& mat_exp_bb, uint64_t norb,
    const std::vector<std::vector<size_t>>& occupations_a,
    const std::vector<std::vector<size_t>>& occupations_b)
{
    const size_t dim_a = vec.rows();
    const size_t dim_b = vec.cols();
    const size_t n_alpha = occupations_a.empty() ? 0 : occupations_a[0].size();
    const size_t n_beta = occupations_b.empty() ? 0 : occupations_b[0].size();

    ArrayXcd alpha_phases = ArrayXcd::Zero(dim_a);
    ArrayXcd beta_phases = ArrayXcd::Zero(dim_b);
    ArrayXXcd phase_map = ArrayXXcd::Ones(dim_a, dim_b);

    for (size_t i = 0; i < dim_b; ++i)
    {
        Complex phase = 1.0;
        for (size_t j = 0; j < n_beta; ++j)
        {
            size_t orb_1 = occupations_b[i][j];
            for (size_t k = j; k < n_beta; ++k)
            {
                size_t orb_2 = occupations_b[i][k];
                phase *= mat_exp_bb(orb_1, orb_2);
            }
        }
        beta_phases(i) = phase;
    }
    for (size_t i = 0; i < dim_a; ++i)
    {
        Complex phase = 1.0;
        for (size_t j = 0; j < n_alpha; ++j)
        {
            size_t orb_1 = occupations_a[i][j];
            phase_map.row(i) = phase_map.row(i).array() * mat_exp_ab.row(orb_1).array();
            for (size_t k = j; k < n_alpha; ++k)
            {
                size_t orb_2 = occupations_a[i][k];
                phase *= mat_exp_aa(orb_1, orb_2);
            }
        }
        alpha_phases(i) = phase;
    }
    for (size_t i = 0; i < dim_a; ++i)
    {
        for (size_t j = 0; j < dim_b; ++j)
        {
            Complex phase = alpha_phases(i) * beta_phases(j);
            for (size_t k = 0; k < n_beta; ++k)
            {
                size_t orb = occupations_b[j][k];
                phase *= phase_map(i, orb);
            }
            vec(i, j) *= phase;
        }
    }
}

void apply_diag_coulomb_evolution_in_place_z_rep(MatrixXcd& vec, MatrixXcd& mat_exp_aa,
                                                 MatrixXcd& mat_exp_ab, MatrixXcd& mat_exp_bb,
                                                 uint64_t norb,
                                                 const std::vector<int64_t>& strings_a,
                                                 const std::vector<int64_t>& strings_b)
{
    MatrixXcd mat_exp_aa_conj = mat_exp_aa.conjugate();
    MatrixXcd mat_exp_ab_conj = mat_exp_ab.conjugate();
    MatrixXcd mat_exp_bb_conj = mat_exp_bb.conjugate();

    size_t dim_a = vec.rows();
    size_t dim_b = vec.cols();

    ArrayXcd alpha_phases = ArrayXcd::Zero(dim_a);
    ArrayXcd beta_phases = ArrayXcd::Zero(dim_b);
    ArrayXXcd phase_map = ArrayXXcd::Ones(dim_a, norb);

    for (size_t i = 0; i < dim_b; ++i)
    {
        Complex phase = 1.0;
        int64_t str0 = strings_b[i];
        for (size_t j = 0; j < norb; ++j)
        {
            bool sign_j = (str0 >> j) & 1;
            for (size_t k = j + 1; k < norb; ++k)
            {
                bool sign_k = (str0 >> k) & 1;
                Complex this_phase = (sign_j ^ sign_k) ? mat_exp_bb_conj(j, k) : mat_exp_bb(j, k);
                phase *= this_phase;
            }
        }
        beta_phases(i) = phase;
    }

    for (size_t i = 0; i < dim_a; ++i)
    {
        Complex phase = 1.0;
        int64_t str0 = strings_a[i];
        for (size_t j = 0; j < norb; ++j)
        {
            bool sign_j = (str0 >> j) & 1;
            auto this_row = sign_j ? mat_exp_ab_conj.row(j) : mat_exp_ab.row(j);
            for (size_t k = 0; k < norb; ++k)
            {
                phase_map(i, k) *= this_row(k);
            }

            for (size_t k = j + 1; k < norb; ++k)
            {
                bool sign_k = (str0 >> k) & 1;
                Complex this_phase = (sign_j ^ sign_k) ? mat_exp_aa_conj(j, k) : mat_exp_aa(j, k);
                phase *= this_phase;
            }
        }
        alpha_phases(i) = phase;
    }

    for (size_t i = 0; i < dim_a; ++i)
    {
        for (size_t j = 0; j < dim_b; ++j)
        {
            Complex phase = alpha_phases(i) * beta_phases(j);
            int64_t str0 = strings_b[j];
            for (size_t k = 0; k < norb; ++k)
            {
                bool sign = (str0 >> k) & 1;
                phase *= sign ? std::conj(phase_map(i, k)) : phase_map(i, k);
            }
            vec(i, j) *= phase;
        }
    }
}

VectorXcd apply_diag_coulomb_evolution_spinfull(
    VectorXcd vec, const Mat& mat, double time, uint64_t norb, std::pair<uint64_t, uint64_t> nelec,
    const std::optional<OrbitalRotation>& orbital_rotation, bool z_representation)
{
    MatrixXcd mat_exp_aa, mat_exp_ab, mat_exp_bb;
    std::tie(mat_exp_aa, mat_exp_ab, mat_exp_bb) =
        get_mat_exp(mat, time, norb, z_representation, mat_exp_aa, mat_exp_ab, mat_exp_bb);

    size_t n_alpha = nelec.first;
    size_t n_beta = nelec.second;
    size_t dim_a = binomial(norb, n_alpha);
    size_t dim_b = binomial(norb, n_beta);

    if (orbital_rotation.has_value())
    {
        auto conj_rot = conjugate_orbital_rotation(orbital_rotation.value());
        vec = apply_orbital_rotation(vec, conj_rot, norb,
                                     Electron{ElectronType::Spinfull, 0, {n_alpha, n_beta}});
    }
    MatrixXcd vec_reshaped = Map<MatrixXcd>(vec.data(), dim_a, dim_b);
    std::vector<size_t> orb_list(norb);
    std::iota(orb_list.begin(), orb_list.end(), 0);
    if (z_representation)
    {
        auto strings_a = make_strings(orb_list, n_alpha);
        auto strings_b = make_strings(orb_list, n_beta);
        apply_diag_coulomb_evolution_in_place_z_rep(vec_reshaped, mat_exp_aa, mat_exp_ab,
                                                    mat_exp_bb, norb, strings_a, strings_b);
    }
    else
    {
        auto occupations_a = gen_occslst(orb_list, n_alpha);
        auto occupations_b = gen_occslst(orb_list, n_beta);
        apply_diag_coulomb_evolution_in_place_num_rep(
            vec_reshaped, mat_exp_aa, mat_exp_ab, mat_exp_bb, norb, occupations_a, occupations_b);
    }

    VectorXcd vec_flat = Map<VectorXcd>(vec_reshaped.data(), dim_a * dim_b);

    if (orbital_rotation.has_value())
    {
        vec_flat = apply_orbital_rotation(vec_flat, orbital_rotation.value(), norb,
                                          Electron{ElectronType::Spinfull, 0, {n_alpha, n_beta}});
    }

    return vec_flat;
}

/**
 * @brief Applies diagonal Coulomb evolution to a vector.
 * @details This function applies the diagonal Coulomb evolution operator to a
 * given vector.
 *
 * @param vec The input vector to be evolved.
 * @param mat The matrix representing the Coulomb interaction.
 * @param time The time parameter for the evolution.
 * @param norb The number of orbitals.
 * @param nelec The number of electrons.
 * @param orb_rot The optional orbital rotation to be applied.
 * @param z_representation Flag indicating whether to use z-representation.
 *
 * @return The evolved vector after applying the diagonal Coulomb evolution
 * operator.
 */
VectorXcd apply_diag_coulomb_evolution(const VectorXcd& vec, const Mat& mat, double time,
                                       uint64_t norb, const Electron& nelec,
                                       const std::optional<OrbitalRotation>& orb_rot,
                                       bool z_representation)
{
    if (nelec.type == ElectronType::Spinless)
    {
        if (mat.type != MatType::Single)
        {
            throw std::runtime_error("Expected single matrix for spinless electron type");
        }
        if (z_representation)
        {
            throw std::runtime_error(
                "z_representation is not supported for spinless electron type");
        }
        return apply_diag_coulomb_evolution_spinfull(vec, mat, time, norb, {nelec.spinless, 0},
                                                     orb_rot, false);
    }
    else
    {
        return apply_diag_coulomb_evolution_spinfull(vec, mat, time, norb, nelec.spinfull, orb_rot,
                                                     z_representation);
    }
}

} // namespace gates
} // namespace ffsim

#endif // DIAG_COULOMB_HPP
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

#ifndef ORBITAL_ROTATION_HPP
#define ORBITAL_ROTATION_HPP

#include "ffsim/gates/phase_shift.hpp"
#include "ffsim/linalg/givens.hpp"

#include <Eigen/Dense>
#include <complex>
#include <numeric>
#include <optional>
#include <vector>

namespace ffsim
{
namespace gates
{

using namespace Eigen;

std::size_t binomial(std::size_t n, std::size_t k)
{
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;

    if (k > n - k)
        k = n - k;

    std::size_t result = 1;
    for (std::size_t i = 1; i <= k; ++i)
    {
        result *= (n - k + i);
        result /= i;
    }
    return result;
}

std::vector<size_t> shifted_orbitals(uint64_t norb, const std::vector<size_t>& target_orbs)
{
    std::vector<size_t> orbitals(norb - target_orbs.size());
    std::iota(orbitals.begin(), orbitals.end(), 0);

    std::vector<std::pair<uint64_t, uint64_t>> values;

    for (size_t i = 0; i < target_orbs.size(); ++i)
    {
        values.emplace_back(target_orbs[i], norb - target_orbs.size() + i);
    }

    std::sort(values.begin(), values.end());

    for (const auto& [idx, val] : values)
    {
        orbitals.insert(orbitals.begin() + idx, val);
    }

    return orbitals;
}

std::vector<int64_t> make_strings(const std::vector<size_t>& orb_list, size_t nelec)
{
    assert(orb_list.size() < 64);
    if (nelec == 0)
    {
        return {0};
    }
    else if (nelec > orb_list.size())
    {
        return {};
    }

    std::function<std::vector<int64_t>(const std::vector<size_t>&, size_t)> gen_str_iter =
        [&](const std::vector<size_t>& orb_list, size_t nelec) -> std::vector<int64_t>
    {
        if (nelec == 1)
        {
            std::vector<int64_t> res;
            for (auto i : orb_list)
                res.push_back(1LL << i);
            return res;
        }
        else if (nelec >= orb_list.size())
        {
            int64_t sum = 0;
            for (auto i : orb_list)
                sum |= (1LL << i);
            return {sum};
        }
        else
        {
            std::vector<int64_t> res = gen_str_iter({orb_list.begin(), orb_list.end() - 1}, nelec);
            for (auto n : gen_str_iter({orb_list.begin(), orb_list.end() - 1}, nelec - 1))
            {
                res.push_back(n | (1LL << orb_list.back()));
            }
            return res;
        }
    };

    auto strings = gen_str_iter(orb_list, nelec);
    assert(strings.size() == binomial(orb_list.size(), nelec));
    return strings;
}

std::vector<size_t> zero_one_subspace_indices(uint64_t norb, size_t nocc,
                                              const std::vector<size_t>& target_orbs)
{
    std::vector<size_t> orbitals = shifted_orbitals(norb, target_orbs);
    std::vector<int64_t> strings = make_strings(orbitals, nocc);

    std::vector<size_t> indices(strings.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return strings[i] < strings[j]; });

    size_t n00 = binomial(norb - 2, nocc);
    size_t n11 = (nocc >= 2) ? binomial(norb - 2, nocc - 2) : 0;

    return std::vector<size_t>(indices.begin() + n00, indices.end() - n11);
}

std::vector<size_t> one_subspace_indices(uint64_t norb, size_t nocc,
                                         const std::vector<size_t>& target_orbs)
{
    std::vector<size_t> orbitals = shifted_orbitals(norb, target_orbs);
    std::vector<int64_t> strings = make_strings(orbitals, nocc);

    std::vector<size_t> indices(strings.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return strings[i] < strings[j]; });

    size_t n0 = binomial(norb, nocc);
    if (nocc >= target_orbs.size())
    {
        n0 -= binomial(norb - target_orbs.size(), nocc - target_orbs.size());
    }

    return std::vector<size_t>(indices.begin() + n0, indices.end());
}

/**
 * @brief Type of orbital rotation.
 *
 * - Spinless: One matrix for both spin-up and spin-down orbitals.
 * - Spinfull: Separate matrices for spin-up and spin-down orbitals.
 */
enum class OrbitalRotationType
{
    Spinless, ///< Shared matrix for both spin components.
    Spinfull  ///< Separate matrices for each spin component.
};

/**
 * @brief Type of electron configuration.
 *
 * - Spinless: One spin sector.
 * - Spinfull: Two separate spin sectors.
 */
enum class ElectronType
{
    Spinless, ///< Single spin type.
    Spinfull, ///< Spin-resolved.
};

/**
 * @brief Orbital rotation specification.
 *
 * Represents either a single orbital rotation matrix (`spinless`)
 * or two matrices for alpha and beta (`spinfull`).
 */
struct OrbitalRotation
{
    OrbitalRotationType type; ///< Indicates if the rotation is spinfull or spinless.
    MatrixXcd spinless;       ///< Orbital rotation matrix for spinless case.
    std::array<std::optional<MatrixXcd>, 2> spinfull; ///< [0]: alpha, [1]: beta
};

/**
 * @brief Electron occupation information.
 *
 * Indicates the number of electrons per spin channel.
 */
struct Electron
{
    ElectronType type;                      ///< Spinless or spinfull electron setting.
    size_t spinless;                        ///< Number of electrons for spinless case.
    std::pair<uint64_t, uint64_t> spinfull; ///< (alpha, beta) electron counts for spinfull.
};

std::pair<std::optional<std::pair<std::vector<linalg::GivensRotation>, VectorXcd>>,
          std::optional<std::pair<std::vector<linalg::GivensRotation>, VectorXcd>>>
get_givens_decomposition(const OrbitalRotation& mat)
{
    if (mat.type == OrbitalRotationType::Spinless)
    {
        auto decomp = linalg::givens_decomposition(mat.spinless);
        return {decomp, decomp};
    }
    else
    {
        std::optional<std::pair<std::vector<linalg::GivensRotation>, VectorXcd>> decomp_a, decomp_b;
        if (mat.spinfull[0].has_value())
        {
            decomp_a = linalg::givens_decomposition(mat.spinfull[0].value());
        }
        if (mat.spinfull[1].has_value())
        {
            decomp_b = linalg::givens_decomposition(mat.spinfull[1].value());
        }
        return {decomp_a, decomp_b};
    }
}

void apply_givens_rotation_in_place(MatrixXcd& vec, double c, Complex s,
                                    const std::vector<size_t>& slice1,
                                    const std::vector<size_t>& slice2)
{
    int dim_b = vec.cols();
    double s_abs = std::abs(s);
    double angle = std::arg(s);
    Complex phase(std::cos(angle), std::sin(angle));
    Complex phase_conj = std::conj(phase);

    for (size_t k = 0; k < slice1.size(); ++k)
    {
        size_t i = slice1[k];
        size_t j = slice2[k];
        VectorXcd row_i = vec.row(i);
        VectorXcd row_j = vec.row(j);

        // altanative method: zscal -> zdrot -> zscal
        row_i *= phase_conj;

        for (int n = 0; n < dim_b; ++n)
        {
            Complex temp = c * row_i[n] + s_abs * row_j[n];
            row_j[n] = c * row_j[n] - s_abs * row_i[n];
            row_i[n] = temp;
        }

        row_i *= phase;
        vec.row(i) = row_i;
        vec.row(j) = row_j;
    }
}

void apply_orbital_rotation_adjacent_spin_inplace(MatrixXcd& vec, double c, Complex s,
                                                  const std::pair<uint64_t, uint64_t>& target_orbs,
                                                  uint64_t norb, size_t nelec)
{
    size_t i = target_orbs.first;
    size_t j = target_orbs.second;
    assert((i == j + 1 || i == j - 1) && "Target orbitals must be adjacent.");

    std::vector<size_t> indices = one_subspace_indices(norb, nelec, {i, j});
    size_t half = indices.size() / 2;
    std::vector<size_t> silce1(indices.begin(), indices.begin() + half);
    std::vector<size_t> slice2(indices.begin() + half, indices.end());
    apply_givens_rotation_in_place(vec, c, s, silce1, slice2);
}

VectorXcd apply_orbital_rotation_spinless(VectorXcd& vec, const MatrixXcd& mat, uint64_t norb,
                                          size_t nelec)
{
    auto [rotations, phase_shifts] = linalg::givens_decomposition(mat);
    MatrixXcd reshaped = vec;
    reshaped.resize(vec.size(), 1);

    for (const auto& rotation : rotations)
    {
        apply_orbital_rotation_adjacent_spin_inplace(reshaped, rotation.c, std::conj(rotation.s),
                                                     std::make_pair(rotation.i, rotation.j), norb,
                                                     nelec);
    }

    for (size_t i = 0; i < phase_shifts.size(); ++i)
    {
        auto indices = one_subspace_indices(norb, nelec, {i});
        apply_phase_shift_in_place(reshaped, phase_shifts(i), indices);
    }

    return Map<VectorXcd>(reshaped.data(), vec.size());
}

VectorXcd apply_orbital_rotation_spinfull(VectorXcd& vec,
                                          const std::array<std::optional<MatrixXcd>, 2>& mat,
                                          uint64_t norb, const std::pair<uint64_t, uint64_t>& nelec)
{

    size_t n_alpha = nelec.first;
    size_t n_beta = nelec.second;
    size_t dim_a = binomial(norb, n_alpha);
    size_t dim_b = binomial(norb, n_beta);

    MatrixXcd reshaped = vec;
    reshaped.resize(dim_a, dim_b);

    OrbitalRotation rot;
    rot.type = OrbitalRotationType::Spinfull;
    rot.spinfull = mat;

    auto [decomp_a, decomp_b] = get_givens_decomposition(rot);

    if (decomp_a)
    {
        auto& [rots_a, phase_shifts_a] = *decomp_a;
        for (const auto& rotation : rots_a)
        {
            apply_orbital_rotation_adjacent_spin_inplace(
                reshaped, rotation.c, std::conj(rotation.s), std::make_pair(rotation.i, rotation.j),
                norb, n_alpha);
        }
        for (size_t i = 0; i < phase_shifts_a.size(); ++i)
        {
            auto indices = one_subspace_indices(norb, n_alpha, {i});
            apply_phase_shift_in_place(reshaped, phase_shifts_a(i), indices);
        }
    }
    if (decomp_b)
    {
        MatrixXcd transposed = reshaped.transpose();
        auto& [rots_b, phase_shifts_b] = *decomp_b;
        for (const auto& rotation : rots_b)
        {
            apply_orbital_rotation_adjacent_spin_inplace(
                transposed, rotation.c, std::conj(rotation.s),
                std::make_pair(rotation.i, rotation.j), norb, n_beta);
        }
        for (size_t i = 0; i < phase_shifts_b.size(); ++i)
        {
            auto indices = one_subspace_indices(norb, n_beta, {i});
            apply_phase_shift_in_place(transposed, phase_shifts_b(i), indices);
        }
        reshaped = transposed.transpose();
    }
    return Map<VectorXcd>(reshaped.data(), vec.size());
}

/**
 * @brief Applies the orbital rotation to a wavefunction.
 *
 * Applies `rotation` to `vec`, interpreting the structure depending on spin
 * type.
 *
 * @param vec Input/output state vector.
 * @param rotation Orbital rotation specification.
 * @param norb Number of orbitals.
 * @param nelec Electron occupation info.
 *
 * @return Rotated vector.
 */
VectorXcd apply_orbital_rotation(VectorXcd& vec, const OrbitalRotation& rotation, uint64_t norb,
                                 const Electron& nelec)
{
    if (nelec.type == ElectronType::Spinless)
    {
        if (rotation.type != OrbitalRotationType::Spinless)
        {
            throw std::runtime_error(
                "Expected spinless orbital rotation with spinless electron type");
        }
        return apply_orbital_rotation_spinless(vec, rotation.spinless, norb, nelec.spinless);
    }
    else
    {
        return apply_orbital_rotation_spinfull(vec, rotation.spinfull, norb, nelec.spinfull);
    }
}

/**
 * @brief Generates all possible occupation lists.
 *
 * Returns all combinations of `nelec` orbitals chosen from `orb_list`.
 *
 * @param orb_list Available orbital indices.
 * @param nelec Number of electrons.
 *
 * @return Vector of occupation lists (each list is a set of orbital indices).
 */
std::vector<std::vector<size_t>> gen_occslst(const std::vector<size_t>& orb_list, size_t nelec)
{
    if (nelec == 0)
    {
        return {std::vector<size_t>(0)};
    }
    else if (nelec > orb_list.size())
    {
        return {{}};
    }
    std::function<std::vector<std::vector<size_t>>(const std::vector<size_t>&, size_t)>
        gen_occs_iter =
            [&](const std::vector<size_t>& list, size_t n) -> std::vector<std::vector<size_t>>
    {
        if (n == 1)
        {
            std::vector<std::vector<size_t>> res;
            for (auto i : list)
            {
                res.push_back({i});
            }
            return res;
        }
        else if (n >= list.size())
        {
            return {list};
        }
        else
        {
            std::vector<std::vector<size_t>> res = gen_occs_iter({list.begin(), list.end() - 1}, n);
            for (auto& v : gen_occs_iter({list.begin(), list.end() - 1}, n - 1))
            {
                v.push_back(list.back());
                res.push_back(v);
            }
            return res;
        }
    };

    auto res = gen_occs_iter(orb_list, nelec);
    std::sort(res.begin(), res.end());
    return res;
}

} // namespace gates
} // namespace ffsim

#endif // ORBITAL_ROTATION_HPP
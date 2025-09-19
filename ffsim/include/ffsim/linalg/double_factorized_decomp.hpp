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

#ifndef DOUBLE_FACTORIZED_T2_HPP
#define DOUBLE_FACTORIZED_T2_HPP

#include <Eigen/Dense>
#include <complex>
#include <optional>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace ffsim
{
namespace linalg
{

using namespace Eigen;
using Complex = std::complex<double>;

/**
 * @brief Computes the quadrature of a complex matrix.
 *
 * This function applies a quadrature transformation to the input matrix.
 *
 * @param mat The input complex matrix
 * @param sign The sign of the imaginary unit (1 or -1)
 * @return The quadrature-transformed matrix
 */
MatrixXcd quadrature(const MatrixXcd& mat, int sign)
{
    Complex i(0.0, 1.0);
    Complex factor = 0.5 * (1.0 - static_cast<double>(sign) * i);
    return factor * (mat + static_cast<double>(sign) * i * mat.adjoint());
}

/**
 * @brief Computes a double factorization of the T2 amplitude tensor.
 *
 * This function approximates the four-index T2 amplitude tensor `t2_amplitudes`
 * as a sum over products of two-body Coulomb-like diagonal matrices and orbital
 * rotations.
 *
 * The returned tuple consists of:
 * - `diag_coulomb_out`: A tensor of diagonal Coulomb operators in the orbital
 * basis.
 * - `orbital_rotations`: A tensor of orbital rotation matrices (one per term).
 *
 * @param t2_amplitudes The original T2 amplitude tensor (shape: `[nocc, nocc,
 * nvrt, nvrt]`)
 * @param tol The tolerance for truncating small eigenvalues in the spectral
 * decomposition.
 * @param max_vecs Optional limit on the number of terms (vectors) returned.
 *
 * @return A tuple of (diag_coulomb_out, orbital_rotations), both tensors of
 * shape `[n_vecs, 2, norb, norb]`
 */
std::tuple<Tensor<Complex, 4>, Tensor<Complex, 4>>
double_factorized_t2(const Tensor<Complex, 4>& t2_amplitudes, double tol,
                     std::optional<size_t> max_vecs)
{

    const auto dims = t2_amplitudes.dimensions();
    const int nocc = dims[0];
    const int nvrt = dims[2];
    const int norb = nocc + nvrt;
    const int n_pairs = nocc * nvrt;

    std::vector<int> occ, vrt;
    for (int i = 0; i < nocc; ++i)
    {
        for (int j = 0; j < nvrt; ++j)
        {
            occ.push_back(i);
            vrt.push_back(j);
        }
    }

    std::vector<int> row, col;
    for (int i = 0; i < nocc; ++i)
    {
        for (int a = nocc; a < norb; ++a)
        {
            col.push_back(i);
            row.push_back(a);
        }
    }

    MatrixXcd t2_mat = MatrixXcd::Zero(n_pairs, n_pairs);
    for (int p = 0; p < n_pairs; ++p)
    {
        for (int q = 0; q < n_pairs; ++q)
        {
            t2_mat(p, q) = t2_amplitudes(occ[p], occ[q], vrt[p], vrt[q]);
        }
    }

    SelfAdjointEigenSolver<MatrixXcd> es(t2_mat);
    VectorXcd eigs = es.eigenvalues();
    MatrixXcd vecs = es.eigenvectors();

    std::vector<size_t> indices(eigs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return std::abs(eigs[j]) < std::abs(eigs[i]); });

    VectorXcd eigs_sorted(eigs.size());
    MatrixXcd vecs_sorted(vecs.rows(), vecs.cols());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        eigs_sorted(i) = eigs(indices[i]);
        vecs_sorted.col(i) = vecs.col(indices[i]);
    }

    double acc = 0.0;
    size_t n_discard = eigs_sorted.size();
    for (int i = static_cast<int>(eigs_sorted.size()) - 1; i >= 0; --i)
    {
        acc += std::abs(eigs_sorted(i));
        if (acc >= tol)
        {
            n_discard = eigs_sorted.size() - 1 - i;
            break;
        }
    }

    size_t n_vecs = eigs_sorted.size() - n_discard;
    if (max_vecs.has_value())
    {
        n_vecs = std::min(n_vecs, max_vecs.value());
    }

    Tensor<Complex, 4> one_body_tensors(n_vecs, 2, norb, norb);
    one_body_tensors.setZero();
    for (size_t k = 0; k < n_vecs; ++k)
    {
        MatrixXcd mat = MatrixXcd::Zero(norb, norb);
        for (int idx = 0; idx < n_pairs; ++idx)
        {
            mat(row[idx], col[idx]) = vecs_sorted(idx, k);
        }
        MatrixXcd Qp = quadrature(mat, 1);
        MatrixXcd Qm = quadrature(mat, -1);

        for (int i = 0; i < norb; ++i)
        {
            for (int j = 0; j < norb; ++j)
            {
                one_body_tensors(k, 0, i, j) = Qp(i, j);
                one_body_tensors(k, 1, i, j) = Qm(i, j);
            }
        }
    }

    Tensor<Complex, 4> orbital_rotations(n_vecs, 2, norb, norb);
    orbital_rotations.setZero();
    Tensor<double, 3> eigs_tensor(n_vecs, 2, norb);

    for (size_t k = 0; k < n_vecs; ++k)
    {
        for (int s = 0; s < 2; ++s)
        {
            MatrixXcd tensor_mat(norb, norb);
            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    tensor_mat(i, j) = one_body_tensors(k, s, i, j);
                }
            }

            SelfAdjointEigenSolver<MatrixXcd> es_tensor(tensor_mat);
            VectorXd evals = es_tensor.eigenvalues();
            MatrixXcd evecs = es_tensor.eigenvectors();

            for (int i = 0; i < norb; ++i)
            {
                eigs_tensor(k, s, i) = evals(i);
            }

            for (int i = 0; i < norb; ++i)
            {
                for (int j = 0; j < norb; ++j)
                {
                    orbital_rotations(k, s, i, j) = evecs(i, j);
                }
            }
        }
    }

    Tensor<double, 4> diag_coulomb(n_vecs, 2, norb, norb);
    for (size_t i = 0; i < n_vecs; ++i)
    {
        for (int s = 0; s < 2; ++s)
        {
            double sign_coeff = (s == 0) ? 1.0 : -1.0;
            for (int a = 0; a < norb; ++a)
            {
                for (int b = 0; b < norb; ++b)
                {
                    diag_coulomb(i, s, a, b) = sign_coeff * eigs_tensor(i, s, a) *
                                               eigs_tensor(i, s, b) * eigs_sorted(i).real();
                }
            }
        }
    }

    Tensor<Complex, 4> diag_coulomb_out(n_vecs, 2, norb, norb);
    diag_coulomb_out.setZero();
    for (size_t i = 0; i < n_vecs; ++i)
    {
        for (int s = 0; s < 2; ++s)
        {
            for (int a = 0; a < norb; ++a)
            {
                for (int b = 0; b < norb; ++b)
                {
                    diag_coulomb_out(i, s, a, b) = Complex(diag_coulomb(i, s, a, b), 0.0);
                }
            }
        }
    }

    return {diag_coulomb_out, orbital_rotations};
}

} // namespace linalg
} // namespace ffsim

#endif // DOUBLE_FACTORIZED_T2_HPP
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

#ifndef SBD_HELPER_HPP_
#define SBD_HELPER_HPP_

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <unistd.h>
#endif

#define USE_MATH_DEFINES
#include <cmath>

#include "mpi.h"
#include "sbd/sbd.h"

struct SBD {
    int task_comm_size = 1;
    int adet_comm_size = 1;
    int bdet_comm_size = 1;
    int h_comm_size = 1;

    int max_it = 1;
    int max_nb = 10;
    double eps = 1.0e-12;
    double max_time = 600.0;
    int init = 0;

    double threshold = 0.0;

    // This default value is for the Fe4S4
    double energy_target = -326.6;
    double energy_variance = 1.0;

    std::string adetfile = "AlphaDets.bin";
    std::string fcidumpfile = "";
};

SBD generate_sbd_data(int argc, char *argv[])
{
    SBD sbd;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--fcidump") {
            sbd.fcidumpfile = std::string(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--iteration") {
            sbd.max_it = std::atoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--block") {
            sbd.max_nb = std::atoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--tolerance") {
            sbd.eps = std::atof(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--max_time") {
            sbd.max_time = std::atof(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--adet_comm_size") {
            sbd.adet_comm_size = std::atoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--bdet_comm_size") {
            sbd.bdet_comm_size = std::atoi(argv[i + 1]);
            i++;
        }
        if (std::string(argv[i]) == "--task_comm_size") {
            sbd.task_comm_size = std::atoi(argv[i + 1]);
            i++;
        }
    }
    return sbd;
}

// energy, occupancy
std::tuple<double, std::vector<double>>
sbd_main(const MPI_Comm &comm, const SBD &sbd_data)
{

    double E = 0.0;

    int mpi_master = 0;
    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);
    int task_comm_size = sbd_data.task_comm_size;
    int adet_comm_size = sbd_data.adet_comm_size;
    int bdet_comm_size = sbd_data.bdet_comm_size;
    int base_comm_size;
    int L;
    int N;

    int max_it = sbd_data.max_it;
    int max_nb = sbd_data.max_nb;
    double eps = sbd_data.eps;
    double max_time = sbd_data.max_time;
    int init = sbd_data.init;

    double energy_target = sbd_data.energy_target;
    double energy_variance = sbd_data.energy_variance;

    size_t bit_length = SBD_BIT_LENGTH;
    std::string adetfile = sbd_data.adetfile;
    std::string fcidumpfile = sbd_data.fcidumpfile;

    base_comm_size = adet_comm_size * bdet_comm_size * task_comm_size;
    int h_comm_size = mpi_size / base_comm_size;

    if (mpi_size != base_comm_size * h_comm_size) {
        throw std::invalid_argument("communicator size is not appropriate");
    }

    /**
       Loading problem (fcidump)
     */

    sbd::FCIDump fcidump;
    if (mpi_rank == 0) {
        fcidump = sbd::LoadFCIDump(fcidumpfile);
    }
    sbd::MpiBcast(fcidump, 0, comm);
    double I0;
    sbd::oneInt<double> I1;
    sbd::twoInt<double> I2;
    sbd::SetupIntegrals(fcidump, L, N, I0, I1, I2);

    /**
       Preparation of dets
     */

    std::vector<std::vector<size_t>> adet;
    std::vector<std::vector<size_t>> bdet;

    if (mpi_rank == 0) {
        adet = sbd::DecodeAlphaDets(adetfile, L);
        sbd::change_bitlength(1, adet, bit_length);
        sbd::sort_bitarray(adet);
    }

    sbd::MpiBcast(adet, 0, comm);
    bdet = adet;

    /**
       Setup helpers
     */
    std::vector<sbd::TaskHelpers> helper;
    std::vector<std::vector<size_t>> sharedMemory;
    MPI_Comm h_comm;
    MPI_Comm b_comm;
    MPI_Comm t_comm;
    sbd::TaskCommunicator(
        comm, h_comm_size, adet_comm_size, bdet_comm_size, task_comm_size, h_comm,
        b_comm, t_comm
    );

    sbd::MakeHelpers(
        adet, bdet, bit_length, L, helper, sharedMemory, h_comm, b_comm, t_comm,
        adet_comm_size, bdet_comm_size
    );
    sbd::RemakeHelpers(
        adet, bdet, bit_length, L, helper, sharedMemory, h_comm, b_comm, t_comm,
        adet_comm_size, bdet_comm_size
    );

    int mpi_rank_h;
    MPI_Comm_rank(h_comm, &mpi_rank_h);
    int mpi_rank_b;
    MPI_Comm_rank(b_comm, &mpi_rank_b);
    int mpi_rank_t;
    MPI_Comm_rank(t_comm, &mpi_rank_t);
    int mpi_size_t;
    MPI_Comm_size(t_comm, &mpi_size_t);
    int mpi_size_b;
    MPI_Comm_size(b_comm, &mpi_size_b);
    int mpi_size_h;
    MPI_Comm_size(h_comm, &mpi_size_h);

    /**
       Initialize/Load wave function
     */
    std::vector<double> W;
    sbd::BasisInitVector(
        W, adet, bdet, adet_comm_size, bdet_comm_size, h_comm, b_comm, t_comm, init
    );
    /**
       Diagonalization
     */
    std::vector<double> hii;
    auto time_start_diag = std::chrono::high_resolution_clock::now();
    sbd::makeQChamDiagTerms(
        adet, bdet, bit_length, L, helper, I0, I1, I2, hii, h_comm, b_comm, t_comm
    );
    sbd::Davidson(
        hii, W, adet, bdet, bit_length, static_cast<size_t>(L), adet_comm_size,
        bdet_comm_size, helper, I0, I1, I2, h_comm, b_comm, t_comm, max_it, max_nb, eps,
        max_time
    );
    auto time_end_diag = std::chrono::high_resolution_clock::now();
    auto elapsed_diag_count = std::chrono::duration_cast<std::chrono::microseconds>(
                                  time_end_diag - time_start_diag
    )
                                  .count();
    double elapsed_diag = 0.000001 * static_cast<double>(elapsed_diag_count);
    if (mpi_rank == 0)
        std::cout << " Elapsed time for diagonalization " << elapsed_diag << " (sec) "
                  << std::endl;

    /**
         Evaluation of Hamiltonian expectation value
    */

    std::vector<double> C(W.size(), 0.0);

    sbd::mult(
        hii, W, C, adet, bdet, bit_length, static_cast<size_t>(L), adet_comm_size,
        bdet_comm_size, helper, I0, I1, I2, h_comm, b_comm, t_comm
    );

    sbd::InnerProduct(W, C, E, b_comm);

    if (energy_target != 0.0 && std::abs(E - energy_target) > energy_variance) {
        E = 0.0;
    }
    if (mpi_rank == 0) {
        std::cout.precision(16);
        std::cout << " Energy = " << E << std::endl;
    }

    /**
       Evaluation of single-particle occupation density
     */
    int p_size = mpi_size_t * mpi_size_h;
    int p_rank = mpi_rank_h * mpi_size_t + mpi_rank_t;
    size_t o_start = 0;
    size_t o_end = L;
    sbd::get_mpi_range(p_size, p_rank, o_start, o_end);
    size_t o_size = o_end - o_start;
    std::vector<int> oIdx(o_size);
    std::iota(oIdx.begin(), oIdx.end(), o_start);
    std::vector<double> res_density;
    sbd::OccupationDensity(
        oIdx, W, adet, bdet, bit_length, adet_comm_size, bdet_comm_size, b_comm,
        res_density
    );
    std::vector<double> density_rank(static_cast<size_t>(2 * L), 0.0);
    std::vector<double> density_group(static_cast<size_t>(2 * L), 0.0);
    std::vector<double> density(static_cast<size_t>(2 * L), 0.0);
    for (size_t io = o_start; io < o_end; io++) {
        density_rank[2 * io] = res_density[2 * (io - o_start)];
        density_rank[2 * io + 1] = res_density[2 * (io - o_start) + 1];
    }
    MPI_Allreduce(
        density_rank.data(), density_group.data(), 2 * L, MPI_DOUBLE, MPI_SUM, t_comm
    );
    MPI_Allreduce(
        density_group.data(), density.data(), 2 * L, MPI_DOUBLE, MPI_SUM, h_comm
    );

    FreeHelpers(helper);
    return {E, density};
}

#endif

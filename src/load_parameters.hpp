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

#include <complex>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

void load_initial_parameters(const std::string& filepath, uint64_t& norb,
                             std::pair<uint64_t, uint64_t>& nelec,
                             std::vector<std::pair<uint64_t, uint64_t>>& alpha_alpha_indices,
                             std::vector<std::pair<uint64_t, uint64_t>>& alpha_beta_indices,
                             std::vector<double>& init_params)
{

    std::ifstream i(filepath);
    if (!i.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    nlohmann::json input;
    i >> input;

    norb = input["norb"].get<uint64_t>();

    const auto& nelec_array = input["nelec"];
    if (!nelec_array.is_array() || nelec_array.size() != 2)
    {
        throw std::runtime_error("'nelec' must be an array of two integers.");
    }
    nelec.first = nelec_array[0].get<uint64_t>();
    nelec.second = nelec_array[1].get<uint64_t>();

    alpha_alpha_indices.clear();
    const auto& aa_indices = input["alpha_alpha_indices"];
    for (const auto& pair : aa_indices)
    {
        alpha_alpha_indices.emplace_back(pair[0].get<uint64_t>(), pair[1].get<uint64_t>());
    }

    alpha_beta_indices.clear();
    const auto& ab_indices = input["alpha_beta_indices"];
    for (const auto& pair : ab_indices)
    {
        alpha_beta_indices.emplace_back(pair[0].get<uint64_t>(), pair[1].get<uint64_t>());
    }

    init_params.clear();
    const auto& init_params_nodes = input["params"];
    for (const auto& param : init_params_nodes)
    {
        for (const auto& val : param)
        {
            init_params.push_back(val.get<double>());
        }
    }
}
# Demonstration of SQD using Qiskit CAPI

This demo shows how to post-process noisy quantum samples to approximate the ground state energy of the Fe₄S₄ cluster, using the [Sample-based Quantum Diagonalization (SQD) algorithm](https://www.science.org/doi/10.1126/sciadv.adu9991).


## Features

- HPC-ready implementation using modern C++17 and MPI.
- Integration with Qiskit C++, QRMI, and qiskit-addon-sqd-hpc.
- Support for hybrid quantum-classical workflows, including:
  - Quantum sampling on real backends.
  - Classical post-processing using the SQD.
  - Diagonalization using the SBD eigensolver.
- Designed for scalable execution on high-performance computing (HPC) clusters.


## Project Structure

```
├── data
│   ├── fcidump_Fe4S4_MO.txt         # Input file containing molecular orbital integrals
│   ├── initial_occupancies_fe4s4.json # JSON file defining initial orbital occupancies
│   └── parameters_fe4s4.json        # JSON file containing parameters for the LUCJ circuit
│
├── deps
│   ├── boost                        # Boost C++ dependency
│   ├── qiskit                       # Qiskit core library
│   ├── qiskit-addon-sqd-hpc         # Qiskit extension for SQD
│   ├── qiskit-cpp                   # C++ bindings for Qiskit
│   ├── qrmi                         # QRMI (quantum resource management interface)
│   └── sbd                          # SBD module
│
├── ffsim　　　　　　　　　　　　　　　　　# C++ header files for the ffsim library
│
├── src
│   ├── load_parameters.hpp          # Utility to load simulation parameters from JSON
│   ├── main.cpp                     # Main entry point of the executable
│   ├── sbd_helper.hpp               # Helper functions for SBD
│   └── sqd_helper.hpp               # Helper functions for SQD

```

## Requirements

To build this project, the following dependencies are required:

- Rust (latest stable recommended)
- C compiler with C++17 support
- CMake and Make (available as RPM packages on RHEL-compatible OS)
- Python ≥ 3.11


## Required Libraries

Please install the following libraries in your environment:

- OpenBLAS
- OpenMPI
- Eigen3

## Git Submodules

This repository uses several submodules. Initialize them before building:

```sh
git submodule update --init --recursive
```

Included submodules (under `deps/`):

- qiskit (https://github.com/Qiskit/qiskit)
- qiskit-cpp (https://github.com/Qiskit/qiskit-cpp)
- qrmi (https://github.com/qiskit-community/qrmi)
- sbd (https://github.com/r-ccs-cms/sbd)
- qiskit-addon-sqd-hpc (https://github.com/Qiskit/qiskit-addon-sqd-hpc)
- boost/dynamic_bitset (https://www.boost.org/library/latest/dynamic_bitset/) and its dependencies

## How to Build

### 1. Build Qiskit C Extension

```sh
cd deps/qiskit
make c
```

### 2. Build QRMI Service

This service enables access to quantum hardware from the Qiskit C++ sampler interface.

```sh
cd deps/qrmi
cargo build --release
```

### 3. Build demo

From the project root:

```sh
mkdir -p build
cd build
cmake ..
make
```

To test with pseudo-random shots instead of a quantum device:

```sh
cmake .. -DCMAKE_CXX_FLAGS="-DUSE_RANDOM_SHOTS=1"
make
```

### 4. Using IBM Quantum Hardware
To run simulations on IBM Quantum hardware via QRMI, set the following environment variables:

```sh
export QISKIT_IBM_TOKEN="your API key"
export QISKIT_IBM_INSTANCE="your CRN"
```

You can obtain these credentials from your IBM Quantum account.

## How to Run

### Single Process

```sh
./capi-demo \
  --fcidump ../data/fcidump_Fe4S4_MO.txt \
  -v \
  --tolerance 1.0e-3 \
  --max_time 600 \
  --recovery 1 \
  --number_of_samples 300 \
  --num_shots 1000 \
  --backend_name <your backend name>
```

### MPI Execution

```sh
mpirun -np 96 ./capi-demo \
  --fcidump ../data/fcidump_Fe4S4_MO.txt \
  -v \
  --tolerance 1.0e-3 \
  --max_time 600 \
  --recovery 1 \
  --number_of_samples 2000 \
  --num_shots 10000 \
  --backend_name <your backend name>
```

## Run Options
The following command-line options are available when running capi-demo. These control the behavior of the SQD simulation and quantum sampling:

### SQD Options
| Option                       | Description                                                        | Default Value |
|------------------------------|--------------------------------------------------------------------|---------------|
| --recovery <int>             | Number of configuration recovery iterations.                       | 3             |
| --number_of_batch <int>      | Number of batches per recovery iteration.                          | 1             |
| --number_of_samples <int>    | Number of samples per batch.                                      | 1000         |
| --backend_name <str>         | Name of the quantum backend to use (e.g., "ibm_torino").| ""            |
| --num_shots <int>           | Number of shots per quantum circuit execution.                    | 10000         |
| -v                           | Enable verbose logging to stdout/stderr.                           | false         |


### SBD Options
| Option                       | Description                                                        | Default Value |
|------------------------------|--------------------------------------------------------------------|---------------|
| --fcidump <path>             | Path to FCIDUMP file containing molecular integrals.               | ""            |
| --iteration <int>            | Maximum number of Davidson iterations.                             | 1             |
| --block <int>                | Maximum size of Litz vector space.                                 | 10            |
| --tolerance <float>          | Convergence tolerance for diagonalization.                        | 1.0e-12      |
| --max_time <float>          | Maximum allowed time (in seconds) for diagonalization.            | 600.0        |
| --adet_comm_size <int>      | Number of nodes used to split the alpha-determinants.            | 1             |
| --bdet_comm_size <int>      | Number of nodes used to split the beta-determinants.             | 1             |
| --task_comm_size <int>      | MPI communicator size for task-level parallelism.                 | 1             |


## Input Data
- The `fcidump_Fe4S4_MO.txt` file used in the examples is based on the Fe₄S₄ cluster model.
This data is from https://github.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters/blob/main/Fe2S2_and_Fe4S4/Fe4S4/fe4s4 .

- The `parameters_fe4s4.json` file contains the parameters for the LUCJ circuit, including the number of orbitals, number of electrons, and other relevant settings.
These parameters can also be obtained using `ffsim`.

- The values in the `initial_occupancies_fe4s4.json` file are the eigenvalues obtained by diagonalizing the contracted one-electron density matrix from the MP2 method.

## Deprecation policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.


## Contributing

The source code is available [on GitHub].
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).


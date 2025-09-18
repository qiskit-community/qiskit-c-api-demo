#!/usr/bin/env bash
set -euo pipefail

echo "Running ----------unit-test script----------"

source ci/scripts/github-login.sh

SOURCE_DIR="$(pwd)"
BUILD_DIR="${SOURCE_DIR}/build"

set -x

# Install KitWare ppa to obtain latest cmake (https://apt.kitware.com/)
apt-get update
apt-get install -y ca-certificates gpg wget
test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Update and install desired packages
apt-get update
apt-get install -y curl build-essential g++ cmake ninja-build libhdf5-dev libopenblas-dev python3.10-dev pkg-config libssl-dev libomp-dev libeigen3-dev openmpi-bin libopenmpi-dev

# Recent rust is essential to build Qiskit
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Clone submodules
git submodule update --init --recursive

# Build qiskit
cd $SOURCE_DIR/deps/qiskit
make c

# Build qrmi
cd $SOURCE_DIR/deps/qrmi
cargo build --release

cd $SOURCE_DIR

# Build
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja
ninja -C "$BUILD_DIR"

# Re-build with USE_RANDOM_SHOTS=1 and test the resulting binary
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja -DCMAKE_CXX_FLAGS="-DUSE_RANDOM_SHOTS=1"
ninja -C "$BUILD_DIR"
cd "$BUILD_DIR"
./capi-demo --fcidump ../data/fcidump_Fe4S4_MO.txt --tolerance 1.0e-3 --max_time 600 --recovery 1 --number_of_samples 300 --num_shots 1000

cd $SOURCE_DIR

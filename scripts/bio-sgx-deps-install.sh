(
mkdir local
mkdir deps
cd deps
apt-get source pkg-config libssl-dev protobuf-compiler
cd protobuf*
./configure --prefix=$(realpath ../../local/)
make
make install
)

curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default nightly
rustup target add x86_64-fortanix-unknown-sgx --toolchain nightly
PATH="$(realpath local/bin):$PATH" \
LD_LIBRARY_PATH="$(realpath local/lib):$LD_LIBRARY_PATH" \
cargo install fortanix-sgx-tools sgxs-tools

curl https://sh.rustup.rs -sSf | sh -s -- -y && \
rustup default nightly && \
rustup target add x86_64-fortanix-unknown-sgx --toolchain nightly && \

(
mkdir local && \
mkdir deps && \
cd deps && \
apt-get source pkg-config libssl-dev protobuf-compiler && \
cd protobuf* &&\
./configure --prefix=$(realpath ../../local/) &&\
make && \
make install
) && \

PATH="$(realpath local/bin):$PATH"
LD_LIBRARY_PATH="$(realpath local/lib):$LD_LIBRARY_PATH" cargo install fortanix-sgx-tools sgxs-tools

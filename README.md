# iDASH's Neural Network Evaluation in SGX
## Dependencies
- [Rust](https://www.rust-lang.org/tools/install)
- [Fortanix EDP](https://edp.fortanix.com/docs/installation/guide/)
### Installing dependencies for bio-sgx
From the main directory:
```bash
source scripts/bio-sgx-deps-install.sh
```
## Build
To build in the SGX Enclave mode:
```bash
make
```
To build in the non-SGX Enclave mode:
```bash
make ENCLAVE_MODE=0
```
## Benchmark
1. Build the program:
```bash
make
```
2. Preprocess the weight file:
```bash
make preprocess WEIGHT_SRC_FILE=<dir_to_hdf5_file>
```
3. Run a local benchmark:
```bash
time make run INPUT_FILE_DIR=<input_file_dir>
```

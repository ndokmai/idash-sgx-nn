# idash-sgx-nn-enclave
## How to build the enclave
1. [Install Fortanix EDP.](https://edp.fortanix.com/docs/installation/guide/)

2. Build the enclave:
```bash
cargo build --release --target=x86_64-fortanix-unknown-sgx
```

# idash-sgx-nn-enclave
## How to run the enclave
1. [Install Fortanix EDP.](https://edp.fortanix.com/docs/installation/guide/)

2. Start the enclave:
```bash
cargo run --release --target=x86_64-fortanix-unknown-sgx &
```

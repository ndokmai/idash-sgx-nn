cargo build --release --target=x86_64-fortanix-unknown-sgx
ftxsgx-elf2sgxs target/x86_64-fortanix-unknown-sgx/release/idash-sgx-nn --heap-size 0x5000000 --stack-size 0x4000 -o app.sgxs -t 5

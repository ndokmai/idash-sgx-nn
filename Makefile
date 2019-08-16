FTXSGX_RUNNER:=ftxsgx-runner
FTXSGX_EFL2SGXS:=ftxsgx-elf2sgxs
ENCLAVE_CARGO:=cargo
ENCLAVE_CARGO_PARAMS:=build --release

ENCLAVE_FILE:=build/enclave.sgxs
LAUNCHER=launcher/target/release/launcher
WEIGHT_SRC_FILE?=data/weight.hdf5
INPUT_FILE_1?=data/GSE25066-Normal-50-sgx.txt
INPUT_FILE_2?=data/GSE25066-Tumor-50-sgx.txt
HOST?=localhost
PORT?=1237

PREPROCESS_WEIGHT_SCRIPT:=scripts/preprocess.py
WEIGHT_TAR_FILE:=build/weight.bin
CONFIG_FILE?=config
ENCLAVE_MODE?=1
DEBUG?=0

TARGET=$(LAUNCHER)
LAUNCH=$(LAUNCHER) launcher $(PORT) $(WEIGHT_TAR_FILE)
CLIENT=$(LAUNCHER) client $(HOST):$(PORT) $(INPUT_FILE_1) $(INPUT_FILE_2)

ifeq ($(DEBUG), 1)
    ENCLAVE_CARGO_PARAMS+=--features debug 
    LAUNCHER_CARGO_PARAMS+=--features debug 
endif

ifeq ($(ENCLAVE_MODE), 1)
    TARGET+=$(ENCLAVE_FILE)
    ENCLAVE_EFL:=enclave/target/x86_64-fortanix-unknown-sgx/release/enclave
    ENCLAVE_CARGO:=RUSTFLAGS="-C target-feature=+aes,+pclmul" $(ENCLAVE_CARGO) +nightly
    ENCLAVE_CARGO_PARAMS+=--target=x86_64-fortanix-unknown-sgx
    include $(CONFIG_FILE) 
    LAUNCH+=$(FTXSGX_RUNNER) $(ENCLAVE_FILE)
else
    ENCLAVE_EFL:=enclave/target/release/enclave
    TARGET+=$(ENCLAVE_EFL)
    LAUNCH+=$(ENCLAVE_EFL)
endif

.PHONY: build run clean $(ENCLAVE_EFL) $(LAUNCHER)

build: $(TARGET)
	cd enclave; \
	$(ENCLAVE_CARGO) $(ENCLAVE_CARGO_PARAMS)

run: build $(INPUT_FILE_1) $(INPUT_FILE_2) $(WEIGHT_TAR_FILE)
	$(CLIENT) &
	$(LAUNCH)

$(WEIGHT_TAR_FILE): $(WEIGHT_SRC_FILE)
	python $(PREPROCESS_WEIGHT_SCRIPT) $(WEIGHT_SRC_FILE) $(WEIGHT_TAR_FILE)

$(ENCLAVE_FILE): $(ENCLAVE_EFL) $(CONFIG_FILE)
	mkdir -p build
	$(FTXSGX_EFL2SGXS) $(ENCLAVE_EFL) --heap-size $(HEAP) --stack-size $(STACK) \
	    -t $(THREADS) -o $(ENCLAVE_FILE) 

$(ENCLAVE_EFL):
	cd enclave; \
	$(ENCLAVE_CARGO) $(ENCLAVE_CARGO_PARAMS) 

$(LAUNCHER):
	cd launcher; \
	cargo build --release $(LAUNCHER_CARGO_PARAMS)

clean:
	rm -rf build
	cd enclave; cargo clean
	cd launcher; cargo clean

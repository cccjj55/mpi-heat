MPICC ?= mpicc
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=c11
LDFLAGS ?= -lm
NP_LIST ?= 1 2 4 8

BIN := build/heat_mpi

.PHONY: all build bench clean

all: build

build: $(BIN)

$(BIN): src/heat_mpi.c
	@mkdir -p build results
	$(MPICC) $(CFLAGS) $< -o $(BIN) $(LDFLAGS)

bench: build
	NP_LIST="$(NP_LIST)" ./scripts/scaling.sh

clean:
	rm -rf build results

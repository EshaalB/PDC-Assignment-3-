CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -Wno-unknown-pragmas
OMPFLAG = -fopenmp
L      ?= 100000000
K      ?= 12
THREADS ?= 8

all: serial parallel

# serial is built with the same -O3 flags (the OpenMP pragma in the untimed
# generator is ignored without -fopenmp)
serial: serial.c kmer_common.h
	$(CC) $(CFLAGS) -o $@ serial.c

parallel: parallel.c kmer_common.h
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ parallel.c

# every version is checked bin-by-bin against a serial reference inside ./parallel
verify: all
	./serial $(L) $(K) 0 42 1
	OMP_NUM_THREADS=$(THREADS) ./parallel -L $(L) -k $(K) -r 1 -v ABCU

clean:
	rm -f serial parallel

.PHONY: all clean verify

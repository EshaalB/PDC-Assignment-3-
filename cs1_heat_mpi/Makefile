CC     = gcc
MPICC  = mpicc
CFLAGS = -O3 -Wall -Wextra
LDLIBS = -lm
NP    ?= 4
N     ?= 1024
T     ?= 500
K     ?= 50

all: serial parallel

serial: serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

parallel: parallel.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

# compare every MPI variant (1-D/2-D x blocking/non-blocking) with the serial grid
verify: all
	./serial $(N) $(T) $(K) ref_$(N).bin
	for d in 1 2; do for m in 0 1; do \
	  mpirun --oversubscribe -np $(NP) ./parallel -n $(N) -t $(T) -k $(K) -d $$d -m $$m -r ref_$(N).bin; \
	done; done
	rm -f ref_$(N).bin

clean:
	rm -f serial parallel *.bin

.PHONY: all clean verify

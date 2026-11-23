CC     = gcc
MPICC  = mpicc
CFLAGS = -O3 -Wall -Wextra
LDLIBS = -lm
NP    ?= 4
N     ?= 100000000

all: serial parallel pingpong

serial: serial.c sort_common.h
	$(CC) $(CFLAGS) -o $@ serial.c $(LDLIBS)

parallel: parallel.c sort_common.h
	$(MPICC) $(CFLAGS) -o $@ parallel.c $(LDLIBS)

pingpong: pingpong.c
	$(MPICC) $(CFLAGS) -o $@ pingpong.c

# sum/hash printed by serial and parallel must be identical for each distribution
verify: all
	for d in 0 1 2; do \
	  ./serial $(N) $$d; \
	  mpirun --oversubscribe -np $(NP) ./parallel -n $(N) -d $$d -c; \
	done

clean:
	rm -f serial parallel pingpong

.PHONY: all clean verify

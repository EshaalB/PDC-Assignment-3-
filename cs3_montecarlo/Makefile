CC       = gcc
CFLAGS   = -O3 -march=native -Wall -Wextra
OMPFLAGS = -fopenmp
LDLIBS   = -lm

all: serial parallel

serial: serial.c mc_common.h
	$(CC) $(CFLAGS) serial.c -o serial $(LDLIBS)

parallel: parallel.c mc_common.h
	$(CC) $(CFLAGS) $(OMPFLAGS) parallel.c -o parallel $(LDLIBS)

clean:
	rm -f serial parallel

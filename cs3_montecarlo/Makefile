# TODO (Member B): switch CC/flags if MPI is chosen (mpicc)
CC = gcc
CFLAGS = -O3 -Wall -fopenmp
LDLIBS = -lm

all: serial parallel

serial: serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
parallel: parallel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f serial parallel
.PHONY: all clean

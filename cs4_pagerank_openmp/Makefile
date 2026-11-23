CC       = gcc
CFLAGS   = -O3 -march=native -Wall -Wextra
OMPFLAGS = -fopenmp
LDLIBS   = -lm
HDRS     = pr_common.h

all: serial parallel

serial: serial.c $(HDRS)
	$(CC) $(CFLAGS) $(OMPFLAGS) serial.c -o serial $(LDLIBS)

parallel: parallel.c $(HDRS)
	$(CC) $(CFLAGS) $(OMPFLAGS) parallel.c -o parallel $(LDLIBS)

clean:
	rm -f serial parallel

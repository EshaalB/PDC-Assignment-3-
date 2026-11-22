CC = gcc
CFLAGS = -O3 -Wall

all: serial parallel

serial: serial.c
	$(CC) $(CFLAGS) -o $@ $<
parallel: parallel.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f serial parallel
.PHONY: all clean

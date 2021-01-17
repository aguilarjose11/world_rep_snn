IDIR=.
CC=gcc
CXXFLAGS=-g -std=c++11 -Wall -pedantic
CFLAGS=-g -I$(IDIR) -DARMA_DONT_USE_WRAPPER -lopenblas -llapack -larmadillo -lstdc++
ODIR=.
LDIR=.
LIBS=-lm

_DEPS = ou_snn.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_LIBSNN = libsnnlfisrm.a
LIBSNN = $(patsubst %,./libsnnlfisrm/lib/%,$(_LIBSNN))



all: snn_main test_snn

snn_main.o: snn_main.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

snn_main: snn_main.o $(LIBSNN)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

test_snn.o: test_snn.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

test_snn: test_snn.o $(LIBSNN)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ snn_main test_snn
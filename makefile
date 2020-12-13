IDIR=.
CC=gcc
CXXFLAGS=-g -std=c++11 -Wall -pedantic
CFLAGS=-g -I$(IDIR) -DARMA_DONT_USE_WRAPPER -lopenblas -llapack -larmadillo -lstdc++
ODIR=.
LDIR=.
LIBS=-lm

_DEPS = ou_snn.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = ou_snn.o snn_main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: snn_main

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

snn_main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
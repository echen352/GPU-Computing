# The variable CC specifies which compiler will be used.
# (because different unix systems may use different compilers)
CC=g++

# The variable CFLAGS specifies compiler options
#   -c :    Only compile (don't link)
#   -Wall:  Enable all warnings about lazy / dangerous C programming 
#  You can add additional options on this same line..
#  WARNING: NEVER REMOVE THE -c FLAG, it is essential to proper operation
CFLAGS=-c -O3 -fopenacc

# All of the .h header files to use as dependencies
HEADERS=convolve.h gradient.h gaussian.h hysteresis.h nonmaxSuppresion.h pgm.h

# All of the object files to produce as intermediary work
OBJECTS=main.o

# The final program to build
EXECUTABLE=edges

# --------------------------------------------

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXECUTABLE)

%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf *.o $(EXECUTABLE) *.pgm *.o* *.e* #remove exec, pgms, and slurm error and output files:

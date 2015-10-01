#Define compiler & flags
GCC=g++
NCC=/usr/local/cuda/bin/nvcc
CFLAGS=-Wall -O2 -arch i386
NCCFLAGS=-O2 -Xlinker -pie -arch sm_11

#Define build configurations
SPHERES1=-DSP1 -DSIM -DDIS1 #-DPRINT
SPHERES2=-DSP2 -DSIM -DDIS1 #-DPRINT
SPHERES3=-DSP3 -DSIM -DDIS1 #-DPRINT

#REQUIRED=src/init.cpp src/print.cpp src/utils.cu
#REQUIRED=src/init.o src/print.o src/utils.o src/stopWatch/stopWatch.o

all: spheres1 spheres2 spheres3

test: clean spheres1 spheres2 spheres3

## Target: spheres1
spheres1: #$(REQUIRED)
	$(NCC) $(NCCFLAGS) $(SPHERES1) src/main.cu src/stopWatch/stopWatch.cpp -o spheres1

## Target: spheres2
spheres2: #$(REQUIRED)
	$(NCC) $(NCCFLAGS) $(SPHERES2) src/main.cu src/stopWatch/stopWatch.cpp -o spheres2

## Target: spheres3
spheres3: #$(REQUIRED)
	$(NCC) $(NCCFLAGS) $(SPHERES3) src/main.cu src/stopWatch/stopWatch.cpp -o spheres3

## Target: stopWatch.o
src/stopWatch/stopWatch.o:
	$(NCC) $(NCCFLAGS) -c src/stopWatch/stopWatch.cpp -o src/stopWatch/stopWatch.o

src/init.o:
	$(NCC) $(NCCFLAGS) -c src/init.cpp -o src/init.o

src/print.o:
	$(NCC) $(NCCFLAGS) -c src/print.cpp -o src/print.o

src/utils.o:
	$(NCC) $(NCCFLAGS) -c src/utils.cu -o src/utils.o

#### Clean target deletes all generated files ####
clean:
	rm spheres1 spheres2 spheres3

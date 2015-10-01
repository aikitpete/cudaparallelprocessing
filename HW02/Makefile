#Define compiler & flags
CC=g++
CFLAGS=-Wall -O2

#Define build configurations
SPHERES1=-DSP1 -DDIS1
SPHERES2=-DSP2 -DDIS1
SPHERES3=-DSP3 -DDIS1
DISTANCES1=-DSP1 -DDIS1
DISTANCES2=-DSP1 -DDIS2
DISTANCES3=-DSP1 -DDIS3
DISTANCES4=-DSP1 -DDIS4

all: spheres1 spheres2 spheres3 distances1 distances2 distances3 distances4

#Defines rules for compiling the project
spheres1: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(SPHERES1) main.cpp stopWatch.o -o ../HW06/spheres1old

spheres2: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(SPHERES2) main.cpp stopWatch.o -o ../HW06/spheres2old

spheres3: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(SPHERES3) main.cpp stopWatch.o -o ../HW06/spheres3old

distances1: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(DISTANCES1) main.cpp stopWatch.o -o distances1

distances2: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(DISTANCES2) main.cpp stopWatch.o -o distances2

distances3: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(DISTANCES3) main.cpp stopWatch.o -o distances3

distances4: main.cpp stopWatch.o
	$(CC) $(CFLAGS) $(DISTANCES4) main.cpp stopWatch.o -o distances4

stopWatch.o: stopWatch.cpp
	$(CC) $(CFLAGS) stopWatch.cpp -c $<

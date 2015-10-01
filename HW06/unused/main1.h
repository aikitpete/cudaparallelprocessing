/* 
 * File:   main.cpp
 * Author: peter
 *
 * Created on March 25, 2012, 1:36 AM
 */

#ifndef MAIN_H
#define MAIN_H

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <math.h>
//#include <cassert.h>
//#include "stopWatch.h"
#include "mpi.h"

using namespace std;

int n;
int d;
int seed;

double t1, t2, t3, time1, time2;
double minVal, maxVal, avgVal;
int collision;

bool init(char** argv);
double** generateCoordinates();
double** generateSpheres();
double* generateDimensions();
double* generateRadiuses();
double& getDistance(int i, int j, double* distances);
void computeDistances(double** xyz, double* distances);
void computeDistances(double* x, double* y, double* z, double* distances);
void detectCollisions(double* r, double* distances);
void detectCollisions(double** xyzr, double* distances);
void printArray(double** array, int m, int n);
void printArray(double* array, int m, int n);
void release(double** array);
void release(double* array);

#endif

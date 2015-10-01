/* 
 * File:   main.cpp
 * Author: peter
 *
 * Created on March 25, 2012, 1:36 AM
 */

#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std;

DT& getDistance(DT* distances, int n, int i, int j) {
#ifdef DIS1
    return distances[i * n + j];
#else
    throw "Missing DIS";
#endif
}

DT *genEmptyMatrix(const int n) {
    DT *ptr;
    //cudaMallocHost(&ptr, n * sizeof (DT));
    ptr = new DT[n];
    return ptr;
}

unsigned int *genEmptyColMatrix(const int n) {
    unsigned int *ptr;
    //cudaMallocHost(&ptr, n * sizeof (DT));
    ptr = new unsigned int[n];
    return ptr;
}

DT *genMatrix(const int n) {
    DT *ptr;
    ptr = genEmptyMatrix(n);
    for (int i = 0; i < n; i++) {
        ptr[i] = ((DT) rand()) / RAND_MAX;
    }
    return ptr;
}

DT **genMatrix(const int n, const int m) {
    DT **ptr;
    //ptr = (DT**)malloc(n*sizeof(DT*));
    //cudaMallocHost((void**)&ptr, n*sizeof(DT*));
    ptr = new DT*[n];
    for (int i = 0; i < n; i++) {
        ptr[i] = genEmptyMatrix(m);
        for (int j = 0; j < m; j++) {
            ptr[i][j] = ((DT) rand()) / RAND_MAX;
        }
    }
    return ptr;
}

void cudaRelease(int n, DT** array) {
    for (int i = 0; i < n; i++) {
        cudaFree(array[i]);
    }
    cudaFree(array);

}

void cudaRelease(DT* array) {
    cudaFree(array);
}

void cudaRelease(unsigned int* array) {
    cudaFree(array);
}

void release(int n, DT** array) {
    for (int i = 0; i < n; i++) {
        free(array[i]);
    }
    free(array);

}

void release(DT* array) {
    free(array);
}

#endif

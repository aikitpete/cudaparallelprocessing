/* 
 * File:   main.cpp
 * Author: peter
 *
 * Created on March 25, 2012, 1:36 AM
 */


#ifndef MAIN_H
#define MAIN_H
#define DT float

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <math.h>
#include "stopWatch/stopWatch.h"
#include <cuda.h>
#include "stdio.h"

#include "init.cpp"
#include "utils.cpp"
#include "print.cpp"

using namespace std;

int n;
int d;
int seed;

//#define printf(f, ...) ((void)(f, __VA_ARGS__),0)

const int threadsA = 16;
const int threadsB = 16;

#ifdef DIS1
DT* distances;
#endif
unsigned int* collisions;

bool init(char** argv) {

    //Set values from arguments
    n = atoi(argv[1]);
    d = atof(argv[2]);
    seed = atoi(argv[3]);

    cout << setprecision(4);

#ifdef PRINT
    cout << "Value of N:" << n << endl;
    cout << "Value of D:" << d << endl;
    cout << "Value of seed:" << seed << endl;
    cout << endl;
#endif

    srand(seed);

    return true;
}

DT** generateCoordinates() {

    //Initialize array to store spheres
    DT **ret = genMatrix(n, 3);


    // Each sphere has attributes
    // ID - row number
    // x,y,z coordinates - in range [-D:D, -D:D, -D:D]
    // radius - in range [1:D]

    for (int i = 0; i < n; i++) {
        ret[i][0] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i][1] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i][2] = (rand() / DT(RAND_MAX))*2 * d - d;

    }

    return ret;
}

DT* generateSimCoordinates(size_t width) {

    //Initialize array to store spheres
    DT *ret = genEmptyMatrix(n * width);


    // Each sphere has attributes
    // ID - row number
    // x,y,z coordinates - in range [-D:D, -D:D, -D:D]
    // radius - in range [1:D]

    for (int i = 0; i < n; i++) {
        ret[i * width + 0] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i * width + 1] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i * width + 2] = (rand() / DT(RAND_MAX))*2 * d - d;

    }

    if (width == 4) {
        for (int i = 0; i < n; i++) {
            ret[i * width + 3] = (rand() / DT(RAND_MAX)) * d;
        }
    }

    return ret;
}

DT** generateSpheres(size_t width) {

    //Initialize array to store spheres
    DT **ret = genMatrix(n, 4);

    //
    // Each sphere has attributes
    // ID - row number
    // x,y,z coordinates - in range [-D:D, -D:D, -D:D]
    // radius - in range [1:D]
    //
    for (int i = 0; i < n; i++) {
        ret[i][0] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i][1] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i][2] = (rand() / DT(RAND_MAX))*2 * d - d;
    }

    if (width == 4) {
        for (int i = 0; i < n; i++) {
            ret[i][3] = (rand() / DT(RAND_MAX)) * d;
        }
    }
    return ret;
}

DT* generateSimSpheres(size_t width) {

    //Initialize array to store spheres
    DT *ret = genEmptyMatrix(n * width);

    //
    // Each sphere has attributes
    // ID - row number
    // x,y,z coordinates - in range [-D:D, -D:D, -D:D]
    // radius - in range [1:D]
    //
    for (int i = 0; i < n; i++) {
        ret[i * width + 0] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i * width + 1] = (rand() / DT(RAND_MAX))*2 * d - d;
        ret[i * width + 2] = (rand() / DT(RAND_MAX))*2 * d - d;
    }

    if (width == 4) {
        for (int i = 0; i < n; i++) {
            ret[i * width + 3] = (rand() / DT(RAND_MAX)) * d;
        }
    }

    return ret;
}

DT* generateDimensions() {

    DT* ret = genEmptyMatrix(n);
    for (int i = 0; i < n; i++) {

        ret[i] = (rand() / DT(RAND_MAX))*2 * d - d;

    }
    return ret;
}

DT* generateRadiuses() {

    DT* r = genEmptyMatrix(n);
    for (int i = 0; i < n; i++) {

        r[i] = (rand() / DT(RAND_MAX)) * d;

    }
    return r;
}

void initDistances() {
#ifdef DIS1
    //size_t pitch;
    //size_t bytes = n * sizeof (DT);
    //cudaMallocPitch((void**) & distances, &pitch, bytes, n);
    size_t bytes = n * n * sizeof (DT);
    cudaMalloc((void**) & distances, bytes);
#endif
}

void initCollisions() {
#ifdef DIS1
    //size_t pitch;
    //size_t bytes = threadsA * sizeof (unsigned int);
    //cudaMallocPitch((void**) & collisions, &pitch, bytes, threadsB);
    size_t bytes = threadsA * threadsB * sizeof (unsigned int);
    cudaMalloc((void**) & collisions, bytes);
    cudaMemset(collisions, 0, bytes);
#endif
}

__global__
void computeDistancesKernel(const size_t n, const size_t width, DT* distances,
        DT* tmpXYZ, const int pitch) {

    int xd;
    int yd;
    int zd;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    //Compute distances
    if (i < n) {
        if (j > i + 1 && j < n) {
            //for (int j = i + 1; j < n; j++) {

            xd = fabs(tmpXYZ[i * width + 0] - tmpXYZ[j * width + 0]);
            yd = fabs(tmpXYZ[i * width + 1] - tmpXYZ[j * width + 1]);
            zd = fabs(tmpXYZ[i * width + 2] - tmpXYZ[j * width + 2]);
            distances[i * n + j] =
                    sqrt((double) (xd * xd + yd * yd + zd * zd));

        }
#ifdef PRINT
        else if (j < i + 1) {
            distances[i * n + j] =
                    -1;
        }
#endif
    }

}

__global__
void computeDistancesKernel(const size_t n, DT* distances, DT* tmpX, DT* tmpY,
        DT* tmpZ) {

    int xd;
    int yd;
    int zd;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    //Compute distances
    if (i < n) {
        if (j > i + 1 && j < n) {
            //for (int j = i + 1; j < n; j++) {

            xd = fabs(tmpX[i] - tmpX[j]);
            yd = fabs(tmpY[i] - tmpY[j]);
            zd = fabs(tmpZ[i] - tmpZ[j]);
            distances[i * n + j] =
                    sqrt((double) (xd * xd + yd * yd + zd * zd));

        }
#ifdef PRINT
        else if (j < i + 1) {
            distances[i * n + j] =
                    -1;
        }
#endif
    }

}

__global__
void detectCollisionsKernel(const size_t n, DT* distances, unsigned int* collisions,
        DT* tmpR) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n) {
        if (j > i + 1 && j < n) {
            //for (int j = i + 1; j < n; j++) {

            if ((tmpR[i] + tmpR[j]) >= distances[i * n + j]) {

                collisions[i * threadsA + j]++;
            }
        }
    }

}

__global__
void detectCollisionsKernel(const size_t n, const size_t width, DT* distances,
        unsigned int* collisions, DT* tmpXYZR, const int pitch) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    DT distance;

    if (i < n) {
        if (j > i + 1 && j < n) {
            //for (int j = i + 1; j < n; j++) {
            distance = distances[i * n + j];

            if (tmpXYZR[i * width + 3] + tmpXYZR[j * width + 3] >= distance) {

                collisions[i * threadsA + j]++;
            }
        }
    }

}

//Used for testing CUDA

/*
__global__
void addKernel(size_t s, DT *a, DT *ans) {

    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < s && j < s) {
        for (size_t k = 0; k < s; k++) {
            ans[i * s + j] += a[i * s + k] * a[k * s + j];

        }
    }
}

void addMatricesGPU(const size_t j, const size_t k, DT **mat, DT *out) {

    DT *tmp, *ans;
    size_t bytes = k * k * sizeof (DT);
    cudaMalloc((void**) & tmp, bytes);
    cudaMalloc((void**) & ans, bytes);

    cudaMemset(ans, 0, bytes);

    const int threads = 16;
    dim3 blocks(threads, threads);
    dim3 grids((k + threads - 1) / threads, (k + threads - 1) / threads);

    for (size_t i = 0; i < j; i++) {
        cudaMemcpy(tmp, mat[i], bytes, cudaMemcpyHostToDevice);
        addKernel << <grids, blocks >> >(k, tmp, ans);
    }

    cudaMemcpy(out, ans, bytes, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();

    cudaFree(tmp);
    cudaFree(ans);

}
 */

int main(int argc, char** argv) {

    //Used for testing CUDA
    /*
    srand(1);
    size_t n = 3;
    size_t s = 2;
    DT **mat = genMatrix(n, s * s);
    DT *out = genEmptyMatrix(s * s);

    printArray(mat, n, s * s);

    addMatricesGPU(n, s, mat, out);
    printArray(out, s*s, 1);
     */

    if (!init(argv)) {
        throw "Invalid arguments, arguments are: N, D, seed";
        //return 1;
    }

#ifdef PRINT
#ifdef SP1
    cout << "Spheres configuration: xyz[N][3],r[N]" << endl;
#elif SP2
    cout << "Spheres configuration: xyzr[N][4]" << endl;
#elif SP3
    cout << "Spheres configuration: x[N],y[N],z[N],r[N]" << endl;
#endif

#ifdef DIS1
    cout << "Distances configuration: simulated + row major" << endl;
#endif

    cout << endl;
#endif

    stopWatch* watch = new stopWatch;
    watch->start();

    //Generate spheres coordinates
#ifdef SP1
#ifdef SIM
    DT* xyz = generateSimCoordinates(3);
#else
    DT** xyz = generateCoordinates(3);
#endif
    DT* r = generateRadiuses();
#elif SP2
#ifdef SIM
    DT* xyzr = generateSimSpheres(4);
#else
    DT** xyzr = generateSpheres(4);
#endif
#elif SP3
    DT* x = generateDimensions();
    DT* y = generateDimensions();
    DT* z = generateDimensions();
    DT* r = generateRadiuses();
#endif

    //const int threadsA = 512;
    //const int threadsB = 1;
    //dim3 blocks(threadsA, threadsB);
    //dim3 grids((n + threads - 1) / threads, 1);

    dim3 blocks(threadsA, threadsB);
    dim3 grids((n + threadsA - 1) / threadsA, (n + threadsB - 1) / threadsB);

#ifdef SP1
    DT *tmpXYZ, *tmpR;
    size_t pitch;

    size_t bytes1 = 3 * sizeof (DT);
    size_t bytes2 = n * sizeof (DT);

    cudaMallocPitch((void**) & tmpXYZ, &pitch, bytes1, n);
    cudaMalloc((void**) & tmpR, bytes2);

    cudaMemcpy2D(tmpXYZ, pitch, xyz, bytes1, bytes1, n,
            cudaMemcpyHostToDevice);
    cudaMemcpy(tmpR, r, bytes2, cudaMemcpyHostToDevice);
#elif SP2
    DT *tmpXYZR;
    size_t pitch;

    size_t bytes = 4 * sizeof (DT);

    cudaMallocPitch((void**) & tmpXYZR, &pitch, bytes, n);

    cudaMemcpy2D(tmpXYZR, pitch, xyzr, bytes, bytes, n,
            cudaMemcpyHostToDevice);
#elif SP3
    DT *tmpX, *tmpY, *tmpZ, *tmpR;
    size_t bytes = n * sizeof (DT);

    cudaMalloc((void**) & tmpX, bytes);
    cudaMalloc((void**) & tmpY, bytes);
    cudaMalloc((void**) & tmpZ, bytes);
    cudaMalloc((void**) & tmpR, bytes);

    cudaMemcpy(tmpX, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(tmpY, y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(tmpZ, z, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(tmpR, r, bytes, cudaMemcpyHostToDevice);
#endif

    cudaThreadSynchronize();

    //Initialize array to store distances
    initDistances();
#ifdef SP1
    computeDistancesKernel << <grids, blocks >> >(n, 3, distances,
            tmpXYZ, pitch);
#elif SP2
    computeDistancesKernel << <grids, blocks >> >(n, 4, distances,
            tmpXYZR, pitch);
#elif SP3
    computeDistancesKernel << <grids, blocks >> >(n, distances, tmpX, tmpY,
            tmpZ);
#endif

    cudaThreadSynchronize();

    //Initialize array to store collisions
    initCollisions();
    //Detect collisions
#ifdef SP1
    detectCollisionsKernel << <grids, blocks >> >(n, distances, collisions,
            tmpR);
#elif SP2
    detectCollisionsKernel << <grids, blocks >> >(n, 4, distances,
            collisions, tmpXYZR, pitch);
#elif SP3
    detectCollisionsKernel << <grids, blocks >> >(n, distances, collisions,
            tmpR);
#endif

    cudaThreadSynchronize();

    watch->stop();

#ifdef PRINT
    cout << "Elapsed time: " << watch->elapsedTime() << endl;

    //Print spheres
    cout << "Spheres:" << endl;
#ifdef SP1
    printArray(xyz, 3, n);
    printArray(r, 1, n);
#elif SP2
    printArray(xyzr, 4, n);
#elif SP3
    printArray(x, 1, n);
    printArray(y, 1, n);
    printArray(z, 1, n);
    printArray(r, 1, n);
#endif


    cudaThreadSynchronize();

    //Print distances
    DT* distancesLocal;
#endif

    int size = n * n * sizeof (float);

#ifdef PRINT
    distancesLocal = genEmptyMatrix(n * n);
    cudaMemcpy(distancesLocal, distances, size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cout << "Distances:" << endl;
    printArray(distancesLocal, n, n);
#endif

    unsigned int* collisionsLocal;
    collisionsLocal = genEmptyColMatrix(threadsA * threadsB);
    size = threadsA * threadsB * sizeof (unsigned int);
    cudaMemcpy(collisionsLocal, collisions, size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

#ifdef PRINT
    cout << "Collisions:" << endl;
    printArray(collisionsLocal, 1, 512);

    printf("LAST ERROR: %d\n", cudaGetLastError());
#endif

    int total = 0;
    for (int i = 0; i < threadsA * threadsB; i++) {
        total = total + collisionsLocal[i];
    }

    //Release
#ifdef SP1
#ifdef SIM
    release(xyz);
#else
    release(n, xyz);
#endif
    release(r);
#elif SP2
#ifdef SIM
    release(xyzr);
#else
    release(n, xyzr);
#endif
#elif SP3
    release(x);
    release(y);
    release(z);
    release(r);
#endif

#ifdef DIS1
    cudaRelease(distances);
    cudaRelease(collisions);
#endif

    cout << watch->elapsedTime() << ", " << n << ", " << total << endl;

    return 0;

}

#endif

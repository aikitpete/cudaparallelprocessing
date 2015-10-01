/* 
 * File:   main.cpp
 * Author: peter
 *
 * Created on March 25, 2012, 1:36 AM
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <math.h>
#include "stopWatch.h"

using namespace std;

int n;
int d;
int seed;

#ifdef DIS1
double* distances;
#elif DIS2
double** distances;
#elif DIS3
double* distances;
#elif DIS4
double** distances;
#endif

bool init(char** argv) {

    //Check if arguments are valid
    if (sizeof (argv) != 4) {
        //return false;
    }

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
    return true;
}

double** generateCoordinates() {

    //Use the given seed
    srand(seed);

    //Initialize array to store spheres
    double **xyz = new double* [n];
    for (int i = 0; i < n; i++) {
        xyz[i] = new double[3];
    }

    /**
     * Each sphere has attributes
     * ID - row number
     * x,y,z coordinates - in range [-D:D, -D:D, -D:D]
     * radius - in range [1:D]
     */
    for (int i = 0; i < n; i++) {
        xyz[i][0] = (rand() / double(RAND_MAX))*2 * d - d;
        xyz[i][1] = (rand() / double(RAND_MAX))*2 * d - d;
        xyz[i][2] = (rand() / double(RAND_MAX))*2 * d - d;

    }

    return xyz;
}

double** generateSpheres() {

    //Use the given seed
    srand(seed);

    //Initialize array to store spheres
    double **xyzr = new double* [n];
    for (int i = 0; i < n; i++) {
        xyzr[i] = new double[4];
    }

    /**
     * Each sphere has attributes
     * ID - row number
     * x,y,z coordinates - in range [-D:D, -D:D, -D:D]
     * radius - in range [1:D]
     */
    for (int i = 0; i < n; i++) {
        xyzr[i][0] = (rand() / double(RAND_MAX))*2 * d - d;
        xyzr[i][1] = (rand() / double(RAND_MAX))*2 * d - d;
        xyzr[i][2] = (rand() / double(RAND_MAX))*2 * d - d;
        xyzr[i][3] = (rand() / double(RAND_MAX)) * d;

    }

    return xyzr;
}

double* generateDimmensions() {

    double* ret = new double[n];
    for (int i = 0; i < n; i++) {

        ret[i] = (rand() / double(RAND_MAX))*2 * d - d;

    }
    return ret;
}

double* generateRadiuses() {

    double* r = new double[n];
    for (int i = 0; i < n; i++) {

        r[i] = (rand() / double(RAND_MAX)) * d;

    }
    return r;
}

void initDistances() {
#ifdef DIS1
    distances = new double[n*n];
#elif DIS2
    distances = new double* [n];
    for (int i = 0; i < n; i++) {
        distances[i] = new double[n];
    }
#elif DIS3
    distances = new double[n*n];
#elif DIS4
    distances = new double* [n];
    for (int i = 0; i < n; i++) {
        distances[i] = new double[n];
    }
#endif
}

double& getDistance(int i, int j) {
#ifdef DIS1
    return distances[i * n + j];
#elif DIS2
    return distances[i][j];
#elif DIS3
    return distances[j * n + i];
#elif DIS4
    return distances[j][i];
#endif    
}

void computeDistances(double** xyz) {

    int xd;
    int yd;
    int zd;

    //Compute distances
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {

            xd = fabs(xyz[i][0] - xyz[j][0]);
            yd = fabs(xyz[i][1] - xyz[j][1]);
            zd = fabs(xyz[i][2] - xyz[j][2]);
            getDistance(i, j) = sqrt(xd * xd + yd * yd + zd * zd);

        }
    }

}

void computeDistances(double* x, double* y, double* z) {

    int xd;
    int yd;
    int zd;

    //Compute distances
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {

            xd = fabs(x[i] - x[j]);
            yd = fabs(y[i] - y[j]);
            zd = fabs(z[i] - z[j]);
            getDistance(i, j) = sqrt(xd * xd + yd * yd + zd * zd);

        }
    }

}

void detectCollisions(double* r) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {

            if ((r[i] + r[j]) >= getDistance(i, j)) {
#ifdef PRINT
                cout << "Sphere" << i << " collides with Sphere" << j << endl;
#endif
            }
        }
    }
}

void detectCollisions(double** xyzr) {
    double distance;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {

            distance = getDistance(i, j);
            
            if ((xyzr[i][3] + xyzr[j][3]) >= distance) {
#ifdef PRINT
                cout << "Sphere" << i << " collides with Sphere" << j << endl;
#endif
            }
        }
    }

}

void printArray(double** array, int m, int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(7) << array[i][j] << " ";
        }
        cout << endl;
    }

    cout << endl;
}

void release(double** array) {
    for (int i = 0; i < n; i++) {
        delete []array[i];
    }
    delete []array;

}

void release(double* array) {
    delete []array;
}

/*
 * 
 */
int main(int argc, char** argv) {

    if (!init(argv)) {
        throw "Invalid arguments, arguments are: N, D, seed";
        return 1;
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
#elif DIS2
    cout << "Distances configuration: look-alike + row major" << endl;
#elif DIS3
    cout << "Distances configuration: simulated + column major" << endl;
#elif DIS4
    cout << "Distances configuration: look-alike + column major" << endl;
#endif

    cout << endl;
#endif
    stopWatch* watch = new stopWatch;
    watch->start();

    //Generate spheres coordinates
#ifdef SP1
    double** xyz = generateCoordinates();
    double* r = generateRadiuses();
#elif SP2
    double** xyzr = generateSpheres();
#elif SP3
    double* x = generateDimmensions();
    double* y = generateDimmensions();
    double* z = generateDimmensions();
    double* r = generateRadiuses();
#endif

    //Initialize array to store distances
    initDistances();

    //Compute distances between spheres
#ifdef SP1
    computeDistances(xyz);
#elif SP2
    computeDistances(xyzr);
#elif SP3
    computeDistances(x, y, z);
#endif

    //Detect collisions
#ifdef SP1
    detectCollisions(r);
#elif SP2
    detectCollisions(xyzr);
#elif SP3
    detectCollisions(r);
#endif

    watch->stop();
	cout.setf(ios_base::fixed, ios_base::floatfield);
	cout.precision(6);
    cout << watch->elapsedTime() << endl;

    //Print spheres
#ifdef PRINT
    cout << "Spheres:" << endl;
#ifdef SP1
    printArray(xyz);
    printArray(r);
#elif SP2
    printArray(xyzr);
#elif SP3
    printArray(x);
    printArray(y);
    printArray(z);
    printArray(r);
#endif
#endif

    //Print distances
#ifdef PRINT
    cout << "Distances:" << endl;
    printArray(distances, n, n);
#endif

    //Release
#ifdef SP1
    release(xyz);
    release(r);
#elif SP2
    release(xyzr);
#elif SP3
    release(x);
    release(y);
    release(z);
    release(r);
#endif

#ifdef DIS1
    release(distances);
#elif DIS2
    release(distances);
#elif DIS3
    release(distances);
#elif DIS4
    release(distances);
#endif

    return 0;
}


#include "main1.h"

bool init(char** argv) {

    //Check if arguments are valid
    if (sizeof (argv) < 4) {

	return false;
    
    } else {

    	//Set values from arguments
    	n = atoi(argv[1]);
    	d = atoi(argv[2]);
    	seed = atoi(argv[3]);
        
	cout.setf(ios_base::fixed, ios_base::floatfield);
        cout.precision(6);
#ifdef PRINT
    	cout << "N: " << n << endl;
    	cout << "D: " << d << endl;
    	cout << "seed: " << seed << endl;

#endif

        srand(seed);

        return true;
    }
}

double* generateDimensions() {

    double* ret = new double[n];
    //assert(ret);

    for (int i = 0; i < n; i++) {

        ret[i] = (rand() / double(RAND_MAX))*2 * d - d;

    }
    return ret;
}

double* generateRadiuses() {

    double* ret = new double[n];
    //assert(ret);
    
    for (int i = 0; i < n; i++) {

        ret[i] = (rand() / double(RAND_MAX)) * d;

    }
    return ret;
}

double& getDistance(int i, int j, double* distances) {

    return distances[i * n + j];

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

void printArray(double* array, int m, int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j<n; j++) {
	    cout << setw(7) << array[i*n + j] << " ";
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

    int rank, size;
    int myShare, myBegin, myEnd;

    int allCollision;
    int index;

    double* x;
    double* y;
    double* z;
    double* r;
    double* distances;
    
    double allMinVal;
    double allMaxVal;
    double allAvgVal;

    if (!init(argv)) {
	throw "Invalid arguments, arguments are: N, D, seed";
        return 1;
    }

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    if(rank==0) {
        //Generate spheres coordinates
        x = generateDimensions();
        y = generateDimensions();
        z = generateDimensions();
        r = generateRadiuses();
    }

    

    //Broadcast the size to all nodes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


//Slaves do not have memory yet
    if(rank!=0) {
        x = new double[n];
	y = new double[n];
	z = new double[n];
	r = new double[n];

    }


    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(z, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(r, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);


//Now each node determines which part they work on
    myShare = n / size;
    myBegin = myShare * rank;
    myEnd = myBegin + myShare;
    if (rank == (size - 1)) {
        myEnd = n;
    }

//cout << "myShare " << myShare << " " << myBegin << " " << myEnd << " " << endl;

    time1 = MPI_Wtime();

    distances = new double[myShare*n];

    double xd;
    double yd;
    double zd;
    index = 0;


//Compute distances
    for (int i=myBegin; i<myEnd; i++) {

	for (int j=i+1; j<n; j++) {

            xd = fabs(x[i] - x[j]);
            yd = fabs(y[i] - y[j]);
            zd = fabs(z[i] - z[j]);
	    distances[index] = sqrt(xd * xd + yd * yd + zd * zd);

//cout << "ARRAY_DIST: " << rank << " " << index << " " << i << " " << j << " "<< (sqrt(xd * xd + yd * yd + zd * zd)) << " " << myShare << " " << myBegin << " " << myEnd << " " << size << endl;
     
	    index++;
	}
    }
  
    time2 = MPI_Wtime();

    t1 = time2 - time1;

#ifdef PRINTARR
    //Print spheres
    cout << "Spheres:" << endl;
    printArray(x, n, 1);
    printArray(y, n, 1);
    printArray(z, n, 1);
    printArray(r, n, 1);

    //Print distances
    cout << "Distances:" << endl;
    printArray(distances, n, n);
#endif

    time1 = MPI_Wtime();

    maxVal = 0;
    avgVal = 0;
    minVal = 1000000;

    for (int i = 0; i<index; i++) {
//cout << rank << "DISTANCES==>" << distances[i] << endl;
	if (distances[i]>maxVal) {
	    maxVal = distances[i];
	}
	if (distances[i]<minVal) {
	    minVal = distances[i];
	}
	avgVal = avgVal + distances[i];
    }

    MPI_Reduce(&maxVal, &allMaxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&minVal, &allMinVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avgVal, &allAvgVal, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   
    if (rank==0) {
        allAvgVal = allAvgVal / (n*((n-1)/2.0));
    }

    time2 = MPI_Wtime();

    t2 = time2 - time1;

    time1 = MPI_Wtime();

    collision = 0;

    int colIndex = 0;

//Compute collisions  
    for (int i=myBegin; i<myEnd; i++) {

	for (int j=i+1; j<n; j++) {

             if ((r[i] + r[j]) >= distances[colIndex]) {
                collision++;
            }        
     
	    colIndex++;
	}
    }


    MPI_Reduce(&collision, &allCollision, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    time2 = MPI_Wtime();

    t3 = time2 - time1;

    if (rank==0) {
        cout << size << ", " << t1 << ", " << t2 << ", " << t3 << ", " << allMaxVal << ", " << allMinVal << ", " << allAvgVal << ", " << allCollision << ", " << endl;

    }

    MPI_Finalize();

    //Release
    release(x);
    release(y);
    release(z);
    release(r);

    release(distances);

    return 0;
}


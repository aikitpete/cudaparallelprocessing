#include "main3.h"

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
    //	cout << "N: " << n << endl;
    //	cout << "D: " << d << endl;
    //	cout << "seed: " << seed << endl;
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

void printArray(double* array, int m, int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j<n; j++) {
	    cout << setw(7) << array[i*n + j] << " ";
	}
    }
    cout << endl;
}

void printArray(string str, int r, double* array, int m, int n) {
    stringstream ss(stringstream::in | stringstream::out);
    ss << str << " " << r << " ";
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j<n; j++) {
	     
	    ss << array[i*n + j] << " ";
        }
    }
    cout << ss.str() << endl;
}

void release(double* array) {
    delete []array;
}

void compute(double* xLocal, double* yLocal, double* zLocal, double* rLocal, double* xLocalA, double*  yLocalA, double* zLocalA, double* rLocalA, double* distances, int rows, int rowsA, bool isSame, int rank) {

#ifdef PRINTARR
//cout << "RANK ROW ROWA" << rank << " " << rows << " " << rowsA << endl;
//printArray("Array ", rank, rLocal, rows, 1);
//printArray("ArrayA ", rank, rLocalA, rowsA, 1);
#endif

#ifndef COMPUTE

double xd;
double yd;
double zd;
int index = 0;

//Compute distances
for (int i=0; i<rows; i++) {
 
    for (int j=i; j<rowsA; j++) {

        if (isSame && i==j) {
            
            continue;

        } else {
            xd = fabs(xLocal[i] - xLocalA[j]);
            yd = fabs(yLocal[i] - yLocalA[j]);
            zd = fabs(zLocal[i] - zLocalA[j]);
	    distances[index] = sqrt(xd * xd + yd * yd + zd * zd);

            index++;
	}
    }	
}



#ifdef PRINTARR
//cout << "RANK ROW ROWA END" << rank << " " << rows << " " << rowsA << endl;
#endif

#ifdef PROC
//}
//void evalDistances (double* rLocal, double* rLocalA, double* distances, int rows, int rowsA, bool isSame) {
#endif

#ifdef PRINTARR
//cout << "RANK ROW ROWA ISSAME " << rank << " " << rows << " " << rowsA << " " << isSame << endl;
#endif

int i;
int j;
int colIndex;

colIndex = 0;

for (i = 0; i<rows; i++) {
    for (j = i; j<rowsA; j++) {

//cout << rank << "DISTANCES==>" << distances[i] << " " << i << "/" << index << endl;
	
	if (isSame && i==j) {
	    
	    continue;
	
	} else {
	
	    if (distances[colIndex]>maxVal) {
	        maxVal = distances[colIndex];
	    }
	    if (distances[colIndex]<minVal) {
	        minVal = distances[colIndex];
	    }
	    avgVal = avgVal + distances[colIndex];
        
	}

	colIndex++;
    }
}

colIndex = 0;

for (i=0; i<rows; i++) {
    for (j=i; j<rowsA; j++) {
    
        if (isSame && i==j) {
	    
	    continue;

        } else {
            
	    if ((rLocal[i] + rLocalA[j]) >= distances[colIndex]) {
                collision++;
	    }

	    colIndex++;
        }
    }

}

#endif

}

/*
 * 
 */
int main(int argc, char** argv) {
        
    maxVal = -1;
    avgVal = 0;
    minVal = 10000;
    collision = 0;

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

    //Determine how work is going to be distributed
    //int work;
    //int count[n], displ[n];
    //int i;
    
    //int* count = new int[size]; 
    //int* displ = new int[size];
  
    //if (rank==0) {
    //    work = n / size;
    //    for (i=0; i<size-1; i++) {
    //	    count[i] = work;
    //    }
    //	  count[i] = n - (size-1) * work;
    //}

    //Let everyone know their problem size
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    
    int mySize;

    if (n%size==0) {
	mySize = n/size;
    } else {
        mySize = n/size + 1;
    }

#ifdef PRINTARR
cout << "RANK: " << rank << " MYSIZE: " << mySize << endl;
#endif

    int mySizeLocal;

    if (n-(mySize*rank)>=mySize) {
	mySizeLocal = mySize;
    } else if (n-(mySize*rank)==(n%mySize)) {
        mySizeLocal = n%mySize;
    } else {
        mySizeLocal = 0;
    }

#ifdef PRINTARR
cout << "RANK: " << rank << " MYSIZELOCAL: " << mySizeLocal << endl;
#endif

    int distSize;
    
    //TODO reduce size
    distSize = (mySize*mySize) - mySize*((mySize-1)/2.0);

    xLocal = new double[mySize];
    yLocal = new double[mySize];
    zLocal = new double[mySize];
    rLocal = new double[mySize];

    xLocalTmp = new double[mySize];
    yLocalTmp = new double[mySize];
    zLocalTmp = new double[mySize];
    rLocalTmp = new double[mySize];
    
    xLocalB = new double[mySize];
    yLocalB = new double[mySize];
    zLocalB = new double[mySize];
    rLocalB = new double[mySize];

    distances = new double[distSize];

//Scatter sphere data
MPI_Scatter(x,mySize,MPI_DOUBLE,xLocal,mySize,MPI_DOUBLE,0,MPI_COMM_WORLD);
MPI_Scatter(y,mySize,MPI_DOUBLE,yLocal,mySize,MPI_DOUBLE,0,MPI_COMM_WORLD);
MPI_Scatter(z,mySize,MPI_DOUBLE,zLocal,mySize,MPI_DOUBLE,0,MPI_COMM_WORLD);
MPI_Scatter(r,mySize,MPI_DOUBLE,rLocal,mySize,MPI_DOUBLE,0,MPI_COMM_WORLD);

#ifndef LOOP

    time1 = MPI_Wtime();

    int tag = 100;
    int to, from;

    MPI_Request reqX[2];
    //MPI_Status statX[2];
    MPI_Request reqY[2];
    //MPI_Status statY[2];
    MPI_Request reqZ[2];
    //MPI_Status statZ[2];
    MPI_Request reqR[2];
    //MPI_Status statR[2];

    to = (rank + 1) % size;
    if (rank != 0) {
        from = (rank - 1);
    } else {
        from = size - 1;
    }
	
compute(xLocal, yLocal, zLocal, rLocal, xLocal, yLocal, zLocal, rLocal, distances, mySizeLocal, mySizeLocal, true, rank);

#ifdef PRINTARR
MPI_Barrier(MPI_COMM_WORLD);
cout << "RANK: " << rank << " MaxVal: " << maxVal << endl;
MPI_Barrier(MPI_COMM_WORLD);
cout << "RANK: " << rank << " AvgVal: " << avgVal << endl;
MPI_Barrier(MPI_COMM_WORLD);
cout << "RANK: " << rank << " MinVal: " << minVal << endl;
MPI_Barrier(MPI_COMM_WORLD);
#endif

int currentRank = rank;
int mySizeA = mySizeLocal;
int mySizeB;

#ifdef PROC
int mySizeAold;
mySizeAold = mySizeA;
#endif

xLocalA = xLocal;
yLocalA = yLocal;
zLocalA = zLocal;
rLocalA = rLocal;

#ifdef PRINTARR
    ////MPI_Barrier(MPI_COMM_WORLD);
    ////printArray("xA", rank, xLocalA, mySizeA, 1);
    ////MPI_Barrier(MPI_COMM_WORLD);
    //printArray("yA", rank, yLocalA, mySizeA, 1);
    //MPI_Barrier(MPI_COMM_WORLD);
    //printArray("zA", rank, zLocalA, mySizeA, 1);
    //MPI_Barrier(MPI_COMM_WORLD);
    //printArray("rA", rank, rLocalA, mySizeA, 1);
    //MPI_Barrier(MPI_COMM_WORLD);
#endif

//LOOP P-1
#ifdef PRINTARR
MPI_Barrier(MPI_COMM_WORLD);
if (rank==0) cout << "##START LOOP##" << endl; 
MPI_Barrier(MPI_COMM_WORLD);
#endif
for (int i=0; i<size-1; i++) {

currentRank = (currentRank!=0)?(currentRank-1):(size-1);

#ifdef PRINTARR
//cout << "RANK: " << rank << " CURENT: " << currentRank << " LOOP: " << i+1 << endl;
#endif

if (n-(mySize*currentRank)>=mySize) {
    mySizeB = mySize;
} else if (n-(mySize*currentRank)>=0) {
    mySizeB = n%mySize;
} else {
    mySizeB = 0;
}

#ifdef PRINTARR
cout << "LOOP " << i+1 << "RANK " << rank << " MSA: " << mySizeA << endl;
cout << "LOOP " << i+1 << "RANK " << rank << " MSB: " << mySizeB << endl;
#endif

MPI_Isend(xLocalA,mySizeA,MPI_DOUBLE,to,tag,MPI_COMM_WORLD,&(reqX[0]));
MPI_Isend(yLocalA,mySizeA,MPI_DOUBLE,to,tag,MPI_COMM_WORLD,&(reqY[0]));
MPI_Isend(zLocalA,mySizeA,MPI_DOUBLE,to,tag,MPI_COMM_WORLD,&(reqZ[0]));
MPI_Isend(rLocalA,mySizeA,MPI_DOUBLE,to,tag,MPI_COMM_WORLD,&(reqR[0]));

MPI_Irecv(xLocalB,mySizeB,MPI_DOUBLE,from,tag,MPI_COMM_WORLD,&(reqX[1]));
MPI_Irecv(yLocalB,mySizeB,MPI_DOUBLE,from,tag,MPI_COMM_WORLD,&(reqY[1]));
MPI_Irecv(zLocalB,mySizeB,MPI_DOUBLE,from,tag,MPI_COMM_WORLD,&(reqZ[1]));
MPI_Irecv(rLocalB,mySizeB,MPI_DOUBLE,from,tag,MPI_COMM_WORLD,&(reqR[1]));

#ifdef PROC
//evalDistances(rLocal, rLocalB, distances, mySizeLocal, mySizeB, (rank<currentRank-1)?1:0);
#endif

//Waits for all MPI requests to complete
MPI_Waitall(2, reqX, MPI_STATUSES_IGNORE);
MPI_Waitall(2, reqY, MPI_STATUSES_IGNORE);
MPI_Waitall(2, reqZ, MPI_STATUSES_IGNORE);
MPI_Waitall(2, reqR, MPI_STATUSES_IGNORE);

#ifdef PRINTARR
//printArray("xB", rank , xLocalB, mySizeB, 1);   
#endif

//EVALUATE
if (rank<currentRank) {
compute(xLocal, yLocal, zLocal, rLocal, xLocalB, yLocalB, zLocalB, rLocalB, distances, mySizeLocal, mySizeB, false, rank);
} else {
compute(xLocal, yLocal, zLocal, rLocal, xLocalB, yLocalB, zLocalB, rLocalB, distances, mySizeLocal, mySizeB, true, rank);
}

if (i!=0) {
xLocalTmp = xLocalA;
yLocalTmp = yLocalA;
zLocalTmp = zLocalA;
rLocalTmp = rLocalA;
}

xLocalA = xLocalB;
yLocalA = yLocalB;
zLocalA = zLocalB;
rLocalA = rLocalB;

xLocalB = xLocalTmp;
yLocalB = yLocalTmp;
zLocalB = zLocalTmp;
rLocalB = rLocalTmp;

#ifdef PROC
//mySizeAold = mySizeA;
#endif

mySizeA = mySizeB;

}

#endif

#ifdef PROC
//currentRank = (currentRank!=0)?(currentRank-1):(size-1);
//evalDistances(rLocal, rLocalA, distances, mySizeLocal, mySizeA, (rank<currentRank)?1:0);
#endif

#ifndef EVAL
//GET RESULT
MPI_Reduce(&maxVal,&maxDistance,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
MPI_Reduce(&minVal,&minDistance,1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
MPI_Reduce(&avgVal,&avgDistance,1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
if (rank==0) {
    avgDistance = avgDistance / (n*((n-1)/2.0));
}

MPI_Reduce(&collision, &allCollision, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

time2 = MPI_Wtime();

t1 = time2 - time1;

if (rank==0) {
cout << size << ", " << t1 << ", " << maxDistance << ", " << minDistance << ", " << avgDistance << ", " << allCollision << ", " << endl;
}

#endif

MPI_Finalize();

//Release
release(x);
release(y);
release(z);
release(r);

#ifndef LOOP
release(distances);
#endif

return 0;

}

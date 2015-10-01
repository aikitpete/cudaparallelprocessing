/* 
 * File:   Timer.cpp
 * Author: peter
 * 
 * Created on March 12, 2012, 12:44 PM
 */

#include "stopWatch.h"
#include <iostream>
using namespace std;

stopWatch::stopWatch() {
    
}

/**
 * Marks end time
 */
void stopWatch::stop() {
    gettimeofday(&t2, NULL);
    stopTime = t2.tv_sec + (t2.tv_usec / 1000000.0);
}

/**
 * Marks start time
 */
void stopWatch::start() {
    gettimeofday(&t1, NULL);
    startTime = t1.tv_sec + (t1.tv_usec / 1000000.0);
}

double stopWatch::elapsedTime() {
    return stopTime - startTime;
}

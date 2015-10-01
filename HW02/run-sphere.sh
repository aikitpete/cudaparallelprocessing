#!/bin/bash

N=(10 50 100 200 500 1000)

D=500;
seed=1;

make

files=(./spheres1 ./spheres2 ./spheres3)

for size in {0..5}
do

  for i in {0..2}
  do

    duration=0;

    for run in {1..300}
    do
      time=`${files[i]} ${N[size]} $D $seed | grep Elapsed | cut -d' ' -f 3 | bc`;
      duration=`echo "$duration + $time" | bc`;
      #echo $duration
    done

    echo "${N[size]} ${files[i]} $duration"

  done

done

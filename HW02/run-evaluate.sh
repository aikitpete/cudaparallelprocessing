#!/bin/bash

N=(10 50 100 200 500 1000 2000 5000 10000);

D=10;
seed=1;

make

files=(./spheres1 ./spheres2 ./spheres3 ./distances1 ./distances2 ./distances3 ./distances4);

for size in $N
do

    for file in $files
    do

      duration=0;

      for run in {1..100}
      do
	    
	time=`$file $size $D $seed | grep Elapsed | cut -d' ' -f3 | bc`;
        duration=`echo "$duration + $time" | bc`;

      done

      echo `$size $file $duration`

    done

done

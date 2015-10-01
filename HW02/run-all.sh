#!/bin/bash

N=50;
D=150;
seed=1;

make

files=(./spheres1 ./spheres2 ./spheres3 ./distances1 ./distances2 ./distances3 ./distances4)

for i in {0..6}
do
  "${files[i]} $N $D $seed"
done

#!/bin/bash

N=10;
D=10;
seed=1;

make

files=(./spheres1 ./spheres2 ./spheres3)

for i in {0..3}
do
  "${files[i]} $N $D $seed"
done

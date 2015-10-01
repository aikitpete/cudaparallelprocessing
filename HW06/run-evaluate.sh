#!/bin/bash

N='10 50 100 200 500 1000 2000 5000'

D=10;
seed=1;

P='spheres1 spheres2 spheres3 spheres1old spheres2old spheres3old'

rm slurm/*

make all

for f in $P
do
for i in $N
do

  rm slurm/slurm-${f}-np$i.out
  rm slurm/slurm-${f}old-np$i.out
  

  duration=0;

  #for run in {1..10}
  #do
    
    ./$f $i $D $seed | tail >> slurm/slurm-${f}.out

    ./${f}old $i $D $seed | tail >> slurm/slurm-${f}old.out
    
  #done

done
done

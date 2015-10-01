#!/usr/bin/gnuplot

set terminal png size 800,600
set output "plots/np500-t1.png"

set key top left
set xlabel "No. Processors"
set xrange [2:16]
set ylabel "Speed up"
set yrange [2:16]
set y2label "Parallel efficiency"
set y2range [0:2]

a=`head -n1 slurm/slurm-program2-np500.out | tr -d "," | cut -f2 -d" "`
b=`head -n1 slurm/slurm-program3-np500.out | tr -d "," | cut -f2 -d" "`

set style line 1 lw 3 lc rgb "black"
set style line 2 lt 1 lw 2 lc rgb "red"
set style line 3 pt 1 lc rgb "red"
set style line 4 pt 2 lw 2 lc rgb "blue"
set style line 5 pt 2 lw 2 lc rgb "blue"

plot \
\
x title "Perfect speedup" ls 1,  \
\
"< sed 's/,//g' slurm/slurm-program2-np500.out" using 1:(a/$2) title "Speedup, Program2" ls 3, \
\
"< sed 's/,//g' slurm/slurm-program3-np500.out" using 1:(b/$2) title "Speedup, Program3" ls 4 , \
\
"< sed 's/,//g' slurm/slurm-program2-np500.out" using 1:(a/$2/$1) axis x1y2 title "Parallel efficiency, Program2" ls 2 with lines, \
\
"< sed 's/,//g' slurm/slurm-program3-np500.out" using 1:(b/$2/$1) axis x1y2 title "Parallel efficiency, Program3" ls 5 with lines


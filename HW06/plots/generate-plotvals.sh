#!/bin/bash
cat plots/plotvals-np1000-t1.sh | sed s/1000/10/ > plots/plotvals-np10-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/50/ > plots/plotvals-np50-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/100/ > plots/plotvals-np100-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/200/ > plots/plotvals-np200-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/500/ > plots/plotvals-np500-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/1000/ > plots/plotvals-np1000-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/2000/ > plots/plotvals-np2000-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/5000/ > plots/plotvals-np5000-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/10000/ > plots/plotvals-np10000-t1.sh
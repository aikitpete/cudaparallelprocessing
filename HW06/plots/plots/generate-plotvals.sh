#!/bin/bash
cat plots/plotvals-np1000-t1.sh | sed s/1000/500/ > plots/plotvals-np500-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/5000/ > plots/plotvals-np5000-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/10000/ > plots/plotvals-np10000-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/15000/ > plots/plotvals-np15000-t1.sh
cat plots/plotvals-np1000-t1.sh | sed s/1000/20000/ > plots/plotvals-np20000-t1.sh

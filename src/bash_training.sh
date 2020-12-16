#!/bin/sh
for i in 1 2 3 4 5 6 7 8 9 10
do
	echo $i
	python training.py supervised 0.001 16 0.0001 0.5 0 $i
done

#!/bin/bash
for l in 2 4 8 16 32
do
	for i in {1..23}
do
	echo $i
	python training.py vae_sup_chb 0.00001 $l 0.0001 0.02 0 $i
	done
done

#!/bin/bash
for i in {1..23}
do
	echo $i
	python training.py vae_sup_chb 0.00001 4 0.0001 0.02 0 $i
done

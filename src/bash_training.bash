#!/bin/bash
for i in {1..24}
do
	echo $i
	python training.py vae_sup_chb 0.00001 32 0.0001 0.02 0 $i
done

#!/bin/bash
for i in {1..24}
do
	echo $i
	python training.py ae_unsupervised_norm 0.00001 16 0.00001 0.5 0 $i
done
